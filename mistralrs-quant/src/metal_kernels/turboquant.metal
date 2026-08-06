#include <metal_stdlib>
using namespace metal;

// ============================================================================
// TurboQuant fused encode kernel (PolarQuant + QJL).
//
// One threadgroup per vector. Each threadgroup processes one [head_dim] slice
// of the input tensor through:
//
//   1. rotate:   y = R · x                       (R is head_dim × head_dim, fp32)
//   2. polar:    radius = ||y||_2;
//                code_i = bin(y_i / radius) ∈ {0,1,2,3}
//   3. recon:    rec_i  = centroid[code_i] * (radius / ||rec||_2)
//   4. residual: res = y - rec
//   5. QJL:      sign_i = sign(P_i · res), i ∈ [0, proj_dim)
//                qjl_j  = (||res||_2 / sqrt(proj_dim)) * Σ_i sign_i · P_{i,j}
//   6. compose:  final_rotated = rec + qjl
//   7. unrotate: out = R^T · final_rotated
//   8. write:    out cast back to input dtype
//
// Output is **lossy reconstruction** in the input dtype — same contract as the
// pure-Rust CandleTurboQuantCodec it replaces. The codec is "lossy within
// fp16 storage": the cache still stores fp16, but the values written are the
// PolarQuant + QJL reconstruction of the originals.
//
// Correctness contract: this kernel must produce values matching the pure-Rust
// reference in `vibecody/vibeui/crates/vibe-infer/src/kv_cache_tq.rs` to
// within fp16 rounding (~1e-3 abs diff per element). See the parity test in
// kv_cache_codec.rs.
//
// Performance: this is a correctness-first implementation. Threadgroup size
// equals head_dim (typical 64–256). R and P are read from device memory each
// time (no shared-mem caching). Tuning passes — coalesced loads, vectorized
// reads, register tiling — happen after parity is proven.
// ============================================================================

// PolarQuant centroids matching the pure-Rust reference's CENTROIDS table.
constant float POLAR_CENTROIDS[4] = {-0.75f, -0.25f, 0.25f, 0.75f};

// Threshold cuts for polar bin assignment, matching PolarCode::encode:
//   unit < -0.5 → 0
//   unit <  0.0 → 1
//   unit <  0.5 → 2
//   else        → 3

// SIMD group size on Apple GPUs (always 32 — verify on target via
// device.simd_group_size if you ever target an oddball).
constant uint TQ_SIMD_SIZE = 32u;

// Cross-simdgroup reduction helper. Reduces `local` (one float per thread) to
// a single float visible to all threads via shared memory. `scratch` must
// have at least `ceil(threads_per_threadgroup / TQ_SIMD_SIZE)` floats.
inline float threadgroup_reduce_sum(float local_val, threadgroup float *scratch,
                                    uint tid, uint sgitg, uint tiisg,
                                    uint ntg) {
  // Step 1: simdgroup reduce.
  for (uint offset = TQ_SIMD_SIZE / 2; offset > 0; offset /= 2) {
    local_val += simd_shuffle_xor(local_val, offset);
  }
  // Step 2: lane 0 of each simdgroup writes to scratch.
  if (tiisg == 0) {
    scratch[sgitg] = local_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Step 3: first simdgroup reduces the scratch.
  const uint num_simdgroups = (ntg + TQ_SIMD_SIZE - 1) / TQ_SIMD_SIZE;
  if (tid < TQ_SIMD_SIZE) {
    float v = (tid < num_simdgroups) ? scratch[tid] : 0.0f;
    for (uint offset = TQ_SIMD_SIZE / 2; offset > 0; offset /= 2) {
      v += simd_shuffle_xor(v, offset);
    }
    if (tid == 0) {
      scratch[0] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

template <typename T>
[[kernel]] void turboquant_encode(
    const device T *input [[buffer(0)]],            // [num_vectors * head_dim]
    const device float *rotation [[buffer(1)]],     // [head_dim * head_dim]
    const device float *projection [[buffer(2)]],   // [proj_dim * head_dim]
    device T *output [[buffer(3)]],                 // [num_vectors * head_dim]
    constant uint &num_vectors [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &proj_dim [[buffer(6)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint ntg [[threads_per_threadgroup]]) {

  if (tgpig >= num_vectors) {
    return;
  }
  const uint tid = tpitg;
  const uint vec_offset = tgpig * head_dim;
  const device T *x = input + vec_offset;
  device T *out = output + vec_offset;

  // Shared memory layout (in floats):
  //   [0 .. head_dim)             : y (rotated input)
  //   [head_dim .. 2*head_dim)    : rec (polar reconstruction)
  //   [2*head_dim .. 3*head_dim)  : residual = y - rec
  //   [3*head_dim .. 4*head_dim)  : final_rotated = rec + qjl_correction
  //   [4*head_dim ..)             : reduction scratch (max ntg/TQ_SIMD_SIZE
  //                                  + a small slack)
  threadgroup float *sh_y = shared_mem;
  threadgroup float *sh_rec = sh_y + head_dim;
  threadgroup float *sh_res = sh_rec + head_dim;
  threadgroup float *sh_final = sh_res + head_dim;
  threadgroup float *sh_scratch = sh_final + head_dim;

  // ── Step 1: rotate. y[t] = Σ_j R[t, j] * x[j] for t in [0, head_dim).
  //   With one thread per dim, thread t computes y[t]. Loop in case
  //   head_dim > ntg (uncommon — head_dim is typically 64 or 128).
  for (uint t = tid; t < head_dim; t += ntg) {
    float acc = 0.0f;
    for (uint j = 0; j < head_dim; ++j) {
      acc += rotation[t * head_dim + j] * float(x[j]);
    }
    sh_y[t] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Step 2a: radius = ||y||_2.
  float local_sq = 0.0f;
  for (uint t = tid; t < head_dim; t += ntg) {
    float v = sh_y[t];
    local_sq += v * v;
  }
  float radius_sq = threadgroup_reduce_sum(local_sq, sh_scratch, tid, sgitg,
                                           tiisg, ntg);
  float radius = sqrt(radius_sq);

  // Tiny-radius guard: matches PolarCode::encode short-circuit. When the
  // input vector is all zero, output is zero too — write directly and exit.
  if (radius < 1e-10f) {
    for (uint t = tid; t < head_dim; t += ntg) {
      out[t] = T(0);
    }
    return;
  }

  // ── Step 2b: polar bin per dim, then ── Step 3: reconstruct unit vector.
  //   We don't need the codes themselves after this kernel (output is fp16
  //   reconstruction, not packed bits) — go straight from y[t] to rec[t].
  float local_rec_sq = 0.0f;
  for (uint t = tid; t < head_dim; t += ntg) {
    float unit = sh_y[t] / radius;
    float rec;
    if (unit < -0.5f) {
      rec = POLAR_CENTROIDS[0];
    } else if (unit < 0.0f) {
      rec = POLAR_CENTROIDS[1];
    } else if (unit < 0.5f) {
      rec = POLAR_CENTROIDS[2];
    } else {
      rec = POLAR_CENTROIDS[3];
    }
    sh_rec[t] = rec;
    local_rec_sq += rec * rec;
  }
  float rec_norm_sq =
      threadgroup_reduce_sum(local_rec_sq, sh_scratch, tid, sgitg, tiisg, ntg);
  float rec_norm = sqrt(max(rec_norm_sq, 1e-20f));
  float rec_scale = radius / rec_norm;

  // Apply scale and compute residual = y - rec_scaled.
  for (uint t = tid; t < head_dim; t += ntg) {
    float scaled_rec = sh_rec[t] * rec_scale;
    sh_rec[t] = scaled_rec;
    sh_res[t] = sh_y[t] - scaled_rec;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Step 4: ||residual||_2.
  float local_res_sq = 0.0f;
  for (uint t = tid; t < head_dim; t += ntg) {
    float r = sh_res[t];
    local_res_sq += r * r;
  }
  float res_norm_sq = threadgroup_reduce_sum(local_res_sq, sh_scratch, tid,
                                             sgitg, tiisg, ntg);
  float residual_norm = sqrt(res_norm_sq);

  // ── Step 5+6: QJL encode-then-decode-add directly into sh_final.
  //   final_rotated[j] = rec[j] + (||res||_2 / sqrt(proj_dim)) *
  //                      Σ_i sign(P_i · res) * P[i, j]
  //
  //   We don't store the sign codes — we accumulate the contribution per
  //   output dim j directly. Per thread t (handling output dim t), loop over
  //   all proj_dim rows of P, recompute each row's dot with residual to get
  //   the sign, then add sign * scale * P[i, t].
  //
  //   This costs O(proj_dim * head_dim) FLOPs per thread = O(proj_dim *
  //   head_dim^2) total — same as the CPU reference. For head_dim=128 and
  //   proj_dim=128 that's 16K FLOPs per thread, fine.
  //
  //   Optimization for later: cooperatively compute and store the proj_dim
  //   signs in shared memory once (~proj_dim * head_dim FLOPs total via
  //   threadgroup reduction), then decode-add (~proj_dim * head_dim FLOPs).
  //   That cuts work from O(proj_dim * head_dim^2) to O(2 * proj_dim *
  //   head_dim). Skipped here for first-pass simplicity; correctness first.
  if (residual_norm < 1e-10f) {
    // Residual collapsed; QJL contribution is zero. final = rec.
    for (uint t = tid; t < head_dim; t += ntg) {
      sh_final[t] = sh_rec[t];
    }
  } else {
    float qjl_scale = residual_norm / sqrt(float(proj_dim));
    for (uint t = tid; t < head_dim; t += ntg) {
      float qjl_acc = 0.0f;
      for (uint i = 0; i < proj_dim; ++i) {
        // Recompute sign of P_i · res (no shared storage of signs in this
        // first pass — see optimization note above).
        float dot = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
          dot += projection[i * head_dim + j] * sh_res[j];
        }
        float sign_i = (dot >= 0.0f) ? 1.0f : -1.0f;
        qjl_acc += sign_i * projection[i * head_dim + t];
      }
      sh_final[t] = sh_rec[t] + qjl_scale * qjl_acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Step 7: inverse rotate. out[t] = Σ_j R[j, t] * final_rotated[j].
  //   Column access into row-major R; uncoalesced but small. Optimize later.
  for (uint t = tid; t < head_dim; t += ntg) {
    float acc = 0.0f;
    for (uint j = 0; j < head_dim; ++j) {
      acc += rotation[j * head_dim + t] * sh_final[j];
    }
    out[t] = T(acc);
  }
}

// ============================================================================
// Kernel instantiations. Output dtype matches input.
// ============================================================================

#define instantiate_turboquant_encode(type)                                    \
  template [[host_name("turboquant_encode_" #type)]] [[kernel]] void           \
  turboquant_encode<type>(                                                     \
      const device type *input [[buffer(0)]],                                  \
      const device float *rotation [[buffer(1)]],                              \
      const device float *projection [[buffer(2)]],                            \
      device type *output [[buffer(3)]],                                       \
      constant uint &num_vectors [[buffer(4)]],                                \
      constant uint &head_dim [[buffer(5)]],                                   \
      constant uint &proj_dim [[buffer(6)]],                                   \
      threadgroup float *shared_mem [[threadgroup(0)]],                        \
      uint tgpig [[threadgroup_position_in_grid]],                             \
      uint tpitg [[thread_position_in_threadgroup]],                           \
      uint sgitg [[simdgroup_index_in_threadgroup]],                           \
      uint tiisg [[thread_index_in_simdgroup]],                                \
      uint ntg [[threads_per_threadgroup]]);

instantiate_turboquant_encode(float);
instantiate_turboquant_encode(half);
// bfloat16_t intentionally omitted for the first pass — KV caches in
// mistral.rs are typically fp16/fp32, and bfloat16 needs the bf16.metal
// helpers. Add when a model that uses bf16 KV demands it.
