// ============================================================================
// TurboQuant fused encode kernel (PolarQuant + QJL) — CUDA implementation.
//
// Mirrors the Metal kernel at
//   mistralrs-quant/src/metal_kernels/turboquant.metal
// and the pure-Rust reference at
//   vibecody/vibeui/crates/vibe-infer/src/kv_cache_tq.rs
//
// One block per vector, head_dim threads per block. Each block processes one
// [head_dim] slice through:
//   1. rotate:   y = R · x                       (R is head_dim × head_dim)
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
// Output is **lossy reconstruction** in the input dtype — same contract as
// the pure-Rust CandleTurboQuantCodec it replaces.
//
// Performance: correctness-first. Block size = head_dim (typical 64–256), R
// and P are read from global memory each time. Tuning passes — coalesced
// loads, vectorized reads, register tiling — happen after parity is proven.
// ============================================================================

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

// PolarQuant centroids matching the pure-Rust reference's CENTROIDS table.
__device__ __constant__ float POLAR_CENTROIDS[4] = {-0.75f, -0.25f, 0.25f,
                                                    0.75f};

// Block-wide reduction via shared memory. `scratch` must hold at least
// blockDim.x floats (we shrink to a power-of-two before the tree reduce).
__device__ __forceinline__ float block_reduce_sum(float val, float *scratch) {
  const unsigned tid = threadIdx.x;
  const unsigned ntg = blockDim.x;
  scratch[tid] = val;
  __syncthreads();
  // Tree reduce; works for any ntg up to 1024.
  for (unsigned offset = ntg / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] += scratch[tid + offset];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename T>
__global__ void turboquant_encode_kernel(const T *__restrict__ input,
                                         const float *__restrict__ rotation,
                                         const float *__restrict__ projection,
                                         T *__restrict__ output,
                                         uint32_t num_vectors,
                                         uint32_t head_dim,
                                         uint32_t proj_dim) {
  const uint32_t vec_idx = blockIdx.x;
  if (vec_idx >= num_vectors) {
    return;
  }
  const uint32_t tid = threadIdx.x;
  const uint32_t ntg = blockDim.x;
  const uint32_t vec_offset = vec_idx * head_dim;

  // Shared memory layout (in floats):
  //   [0 .. head_dim)             : y (rotated input)
  //   [head_dim .. 2*head_dim)    : rec (polar reconstruction, then scaled)
  //   [2*head_dim .. 3*head_dim)  : residual = y - rec_scaled
  //   [3*head_dim .. 4*head_dim)  : final_rotated = rec_scaled + qjl
  //   [4*head_dim .. 4*head_dim + ntg) : reduction scratch
  extern __shared__ float shared_mem[];
  float *sh_y = shared_mem;
  float *sh_rec = sh_y + head_dim;
  float *sh_res = sh_rec + head_dim;
  float *sh_final = sh_res + head_dim;
  float *sh_scratch = sh_final + head_dim;

  const T *x = input + vec_offset;
  T *out = output + vec_offset;

  // ── Step 1: rotate. y[t] = Σ_j R[t, j] * x[j].
  for (uint32_t t = tid; t < head_dim; t += ntg) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < head_dim; ++j) {
      acc += rotation[t * head_dim + j] * static_cast<float>(x[j]);
    }
    sh_y[t] = acc;
  }
  __syncthreads();

  // ── Step 2a: radius = ||y||_2.
  float local_sq = 0.0f;
  for (uint32_t t = tid; t < head_dim; t += ntg) {
    float v = sh_y[t];
    local_sq += v * v;
  }
  float radius_sq = block_reduce_sum(local_sq, sh_scratch);
  float radius = sqrtf(radius_sq);

  // Tiny-radius guard: matches PolarCode::encode short-circuit.
  if (radius < 1e-10f) {
    for (uint32_t t = tid; t < head_dim; t += ntg) {
      out[t] = static_cast<T>(0.0f);
    }
    return;
  }

  // ── Step 2b: polar bin per dim, then ── Step 3: reconstruct unit vector.
  float local_rec_sq = 0.0f;
  for (uint32_t t = tid; t < head_dim; t += ntg) {
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
  float rec_norm_sq = block_reduce_sum(local_rec_sq, sh_scratch);
  float rec_norm = sqrtf(fmaxf(rec_norm_sq, 1e-20f));
  float rec_scale = radius / rec_norm;

  // Apply scale and compute residual = y - rec_scaled.
  for (uint32_t t = tid; t < head_dim; t += ntg) {
    float scaled_rec = sh_rec[t] * rec_scale;
    sh_rec[t] = scaled_rec;
    sh_res[t] = sh_y[t] - scaled_rec;
  }
  __syncthreads();

  // ── Step 4: ||residual||_2.
  float local_res_sq = 0.0f;
  for (uint32_t t = tid; t < head_dim; t += ntg) {
    float r = sh_res[t];
    local_res_sq += r * r;
  }
  float res_norm_sq = block_reduce_sum(local_res_sq, sh_scratch);
  float residual_norm = sqrtf(res_norm_sq);

  // ── Step 5+6: QJL encode-then-decode-add directly into sh_final.
  //   final_rotated[j] = rec[j] + (||res||_2 / sqrt(proj_dim)) *
  //                      Σ_i sign(P_i · res) * P[i, j]
  //
  //   Each thread t handles output dim t; recomputes the proj_dim signs by
  //   re-scanning P against sh_res. O(proj_dim * head_dim) FLOPs per thread.
  //   See the Metal kernel for the optimization note (cooperative sign
  //   storage cuts this to O(proj_dim * head_dim) total).
  if (residual_norm < 1e-10f) {
    for (uint32_t t = tid; t < head_dim; t += ntg) {
      sh_final[t] = sh_rec[t];
    }
  } else {
    float qjl_scale = residual_norm / sqrtf(static_cast<float>(proj_dim));
    for (uint32_t t = tid; t < head_dim; t += ntg) {
      float qjl_acc = 0.0f;
      for (uint32_t i = 0; i < proj_dim; ++i) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < head_dim; ++j) {
          dot += projection[i * head_dim + j] * sh_res[j];
        }
        float sign_i = (dot >= 0.0f) ? 1.0f : -1.0f;
        qjl_acc += sign_i * projection[i * head_dim + t];
      }
      sh_final[t] = sh_rec[t] + qjl_scale * qjl_acc;
    }
  }
  __syncthreads();

  // ── Step 7: inverse rotate. out[t] = Σ_j R[j, t] * final_rotated[j].
  for (uint32_t t = tid; t < head_dim; t += ntg) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < head_dim; ++j) {
      acc += rotation[j * head_dim + t] * sh_final[j];
    }
    out[t] = static_cast<T>(acc);
  }
}

// ============================================================================
// Launchers (extern "C", one per dtype). Caller picks block size = head_dim.
// Shared memory size = (4 * head_dim + block_size) floats — enough for the
// four working buffers plus the reduction scratch.
// ============================================================================

static inline uint32_t pick_block_size(uint32_t head_dim) {
  // Round up to a power of two for the tree reduce to behave (block_reduce_sum
  // halves the offset). head_dim is typically 64/96/128/256 — already pow2 for
  // the common cases. For oddball sizes, round up to the next pow2 ≤ 1024.
  uint32_t bs = 1;
  while (bs < head_dim) {
    bs <<= 1;
  }
  if (bs > 1024) {
    bs = 1024;
  }
  return bs;
}

extern "C" void launch_turboquant_encode_f32(const float *d_input,
                                             const float *d_rotation,
                                             const float *d_projection,
                                             float *d_output,
                                             uint32_t num_vectors,
                                             uint32_t head_dim,
                                             uint32_t proj_dim,
                                             cudaStream_t stream) {
  const uint32_t block_size = pick_block_size(head_dim);
  const size_t shared_bytes =
      (4u * head_dim + block_size) * sizeof(float);
  turboquant_encode_kernel<float>
      <<<num_vectors, block_size, shared_bytes, stream>>>(
          d_input, d_rotation, d_projection, d_output, num_vectors, head_dim,
          proj_dim);
}

extern "C" void launch_turboquant_encode_f16(const __half *d_input,
                                             const float *d_rotation,
                                             const float *d_projection,
                                             __half *d_output,
                                             uint32_t num_vectors,
                                             uint32_t head_dim,
                                             uint32_t proj_dim,
                                             cudaStream_t stream) {
  const uint32_t block_size = pick_block_size(head_dim);
  const size_t shared_bytes =
      (4u * head_dim + block_size) * sizeof(float);
  turboquant_encode_kernel<__half>
      <<<num_vectors, block_size, shared_bytes, stream>>>(
          d_input, d_rotation, d_projection, d_output, num_vectors, head_dim,
          proj_dim);
}
