#![allow(dead_code)]

use half::f16;

/// Whether the CUDA TurboQuant kernel is compiled into the build.
///
/// Always `true` when the `cuda` feature is enabled — turboquant.cu has no
/// SM compute-capability requirement (uses fp16 ops widely supported on
/// sm_53+) and no `_dummy.cu` exclusion path. Kept as a constant for
/// symmetry with other quant FFI modules.
pub(crate) const HAVE_TURBOQUANT_KERNELS: bool = true;

extern "C" {
    /// Fused PolarQuant + QJL encode for an fp32 KV cache slice.
    ///
    /// Buffer contract (all device pointers, contiguous):
    /// - `d_input`:      [num_vectors * head_dim] fp32
    /// - `d_rotation`:   [head_dim * head_dim] fp32, row-major (R)
    /// - `d_projection`: [proj_dim * head_dim] fp32, row-major (P)
    /// - `d_output`:     [num_vectors * head_dim] fp32 (lossy reconstruction)
    pub(crate) fn launch_turboquant_encode_f32(
        d_input: *const f32,
        d_rotation: *const f32,
        d_projection: *const f32,
        d_output: *mut f32,
        num_vectors: u32,
        head_dim: u32,
        proj_dim: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    /// Fused PolarQuant + QJL encode for an fp16 KV cache slice.
    ///
    /// Same contract as `launch_turboquant_encode_f32` but input/output are
    /// fp16. Rotation/projection matrices remain fp32 (matches the pure-Rust
    /// reference and the Metal kernel — small matrices, kept fp32 to avoid
    /// accumulating QJL projection error).
    pub(crate) fn launch_turboquant_encode_f16(
        d_input: *const f16,
        d_rotation: *const f32,
        d_projection: *const f32,
        d_output: *mut f16,
        num_vectors: u32,
        head_dim: u32,
        proj_dim: u32,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
