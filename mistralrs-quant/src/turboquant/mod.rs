//! TurboQuant (PolarQuant + QJL) fused encode kernels.
//!
//! Native CUDA + Metal implementations of the encode pipeline for the KV
//! cache codec. Mirrors the pure-Rust reference at
//! `vibecody/vibeui/crates/vibe-infer/src/kv_cache_tq.rs` and the parity-
//! tested `CandleTurboQuantCodec`. The codec dispatch wrapper lives
//! downstream in `vibe-infer` (mistralrs-quant cannot depend on
//! mistralrs-core, so the `KvCacheCodec` trait impl is wired on the
//! consumer side).
//!
//! See:
//! - Metal kernel: `src/metal_kernels/turboquant.metal`
//!   (launcher: `metal_kernels::call_turboquant_encode`)
//! - CUDA kernel:  `kernels/turboquant/turboquant.cu`
//!   (FFI bindings: `turboquant::ffi`)
//!
//! The public surface is [`encode`] — a single entry point that picks the
//! right backend by inspecting `input.device()`. Returns `Err` for CPU
//! tensors (callers should use `CandleTurboQuantCodec` for the host path).

mod ops;

pub use ops::encode;
