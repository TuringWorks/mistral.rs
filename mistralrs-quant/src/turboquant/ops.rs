//! TurboQuant fused encode — CustomOp1 dispatch for CUDA and Metal.
//!
//! The pure-Rust reference lives in `vibe-infer::kv_cache_tq`; this module
//! provides the device-side kernel backed by either
//! `kernels/turboquant/turboquant.cu` or
//! `src/metal_kernels/turboquant.metal`. The public entry point, [`encode`],
//! dispatches by the input tensor's device.

use candle_core::{CpuStorage, CustomOp1, DType, Result, Tensor};

/// Per-call wrapper that bakes the rotation/projection matrices into a
/// `CustomOp1`. The matrices are passed by reference — the backend paths
/// extract device pointers when they fire.
#[allow(dead_code)]
struct TurboQuantEncode<'a> {
    rotation: &'a Tensor,   // F32, [head_dim, head_dim]
    projection: &'a Tensor, // F32, [proj_dim, head_dim]
    head_dim: usize,
    proj_dim: usize,
}

impl<'a> CustomOp1 for TurboQuantEncode<'a> {
    fn name(&self) -> &'static str {
        "turboquant-encode"
    }

    fn cpu_fwd(
        &self,
        _input_s: &CpuStorage,
        _input_l: &candle_core::Layout,
    ) -> Result<(CpuStorage, candle_core::Shape)> {
        candle_core::bail!(
            "TurboQuant native encode is device-only; use CandleTurboQuantCodec for CPU inputs"
        );
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input_s: &candle_core::CudaStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::Storage;
        use half::f16;

        use crate::utils::slice_ptr;

        if !super::ffi::HAVE_TURBOQUANT_KERNELS {
            candle_core::bail!("Do not have TurboQuant CUDA kernels.");
        }
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("TurboQuantEncode: input must have start_offset 0 and be contiguous");
        }

        let dtype = input_s.dtype();
        let dev = input_s.device();
        let total = input_l.shape().elem_count();
        if total % self.head_dim != 0 {
            candle_core::bail!(
                "TurboQuantEncode: input element count {} not a multiple of head_dim {}",
                total,
                self.head_dim
            );
        }
        let num_vectors = total / self.head_dim;

        let (rot_storage, _) = self.rotation.storage_and_layout();
        let Storage::Cuda(rot_cuda) = &*rot_storage else {
            candle_core::bail!("TurboQuantEncode: rotation must live on CUDA");
        };
        let (proj_storage, _) = self.projection.storage_and_layout();
        let Storage::Cuda(proj_cuda) = &*proj_storage else {
            candle_core::bail!("TurboQuantEncode: projection must live on CUDA");
        };
        let rot_slice = match &rot_cuda.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("TurboQuantEncode: rotation must be F32"),
        };
        let proj_slice = match &proj_cuda.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("TurboQuantEncode: projection must be F32"),
        };
        let (rotation_ptr, _rotation_guard) = slice_ptr(rot_slice, 0);
        let (projection_ptr, _projection_guard) = slice_ptr(proj_slice, 0);

        let stream = dev.cuda_stream().cu_stream();
        let num_vectors_u32: u32 = num_vectors
            .try_into()
            .map_err(|_| candle_core::Error::Msg("num_vectors exceeds u32::MAX".into()))?;
        let head_dim_u32: u32 = self
            .head_dim
            .try_into()
            .map_err(|_| candle_core::Error::Msg("head_dim exceeds u32::MAX".into()))?;
        let proj_dim_u32: u32 = self
            .proj_dim
            .try_into()
            .map_err(|_| candle_core::Error::Msg("proj_dim exceeds u32::MAX".into()))?;

        let out_storage = match dtype {
            DType::F32 => {
                let input_slice = match &input_s.slice {
                    CudaStorageSlice::F32(s) => s,
                    _ => candle_core::bail!("input slice dtype mismatch (expected F32)"),
                };
                let output = dev.alloc_zeros::<f32>(total)?;
                let (input_ptr, _input_guard) = slice_ptr(input_slice, 0);
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    super::ffi::launch_turboquant_encode_f32(
                        input_ptr as *const f32,
                        rotation_ptr as *const f32,
                        projection_ptr as *const f32,
                        output_ptr as *mut f32,
                        num_vectors_u32,
                        head_dim_u32,
                        proj_dim_u32,
                        stream,
                    );
                }
                drop(output_guard);
                candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            DType::F16 => {
                let input_slice = match &input_s.slice {
                    CudaStorageSlice::F16(s) => s,
                    _ => candle_core::bail!("input slice dtype mismatch (expected F16)"),
                };
                let output = dev.alloc_zeros::<f16>(total)?;
                let (input_ptr, _input_guard) = slice_ptr(input_slice, 0);
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    super::ffi::launch_turboquant_encode_f16(
                        input_ptr as *const f16,
                        rotation_ptr as *const f32,
                        projection_ptr as *const f32,
                        output_ptr as *mut f16,
                        num_vectors_u32,
                        head_dim_u32,
                        proj_dim_u32,
                        stream,
                    );
                }
                drop(output_guard);
                candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            other => candle_core::bail!(
                "TurboQuantEncode CUDA: unsupported dtype {:?} (only F32/F16)",
                other
            ),
        };

        Ok((out_storage, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input_s: &candle_core::MetalStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::Storage;

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("TurboQuantEncode: input must have start_offset 0 and be contiguous");
        }
        let dtype = input_s.dtype();
        if !matches!(dtype, DType::F32 | DType::F16) {
            candle_core::bail!(
                "TurboQuantEncode Metal: unsupported dtype {:?} (only F32/F16)",
                dtype
            );
        }
        let total = input_l.shape().elem_count();
        if total % self.head_dim != 0 {
            candle_core::bail!(
                "TurboQuantEncode: input element count {} not a multiple of head_dim {}",
                total,
                self.head_dim
            );
        }
        let num_vectors = total / self.head_dim;

        let device = input_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("turboquant-encode");

        let (rot_storage, _) = self.rotation.storage_and_layout();
        let Storage::Metal(rot_metal) = &*rot_storage else {
            candle_core::bail!("TurboQuantEncode: rotation must live on Metal");
        };
        if rot_metal.dtype() != DType::F32 {
            candle_core::bail!("TurboQuantEncode: rotation must be F32");
        }
        let (proj_storage, _) = self.projection.storage_and_layout();
        let Storage::Metal(proj_metal) = &*proj_storage else {
            candle_core::bail!("TurboQuantEncode: projection must live on Metal");
        };
        if proj_metal.dtype() != DType::F32 {
            candle_core::bail!("TurboQuantEncode: projection must be F32");
        }

        let output = device.new_buffer(total, dtype, "turboquant-encode-output")?;

        crate::metal_kernels::call_turboquant_encode(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            dtype,
            input_s.buffer(),
            rot_metal.buffer(),
            proj_metal.buffer(),
            &output,
            num_vectors,
            self.head_dim,
            self.proj_dim,
        )
        .map_err(candle_core::Error::wrap)?;

        let new_storage = candle_core::MetalStorage::new(output, device.clone(), total, dtype);
        Ok((new_storage, input_l.shape().clone()))
    }
}

/// Fused TurboQuant encode (rotate → polar quantize → reconstruct → QJL
/// residual → inverse rotate) for one KV cache slice on CUDA or Metal.
///
/// # Arguments
/// - `input`: contiguous tensor on CUDA or Metal, `F32` or `F16`. Last axis
///   must equal `head_dim` (rotation matrix dimension). All other axes
///   flatten into `num_vectors`.
/// - `rotation`: `F32` tensor on the same device, shape
///   `[head_dim, head_dim]`, row-major.
/// - `projection`: `F32` tensor on the same device, shape
///   `[proj_dim, head_dim]`, row-major.
///
/// Returns a new tensor on the same device, with the same shape and dtype as
/// `input`, containing the PolarQuant + QJL reconstruction.
///
/// # Errors
/// Returns an error for CPU tensors or shape / dtype mismatches. Use
/// `CandleTurboQuantCodec` (pure-Rust host impl) for the CPU path.
pub fn encode(input: &Tensor, rotation: &Tensor, projection: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    if !matches!(dtype, DType::F32 | DType::F16) {
        candle_core::bail!(
            "TurboQuant encode supports F32 / F16 only; got input dtype {:?}",
            dtype
        );
    }
    if rotation.dtype() != DType::F32 {
        candle_core::bail!(
            "TurboQuant encode requires F32 rotation matrix; got {:?}",
            rotation.dtype()
        );
    }
    if projection.dtype() != DType::F32 {
        candle_core::bail!(
            "TurboQuant encode requires F32 projection matrix; got {:?}",
            projection.dtype()
        );
    }
    if rotation.dims().len() != 2 || rotation.dims()[0] != rotation.dims()[1] {
        candle_core::bail!(
            "TurboQuant encode: rotation must be square [head_dim, head_dim]; got {:?}",
            rotation.dims()
        );
    }
    if projection.dims().len() != 2 {
        candle_core::bail!(
            "TurboQuant encode: projection must be 2-D [proj_dim, head_dim]; got {:?}",
            projection.dims()
        );
    }
    let head_dim = rotation.dims()[0];
    if projection.dims()[1] != head_dim {
        candle_core::bail!(
            "TurboQuant encode: projection inner dim {} != head_dim {}",
            projection.dims()[1],
            head_dim
        );
    }
    let proj_dim = projection.dims()[0];

    let dims = input.dims();
    if dims.is_empty() || *dims.last().unwrap() != head_dim {
        candle_core::bail!(
            "TurboQuant encode: input last axis must equal head_dim={}; got shape {:?}",
            head_dim,
            dims
        );
    }

    let rotation = rotation.contiguous()?;
    let projection = projection.contiguous()?;
    let input = input.contiguous()?;

    input.apply_op1_no_bwd(&TurboQuantEncode {
        rotation: &rotation,
        projection: &projection,
        head_dim,
        proj_dim,
    })
}
