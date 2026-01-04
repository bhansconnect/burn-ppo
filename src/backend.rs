/// Backend selection via feature flags
///
/// Usage:
/// - `cargo run --release` → WGPU (default, cross-platform)
/// - `cargo run --release --features cuda` → CUDA (NVIDIA GPUs)
/// - `cargo run --release --features libtorch` → `LibTorch` (`PyTorch`)
/// - `cargo run --release --no-default-features` → `NdArray` (CPU, for testing)
///
/// `TrainingBackend` = Autodiff wrapper for gradient computation during training
/// `InferenceBackend` = Inner backend without autodiff for rollout inference
use burn::backend::Autodiff;
use burn::tensor::backend::AutodiffBackend;

// CUDA backend (highest priority)
#[cfg(feature = "cuda")]
pub type TrainingBackend = Autodiff<burn::backend::Cuda>;

#[cfg(feature = "cuda")]
pub fn init_device() -> <TrainingBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::cuda::CudaDevice::default()
}

#[cfg(feature = "cuda")]
pub fn backend_name() -> &'static str {
    "CUDA"
}

#[cfg(feature = "cuda")]
pub fn device_name(device: &<TrainingBackend as burn::tensor::backend::Backend>::Device) -> String {
    format!("{:?}", device)
}

// LibTorch backend (second priority)
#[cfg(all(feature = "libtorch", not(feature = "cuda")))]
pub type TrainingBackend = Autodiff<burn::backend::LibTorch>;

#[cfg(all(feature = "libtorch", not(feature = "cuda")))]
pub fn init_device() -> <TrainingBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::libtorch::LibTorchDevice::Cuda(0)
}

#[cfg(all(feature = "libtorch", not(feature = "cuda")))]
pub fn backend_name() -> &'static str {
    "LibTorch"
}

#[cfg(all(feature = "libtorch", not(feature = "cuda")))]
pub fn device_name(device: &<TrainingBackend as burn::tensor::backend::Backend>::Device) -> String {
    format!("{:?}", device)
}

// WGPU backend (default)
#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
pub type TrainingBackend = Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
pub fn init_device() -> <TrainingBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::wgpu::WgpuDevice::default()
}

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
pub const fn backend_name() -> &'static str {
    "WGPU"
}

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
pub fn device_name(device: &<TrainingBackend as burn::tensor::backend::Backend>::Device) -> String {
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::RuntimeOptions;
    let setup =
        burn::backend::wgpu::init_setup::<AutoGraphicsApi>(device, RuntimeOptions::default());
    let info = setup.adapter.get_info();
    format!("{} via {:?}", info.name, info.backend)
}

// NdArray backend (CPU fallback for testing)
#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "libtorch")
))]
pub type TrainingBackend = Autodiff<burn::backend::NdArray>;

#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "libtorch")
))]
pub fn init_device() -> <TrainingBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::ndarray::NdArrayDevice::default()
}

#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "libtorch")
))]
pub const fn backend_name() -> &'static str {
    "NdArray"
}

#[cfg(all(
    not(feature = "wgpu"),
    not(feature = "cuda"),
    not(feature = "libtorch")
))]
pub fn device_name(
    _device: &<TrainingBackend as burn::tensor::backend::Backend>::Device,
) -> String {
    "CPU".to_string()
}

/// Inner backend for inference (no autodiff graph accumulation)
///
/// Use `model.valid()` to get an `ActorCritic<InferenceBackend>` from
/// an `ActorCritic<TrainingBackend>` for rollout collection.
pub type InferenceBackend = <TrainingBackend as AutodiffBackend>::InnerBackend;
