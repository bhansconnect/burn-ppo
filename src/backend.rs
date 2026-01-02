/// Backend selection via feature flags
///
/// Usage:
/// - `cargo run --release` → WGPU (default, cross-platform)
/// - `cargo run --release --features cuda` → CUDA (NVIDIA GPUs)
/// - `cargo run --release --features libtorch` → LibTorch (PyTorch)
use burn::backend::Autodiff;

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
pub fn backend_name() -> &'static str {
    "WGPU"
}

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
pub fn device_name(device: &<TrainingBackend as burn::tensor::backend::Backend>::Device) -> String {
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    let setup = burn::backend::wgpu::init_setup::<AutoGraphicsApi>(device, Default::default());
    let info = setup.adapter.get_info();
    format!("{} via {:?}", info.name, info.backend)
}
