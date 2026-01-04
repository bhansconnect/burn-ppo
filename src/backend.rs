//! Backend selection via feature flags and runtime dispatch
//!
//! Runtime usage (preferred):
//! - `cargo run --release -- train --backend wgpu`
//! - `cargo run --release -- train --backend ndarray`
//! - `cargo run --release --features cuda -- train --backend cuda`
//!
//! `TB` = Training backend (Autodiff wrapper for gradient computation)
//! Use `TB::InnerBackend` for inference without autodiff

// ============================================================================
// Runtime backend selection
// ============================================================================

/// Returns list of backends available in this build
#[must_use]
pub fn available_backends() -> Vec<&'static str> {
    let mut backends = vec!["ndarray"];
    #[cfg(feature = "wgpu")]
    backends.push("wgpu");
    #[cfg(feature = "cuda")]
    backends.push("cuda");
    #[cfg(feature = "libtorch")]
    backends.push("libtorch");
    backends
}

/// Returns the best available backend for this build (cuda > libtorch > wgpu > ndarray)
#[must_use]
pub fn default_backend() -> &'static str {
    #[cfg(feature = "cuda")]
    return "cuda";
    #[cfg(all(feature = "libtorch", not(feature = "cuda")))]
    return "libtorch";
    #[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "libtorch")))]
    return "wgpu";
    #[cfg(all(
        not(feature = "wgpu"),
        not(feature = "cuda"),
        not(feature = "libtorch")
    ))]
    return "ndarray";
}

/// Dispatch to the appropriate backend based on runtime string selection.
///
/// This macro provides the type alias `TB` (training backend with autodiff)
/// and initializes `$device`. Use `TB::InnerBackend` for inference.
///
/// # Usage
/// ```ignore
/// dispatch_backend!(backend_name, device, {
///     // TB = Autodiff<Backend>, device is initialized
///     // Use TB::InnerBackend for inference
///     run_training::<TB, E>(&device, ...)
/// })
/// ```
///
/// # Errors
/// Returns `anyhow::bail!` if the backend name is unknown or not compiled in.
#[macro_export]
macro_rules! dispatch_backend {
    ($backend_name:expr, $device:ident, $callback:expr) => {{
        use burn::backend::Autodiff;

        match $backend_name {
            #[cfg(feature = "wgpu")]
            "wgpu" => {
                type TB = Autodiff<burn::backend::wgpu::Wgpu>;
                let $device = burn::backend::wgpu::WgpuDevice::default();
                $callback
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                type TB = Autodiff<burn::backend::Cuda>;
                let $device = burn::backend::cuda::CudaDevice::default();
                $callback
            }
            #[cfg(feature = "libtorch")]
            "libtorch" => {
                type TB = Autodiff<burn::backend::LibTorch>;
                let $device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
                $callback
            }
            "ndarray" => {
                type TB = Autodiff<burn::backend::NdArray>;
                let $device = burn::backend::ndarray::NdArrayDevice::default();
                $callback
            }
            _ => anyhow::bail!(
                "Unknown backend '{}'. Available: {}",
                $backend_name,
                $crate::backend::available_backends().join(", ")
            ),
        }
    }};
}

/// Get backend display name for a given backend string
#[must_use]
pub fn get_backend_display_name(backend: &str) -> &'static str {
    match backend {
        "cuda" => "CUDA",
        "libtorch" => "LibTorch",
        "wgpu" => "WGPU",
        "ndarray" => "NdArray",
        _ => "Unknown",
    }
}
