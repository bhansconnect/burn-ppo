//! Zero-cost profiling macros that only activate with the `tracy` feature.
//!
//! This module provides Tracy integration directly (not through the `profiling` crate)
//! so that only our application code gets instrumented, not dependencies like wgpu.

/// Create a named profiling scope. Zero-cost when tracy feature is disabled.
#[cfg(feature = "tracy")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracy_client::span!($name);
    };
}

#[cfg(not(feature = "tracy"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}

/// Mark the current function for profiling. Zero-cost when tracy feature is disabled.
/// Place at the start of a function body.
#[cfg(feature = "tracy")]
macro_rules! profile_function {
    () => {
        let _span = tracy_client::span!();
    };
}

#[cfg(not(feature = "tracy"))]
macro_rules! profile_function {
    () => {};
}

/// Mark the end of a frame for Tracy's frame view. Zero-cost when tracy feature is disabled.
#[cfg(feature = "tracy")]
macro_rules! profile_frame {
    () => {
        tracy_client::frame_mark();
    };
}

#[cfg(not(feature = "tracy"))]
macro_rules! profile_frame {
    () => {};
}

/// Force GPU synchronization for accurate profiling timing.
/// This forces all queued GPU operations to complete before continuing.
/// Zero-cost when tracy feature is disabled.
#[cfg(feature = "tracy")]
macro_rules! gpu_sync {
    ($tensor:expr) => {{
        let _ = $tensor.clone().into_data();
    }};
}

#[cfg(not(feature = "tracy"))]
macro_rules! gpu_sync {
    ($tensor:expr) => {};
}

pub(crate) use gpu_sync;
pub(crate) use profile_frame;
pub(crate) use profile_function;
pub(crate) use profile_scope;
