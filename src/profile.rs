//! Zero-cost profiling macros that only activate with the `tracy` feature.
//!
//! This module provides Tracy integration directly (not through the `profiling` crate)
//! so that only our application code gets instrumented, not dependencies like wgpu.

// Memory tracking: callstack depth 0 by default (fast), 10 with tracy-callstack feature
#[cfg(all(feature = "tracy", feature = "tracy-callstack"))]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 30);

#[cfg(all(feature = "tracy", not(feature = "tracy-callstack")))]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 0);

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

pub(crate) use profile_frame;
pub(crate) use profile_function;
pub(crate) use profile_scope;
