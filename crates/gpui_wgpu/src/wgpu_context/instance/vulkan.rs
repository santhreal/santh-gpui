//! The Vulkan instance of a display.

use std::ffi::{CStr, OsString};
use std::sync::PoisonError;

use raw_window_handle::RawDisplayHandle;
use wgpu::hal::{api::Vulkan, vulkan};
use wgpu::wgt::WgpuHasDisplayHandle;

use super::CREATION;

/// Creates a Vulkan instance under [`CREATION`] whose only surface
/// extension is the one of `display`'s window system.
pub(super) fn vulkan_instance(
    display: &dyn WgpuHasDisplayHandle,
) -> anyhow::Result<wgpu::Instance> {
    let handle = display
        .display_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get display handle: {e}"))?;
    let kept = surface_extension(handle.as_raw());
    let descriptor = wgpu::hal::InstanceDescriptor {
        name: "wgpu",
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        telemetry: None,
        display: Some(handle),
    };
    let instance = {
        let _serialized = CREATION.lock().unwrap_or_else(PoisonError::into_inner);
        let _hidden = HiddenDisplay::new(handle.as_raw());
        // Safety: the callback removes only the surface extensions of other
        // window systems. wgpu-hal reads a surface extension only to create
        // a surface, and fails to create one for a window system whose
        // extension the instance did not enable.
        unsafe {
            vulkan::Instance::init_with_callback(
                &descriptor,
                Some(Box::new(
                    move |args: vulkan::CreateInstanceCallbackArgs<'_, '_, '_>| {
                        if let Some(kept) = kept {
                            retain_surface_extension(args.extensions, kept);
                        }
                    },
                )),
            )
        }
    }
    .map_err(|e| anyhow::anyhow!("Failed to create the Vulkan instance: {e}"))?;
    // Safety: wgpu-hal created the instance, and nothing else owns it.
    Ok(unsafe { wgpu::Instance::from_hal::<Vulkan>(instance) })
}

/// The Vulkan surface extension of the window system of `display`, or
/// `None` for a window system this crate creates no Vulkan surface on.
fn surface_extension(display: RawDisplayHandle) -> Option<&'static CStr> {
    match display {
        RawDisplayHandle::Wayland(_) => Some(c"VK_KHR_wayland_surface"),
        RawDisplayHandle::Xcb(_) => Some(c"VK_KHR_xcb_surface"),
        RawDisplayHandle::Xlib(_) => Some(c"VK_KHR_xlib_surface"),
        RawDisplayHandle::Windows(_) => Some(c"VK_KHR_win32_surface"),
        RawDisplayHandle::Android(_) => Some(c"VK_KHR_android_surface"),
        _ => None,
    }
}

/// Removes from `extensions` every surface extension of a window system,
/// `VK_<vendor>_<system>_surface`, other than `kept`. `VK_KHR_surface`, the
/// extension each of them requires, stays.
fn retain_surface_extension(extensions: &mut Vec<&'static CStr>, kept: &CStr) {
    extensions.retain(|&extension| {
        extension == kept
            || extension == c"VK_KHR_surface"
            || !extension.to_bytes().ends_with(b"_surface")
    });
}

/// `DISPLAY` removed from the environment while the Vulkan instance of a
/// Wayland display is created, and restored on drop.
///
/// Another thread may read the environment while it changes, so the
/// variable is removed only while the process has one thread. The loader,
/// the NVIDIA driver, and Mesa's drivers start no thread before the
/// instance is created, so the process still has one when the variable is
/// restored.
struct HiddenDisplay(Option<OsString>);

impl HiddenDisplay {
    fn new(display: RawDisplayHandle) -> Self {
        let hidden = match display {
            RawDisplayHandle::Wayland(_) if single_threaded() => std::env::var_os("DISPLAY"),
            _ => None,
        };
        if hidden.is_some() {
            // Safety: the calling thread is the only thread of the process.
            unsafe { std::env::remove_var("DISPLAY") };
        }
        Self(hidden)
    }
}

impl Drop for HiddenDisplay {
    fn drop(&mut self) {
        if let Some(display) = self.0.take() {
            // Safety: the process has had one thread since the variable was
            // removed.
            unsafe { std::env::set_var("DISPLAY", display) };
        }
    }
}

/// Whether the calling thread is the only thread of the process; false
/// where `/proc` does not list the threads.
fn single_threaded() -> bool {
    std::fs::read_dir("/proc/self/task").is_ok_and(|tasks| tasks.count() == 1)
}

#[cfg(test)]
mod tests;
