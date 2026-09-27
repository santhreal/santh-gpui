//! The creation of every wgpu instance, and the instances of a display.

use std::sync::{Mutex, PoisonError};

#[cfg(not(target_family = "wasm"))]
use std::{cell::OnceCell, sync::Arc};

#[cfg(not(target_family = "wasm"))]
use raw_window_handle::RawDisplayHandle;
#[cfg(not(target_family = "wasm"))]
use wgpu::wgt::WgpuHasDisplayHandle;

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
mod vulkan;

/// Held while an instance is created. The Vulkan loader (1.3.275 with the
/// NVIDIA driver) calls a null ICD entry point when a thread enumerates the
/// instance extensions while another thread negotiates the interface
/// version of the same ICD. Instance creation is rare, so the lock is
/// uncontended outside tests.
static CREATION: Mutex<()> = Mutex::new(());

/// Creates a wgpu instance under [`CREATION`].
#[allow(clippy::disallowed_methods)]
pub(crate) fn create_instance(descriptor: wgpu::InstanceDescriptor) -> wgpu::Instance {
    let _serialized = CREATION.lock().unwrap_or_else(PoisonError::into_inner);
    wgpu::Instance::new(descriptor)
}

/// The instances the native contexts of one display select adapters from.
///
/// The Vulkan instance enables the surface extension of the display's
/// window system and no other surface extension. Mesa's device selection
/// layer connects to the X server `DISPLAY` names when an instance with
/// `VK_KHR_xcb_surface` enumerates its devices and the Wayland compositor
/// reports no device; under a Wayland compositor that server is an
/// XWayland the compositor starts at the first X11 connection.
///
/// The GL instance is created at the first selection that includes the GL
/// adapters: creating it loads every installed EGL vendor library.
///
/// On a Wayland display the GL instance is a last resort, created only when
/// the Vulkan instance offers no adapter that can be selected. The client
/// has started threads by then, so `DISPLAY` is in the environment, and
/// Mesa's GL driver on Vulkan (zink) loads the Vulkan drivers as the GL
/// instance is created.
#[cfg(not(target_family = "wasm"))]
pub struct DisplayInstances {
    display: Arc<dyn WgpuHasDisplayHandle>,
    vulkan: anyhow::Result<wgpu::Instance>,
    gl: OnceCell<wgpu::Instance>,
    gl_last_resort: bool,
}

#[cfg(not(target_family = "wasm"))]
impl DisplayInstances {
    /// Creates the Vulkan instance of `display`.
    ///
    /// On a Wayland display, the loader and the Vulkan drivers run without
    /// `DISPLAY` in the environment while the instance is created, if the
    /// calling thread is the only thread of the process. The NVIDIA driver
    /// connects to the X server `DISPLAY` names when the loader loads it,
    /// and the loader loads the drivers in the calls that create the
    /// instance. A Wayland client creates the instances of its display
    /// before it starts a thread.
    pub fn new<D: WgpuHasDisplayHandle>(display: D) -> Self {
        let display: Arc<dyn WgpuHasDisplayHandle> = Arc::new(display);
        let vulkan = vulkan_instance(&*display);
        if let Err(error) = &vulkan {
            log::info!("No Vulkan instance: {error:#}");
        }
        let gl_last_resort = display
            .display_handle()
            .is_ok_and(|handle| matches!(handle.as_raw(), RawDisplayHandle::Wayland(_)));
        Self {
            display,
            vulkan,
            gl: OnceCell::new(),
            gl_last_resort,
        }
    }

    /// The Vulkan instance, or the error that prevented its creation.
    pub(crate) fn vulkan(&self) -> Result<&wgpu::Instance, &anyhow::Error> {
        self.vulkan.as_ref()
    }

    /// Whether the GL instance is created only when the Vulkan instance
    /// offers no adapter that can be selected.
    pub(crate) fn gl_last_resort(&self) -> bool {
        self.gl_last_resort
    }

    /// The GL instance, created at the first call.
    pub(crate) fn gl(&self) -> &wgpu::Instance {
        self.gl.get_or_init(|| {
            create_instance(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::GL,
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: Some(Box::new(Arc::clone(&self.display))),
            })
        })
    }
}

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
))]
use vulkan::vulkan_instance;

/// wgpu has a Vulkan backend on Windows, Linux, Android, and FreeBSD alone.
#[cfg(not(any(
    target_family = "wasm",
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd"
)))]
fn vulkan_instance(_display: &dyn WgpuHasDisplayHandle) -> anyhow::Result<wgpu::Instance> {
    anyhow::bail!("wgpu has no Vulkan backend on this platform")
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
