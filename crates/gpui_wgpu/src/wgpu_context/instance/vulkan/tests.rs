//! The Vulkan instance of a display, and the GL instance beside it.
//!
//! Closes the class of a Vulkan instance that enables the surface extension
//! of a window system other than its display's: Mesa's device selection
//! layer connects to `DISPLAY` from an instance with `VK_KHR_xcb_surface`,
//! which starts an XWayland under a Wayland compositor. Each case reads the
//! extensions the instance was created with. Does not cover a host whose
//! Vulkan loader offers none of the window-system surface extensions, and
//! does not cover `DISPLAY` being hidden from the drivers, which needs a
//! single-threaded process and a driver that records its environment.
//!
//! Also closes the class of a GL instance created while a Vulkan tier
//! selects the context: creating it loads every installed EGL vendor
//! library, and on a Wayland display it loads the Vulkan drivers with
//! `DISPLAY` set when the GL driver runs on Vulkan. A Wayland display
//! selects from every Vulkan adapter before it creates the GL instance.
//! Does not cover the Vulkan and GL tier, whose GL instance on an
//! unconnected display would reach the X server `DISPLAY` names. A
//! surfaceless instance has no GL backend beside a Vulkan GPU either.

use std::ffi::CStr;
use std::ptr::NonNull;

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, RawDisplayHandle, WaylandDisplayHandle,
    XcbDisplayHandle, XlibDisplayHandle,
};
use wgpu::hal::api::Vulkan;

use super::{retain_surface_extension, surface_extension};
use crate::wgpu_context::adapter_selection::BackendTier;
use crate::wgpu_context::create_instance;
use crate::{DisplayInstances, WgpuContext};

/// A display handle no connection stands behind. Creating a Vulkan
/// instance reads only which window system the handle is of.
#[derive(Debug)]
struct Unconnected(RawDisplayHandle);

// Safety: the handle's pointers are never dereferenced.
unsafe impl Send for Unconnected {}
unsafe impl Sync for Unconnected {}

impl HasDisplayHandle for Unconnected {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // Safety: see `Unconnected`.
        Ok(unsafe { DisplayHandle::borrow_raw(self.0) })
    }
}

/// A display of each window system a Linux client runs on.
fn linux_displays() -> [RawDisplayHandle; 3] {
    [
        WaylandDisplayHandle::new(NonNull::dangling()).into(),
        XcbDisplayHandle::new(None, 0).into(),
        XlibDisplayHandle::new(None, 0).into(),
    ]
}

fn is_window_system_surface(extension: &CStr) -> bool {
    extension != c"VK_KHR_surface" && extension.to_bytes().ends_with(b"_surface")
}

#[test]
fn a_vulkan_instance_enables_the_surface_extension_of_its_display_alone() {
    for display in linux_displays() {
        let kept = surface_extension(display).expect("a Linux window system has an extension");
        let instances = DisplayInstances::new(Unconnected(display));
        let instance = instances
            .vulkan()
            .unwrap_or_else(|error| panic!("no Vulkan instance for {display:?}: {error:#}"));
        // Safety: the instance is only read.
        let hal = unsafe { instance.as_hal::<Vulkan>() }.expect("a Vulkan instance");
        let extensions = hal.shared_instance().extensions();
        let surfaces: Vec<&CStr> = extensions
            .iter()
            .copied()
            .filter(|&extension| is_window_system_surface(extension))
            .collect();
        assert_eq!(
            surfaces,
            [kept],
            "the instance for {display:?} enabled {extensions:?}"
        );
        assert!(
            extensions.contains(&c"VK_KHR_surface"),
            "the instance for {display:?} enabled {extensions:?}"
        );
    }
}

/// The surface extension of each window system of the Vulkan registry.
const WINDOW_SYSTEM_SURFACES: [&CStr; 7] = [
    c"VK_KHR_xlib_surface",
    c"VK_KHR_xcb_surface",
    c"VK_KHR_wayland_surface",
    c"VK_KHR_win32_surface",
    c"VK_KHR_android_surface",
    c"VK_EXT_metal_surface",
    c"VK_EXT_headless_surface",
];

/// Instance extensions that are no window system's surface extension,
/// the surface-related ones among them.
const OTHERS: [&CStr; 7] = [
    c"VK_EXT_acquire_drm_display",
    c"VK_KHR_display",
    c"VK_KHR_get_surface_capabilities2",
    c"VK_KHR_surface_protected_capabilities",
    c"VK_EXT_swapchain_colorspace",
    c"VK_KHR_get_physical_device_properties2",
    c"VK_EXT_debug_utils",
];

#[test]
fn restriction_removes_the_surface_extensions_of_other_window_systems_alone() {
    let offered: Vec<&'static CStr> = [c"VK_KHR_surface"]
        .into_iter()
        .chain(WINDOW_SYSTEM_SURFACES)
        .chain(OTHERS)
        .collect();
    for display in linux_displays() {
        let kept = surface_extension(display).unwrap();
        let mut extensions = offered.clone();
        retain_surface_extension(&mut extensions, kept);
        let expected: Vec<&CStr> = [c"VK_KHR_surface", kept]
            .into_iter()
            .chain(OTHERS)
            .collect();
        assert_eq!(extensions, expected, "for {display:?}");
    }
}

#[test]
fn a_context_a_vulkan_tier_selects_leaves_the_gl_instance_uncreated() {
    for display in linux_displays() {
        let instances = DisplayInstances::new(Unconnected(display));
        let mut tiers = Vec::new();
        WgpuContext::by_backend_tier(&instances, |tier, candidates| {
            tiers.push((tier, candidates.len()));
            Ok(())
        })
        .unwrap_or_else(|error| panic!("no context for {display:?}: {error:#}"));
        let vulkan_tier = match display {
            RawDisplayHandle::Wayland(_) => BackendTier::EveryVulkan,
            _ => BackendTier::Vulkan,
        };
        assert_eq!(tiers, [(vulkan_tier, 1)], "for {display:?}");
        assert!(
            instances.gl.get().is_none(),
            "the Vulkan tier created the GL instance for {display:?}"
        );
    }
}

fn surfaceless(backends: wgpu::Backends) -> wgpu::Instance {
    create_instance(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    })
}

fn has_gl_backend(instance: &wgpu::Instance) -> bool {
    // Safety: the returned reference is only tested for presence.
    unsafe { instance.as_hal::<wgpu::hal::api::Gles>() }.is_some()
}

/// With a GL backend beside a Vulkan GPU, the NVIDIA driver (580)
/// deadlocked the test process: one test destroyed an EGL context while
/// another destroyed a Vulkan device. Does not cover the backends
/// `ZED_BACKEND` and `WGPU_BACKEND` select.
#[test]
fn a_surfaceless_instance_has_a_gl_backend_only_without_a_vulkan_gpu() {
    for variable in ["ZED_BACKEND", "WGPU_BACKEND"] {
        assert!(
            std::env::var_os(variable).is_none(),
            "unset {variable} to run this test"
        );
    }
    let vulkan_gpu = gpui::block_on(
        surfaceless(wgpu::Backends::VULKAN).enumerate_adapters(wgpu::Backends::all()),
    )
    .iter()
    .any(|adapter| {
        matches!(
            adapter.get_info().device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        )
    });
    let gl_expected = !vulkan_gpu && has_gl_backend(&surfaceless(wgpu::Backends::GL));
    assert_eq!(
        has_gl_backend(&WgpuContext::surfaceless_instance()),
        gl_expected,
        "Vulkan GPU present: {vulkan_gpu}"
    );
}
