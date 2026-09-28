//! The GPU context thread of an X11 client (`crate::linux::gpu_context`).
//! The thread loads the GPU driver, creates the instances the client's
//! contexts are created from, and creates the device once the X connection
//! is set up.
//!
//! The X connection is set up before the thread starts. Mesa's device
//! selection layer and the NVIDIA driver open an X connection of their own
//! as the thread loads them. An X server started without -noreset resets
//! when its last client disconnects, and the reset closes every connection
//! still in setup: on a server with no other client, as a fresh Xvfb is, a
//! connection set up beside the driver's fails with "Unknown connection
//! error".

use std::{ffi::c_void, ptr::NonNull, rc::Rc};

use anyhow::Context as _;
use gpui_wgpu::{CompositorGpuHint, DisplayInstances, WgpuContext};
use raw_window_handle as rwh;
use x11rb::xcb_ffi::XCBConnection;

use crate::linux::gpu_context::{self, Pending};

/// The instances of the client's display and the client's GPU context,
/// created on the `gpu-context` thread. It holds the client's X connection
/// open while the thread reads it.
pub(crate) type PendingContext = gpu_context::PendingContext<Rc<XCBConnection>>;

/// What an X11 window's renderer is created from.
pub(crate) type WindowGpu = gpu_context::WindowGpu<Rc<XCBConnection>>;

/// Starts the thread that creates the GPU context of the client connected
/// through `connection`, whose X screen is `screen`.
pub(crate) fn spawn(
    connection: &Rc<XCBConnection>,
    screen: usize,
    compositor_gpu: Option<CompositorGpuHint>,
) -> anyhow::Result<PendingContext> {
    let raw = as_raw_xcb_connection::AsRawXcbConnection::as_raw_xcb_connection(&**connection);
    let display = XcbDisplay {
        connection: NonNull::new(raw.cast()).context("The X connection has no xcb handle")?,
        screen: i32::try_from(screen).context("X screen number out of range")?,
    };
    Pending::spawn("gpu-context", Rc::clone(connection), move || {
        let instances = DisplayInstances::new(display);
        let context = WgpuContext::for_display(&instances, compositor_gpu);
        (instances, context)
    })
}

/// The client's X connection as the display of the GPU instance.
#[derive(Debug, Clone, Copy)]
struct XcbDisplay {
    connection: NonNull<c_void>,
    screen: i32,
}

// Safety: an xcb connection is safe to use from any thread. The pending
// context holds the connection open until its thread has ended, and the
// client holds it open while the instance exists.
unsafe impl Send for XcbDisplay {}
unsafe impl Sync for XcbDisplay {}

impl rwh::HasDisplayHandle for XcbDisplay {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        let handle = rwh::XcbDisplayHandle::new(Some(self.connection), self.screen);
        Ok(unsafe { rwh::DisplayHandle::borrow_raw(handle.into()) })
    }
}
