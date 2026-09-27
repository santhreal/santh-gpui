//! The GPU context of an X11 client. A thread loads the GPU driver and
//! creates the device once the X connection is set up, while the client
//! queries the X server, loads the system fonts, and runs the
//! application's start. The first window takes the result.
//!
//! The X connection is set up before the thread starts. Mesa's device
//! selection layer and the NVIDIA driver open an X connection of their own
//! as the thread loads them. An X server started without -noreset resets
//! when its last client disconnects, and the reset closes every connection
//! still in setup: on a server with no other client, as a fresh Xvfb is, a
//! connection set up beside the driver's fails with "Unknown connection
//! error".
//!
//! The thread creates the instances the client's contexts are created
//! from, and selects the adapter without a surface. The first window
//! keeps that context only if the renderer of its surface is created on it
//! with no error, the surface configuration included; otherwise the window
//! creates a context whose adapter is tested against its surface, on the
//! same instances. On a host with more than one GPU, an adapter can report
//! a surface as compatible and fail to configure it.

use std::{ffi::c_void, fmt::Debug, ptr::NonNull, rc::Rc, sync::Arc, thread::JoinHandle};

use anyhow::{Context as _, anyhow};
use gpui_wgpu::{
    CompositorGpuHint, DisplayInstances, GpuContext, WgpuContext, WgpuRenderer, WgpuSurfaceConfig,
    wgpu,
};
use raw_window_handle as rwh;
use x11rb::xcb_ffi::XCBConnection;

/// The result of a thread that reads a value `keep` holds alive. Dropping
/// it joins the thread before it releases `keep`.
pub(crate) struct Pending<T: Send + 'static, K> {
    thread: Option<JoinHandle<T>>,
    _keep: K,
}

impl<T: Send + 'static, K> Pending<T, K> {
    fn spawn(
        name: &str,
        keep: K,
        body: impl FnOnce() -> T + Send + 'static,
    ) -> anyhow::Result<Self> {
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(body)
            .with_context(|| format!("Failed to start the {name} thread"))?;
        Ok(Self {
            thread: Some(thread),
            _keep: keep,
        })
    }

    /// Waits for the thread and returns its result. A panic on the thread
    /// continues on the caller's.
    fn join(mut self) -> T {
        self.thread
            .take()
            .map(|thread| {
                thread
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .expect("a pending thread is joined once")
    }
}

impl<T: Send + 'static, K> Drop for Pending<T, K> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            // Its result is dropped, and so is a panic on it: nothing reads
            // either once the client is gone.
            thread.join().ok();
        }
    }
}

/// The instances of the client's display and the client's GPU context,
/// created on the `gpu-context` thread. It holds the client's X connection
/// open while the thread reads it.
pub(crate) type PendingContext =
    Pending<(DisplayInstances, anyhow::Result<WgpuContext>), Rc<XCBConnection>>;

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

/// What a window's renderer is created from.
pub(crate) struct WindowGpu {
    /// The context the client's windows share.
    pub context: GpuContext,
    pub compositor_gpu: Option<CompositorGpuHint>,
    /// The instances and the context from the GPU context thread, for the
    /// client's first window.
    pub pending: Option<PendingContext>,
}

impl WindowGpu {
    /// Creates the renderer of `window`'s surface.
    pub(crate) fn renderer<W>(
        self,
        window: &W,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<WgpuRenderer>
    where
        W: rwh::HasWindowHandle + rwh::HasDisplayHandle + Debug + Send + Sync + Clone + 'static,
    {
        let Self {
            context,
            compositor_gpu,
            pending,
        } = self;
        if let Some(pending) = pending {
            let (instances, prewarmed) = pending.join();
            context.set_instances(instances)?;
            match prewarmed {
                Ok(prewarmed) => {
                    match renderer_on(&context, prewarmed, window, &config, compositor_gpu) {
                        Ok(renderer) => return Ok(renderer),
                        Err(error) => log::info!(
                            "The GPU context created before the first window cannot draw its \
                             surface ({error:#}); creating one tested against the surface"
                        ),
                    }
                }
                Err(error) => log::info!(
                    "The GPU context thread failed ({error:#}); creating the context with the \
                     first window"
                ),
            }
        }
        WgpuRenderer::new(context, window, config, compositor_gpu)
    }
}

/// Creates the renderer on `prewarmed`, which becomes the client's context.
/// Fails, and leaves the client without a context, if creating the
/// renderer or configuring the surface reports an error.
fn renderer_on<W>(
    context: &GpuContext,
    prewarmed: WgpuContext,
    window: &W,
    config: &WgpuSurfaceConfig,
    compositor_gpu: Option<CompositorGpuHint>,
) -> anyhow::Result<WgpuRenderer>
where
    W: rwh::HasWindowHandle + rwh::HasDisplayHandle + Debug + Send + Sync + Clone + 'static,
{
    let device = Arc::clone(&prewarmed.device);
    *context.borrow_mut() = Some(prewarmed);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let config = WgpuSurfaceConfig {
        size: config.size,
        transparent: config.transparent,
        preferred_present_mode: config.preferred_present_mode,
    };
    let renderer = WgpuRenderer::new(context.clone(), window, config, compositor_gpu);
    let result = adopted(renderer, gpui::block_on(scope.pop()));
    if result.is_err() {
        *context.borrow_mut() = None;
    }
    result
}

/// `renderer` if it was created without an error, `validation` being the
/// first validation error its creation reported.
fn adopted<R>(renderer: anyhow::Result<R>, validation: Option<wgpu::Error>) -> anyhow::Result<R> {
    match (renderer, validation) {
        (Ok(renderer), None) => Ok(renderer),
        (_, Some(error)) => Err(anyhow!("{error}")),
        (Err(error), None) => Err(error),
    }
}

#[cfg(test)]
mod tests;
