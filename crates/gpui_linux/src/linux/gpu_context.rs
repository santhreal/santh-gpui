//! The GPU context of a Linux client. A `gpu-context` thread selects the
//! adapter and creates the device while the client loads the system fonts
//! and runs the application's start. The first window takes the result, so
//! no window waits for the device unless it opens before the thread ends.
//!
//! The thread selects the adapter without a surface. The first window keeps
//! that context if the renderer of its surface is created on it with no
//! error; otherwise the window creates a context whose adapter is tested
//! against its surface, on the same instances. On a host with more than one
//! GPU, an adapter can report a surface as compatible and fail to configure
//! it. The window does not wait for its surface's configure, which runs on a
//! worker: a configure that fails on a context that has configured no surface
//! rejects the context, and the window's next draw replaces it
//! (`WgpuRenderer::needs_recovery`).

use std::{fmt::Debug, sync::Arc, thread::JoinHandle};

use anyhow::{Context as _, anyhow};
use gpui_wgpu::{
    CompositorGpuHint, DisplayInstances, GpuContext, WgpuContext, WgpuRenderer, WgpuSurfaceConfig,
    wgpu,
};
use raw_window_handle as rwh;

/// The result of a thread that reads a value `keep` holds alive. Dropping
/// it joins the thread before it releases `keep`.
pub(crate) struct Pending<T: Send + 'static, K> {
    thread: Option<JoinHandle<T>>,
    _keep: K,
}

impl<T: Send + 'static, K> Pending<T, K> {
    pub(crate) fn spawn(
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
/// created on the `gpu-context` thread, which reads the display connection
/// `K` holds open.
pub(crate) type PendingContext<K> = Pending<(DisplayInstances, anyhow::Result<WgpuContext>), K>;

/// Starts the thread that creates the GPU context of `instances`, created
/// by the caller. Its display holds its connection open. If the thread
/// cannot start, returns `instances` with the error, for the first window
/// to create the context from.
#[cfg(feature = "wayland")]
pub(crate) fn spawn_on(
    instances: DisplayInstances,
    compositor_gpu: Option<CompositorGpuHint>,
) -> Result<PendingContext<()>, (DisplayInstances, anyhow::Error)> {
    // A thread that fails to start drops its closure, so the instances
    // wait in a slot the caller can take them back from.
    let slot = Arc::new(parking_lot::Mutex::new(Some(instances)));
    let thread_slot = Arc::clone(&slot);
    Pending::spawn("gpu-context", (), move || {
        let instances = thread_slot
            .lock()
            .take()
            .expect("the gpu-context thread takes the instances once");
        let context = WgpuContext::for_display(&instances, compositor_gpu);
        (instances, context)
    })
    .map_err(|error| {
        let instances = slot
            .lock()
            .take()
            .expect("a thread that did not start left the instances");
        (instances, error)
    })
}

/// What a window's renderer is created from.
pub(crate) struct WindowGpu<K> {
    /// The context the client's windows share.
    pub context: GpuContext,
    pub compositor_gpu: Option<CompositorGpuHint>,
    /// The instances and the context from the GPU context thread, for the
    /// client's first window.
    pub pending: Option<PendingContext<K>>,
    /// Pinged by the worker that configures the window's surface as the
    /// configure returns.
    pub configured: calloop::ping::Ping,
}

impl<K> WindowGpu<K> {
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
            configured,
        } = self;
        let mut renderer = Self::create(context, compositor_gpu, pending, window, config)?;
        renderer.notify_configured(Arc::new(move || configured.ping()));
        Ok(renderer)
    }

    fn create<W>(
        context: GpuContext,
        compositor_gpu: Option<CompositorGpuHint>,
        pending: Option<PendingContext<K>>,
        window: &W,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<WgpuRenderer>
    where
        W: rwh::HasWindowHandle + rwh::HasDisplayHandle + Debug + Send + Sync + Clone + 'static,
    {
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
/// renderer reports an error. The surface's configure, which runs on a
/// worker outside this thread's error scope, is not waited for: its
/// failure rejects `prewarmed` as the window next draws.
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
