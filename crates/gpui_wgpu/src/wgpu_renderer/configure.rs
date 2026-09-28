//! A window surface's first configure, run off the thread that draws.
//!
//! Creating a swapchain can take tens of milliseconds of driver time: the
//! NVIDIA Vulkan driver spends about 33 ms of CPU in `vkCreateSwapchainKHR`
//! for each new X11 window. On the thread that draws, that time stalls the
//! frames of every other window of the process. A new window's surface is
//! configured on a worker instead; the window skips its frames until the
//! worker finishes, and every later configure of the surface waits for it.

use std::sync::Arc;
#[cfg(not(target_family = "wasm"))]
use std::thread::JoinHandle;

#[cfg(not(target_family = "wasm"))]
use parking_lot::Mutex;

/// Called on the configure worker when a configure returns.
pub(super) type Notify = Arc<dyn Fn() + Send + Sync>;

/// The first configure of one surface, running or finished.
pub(super) struct Configuring {
    #[cfg(not(target_family = "wasm"))]
    worker: Option<Worker>,
}

#[cfg(not(target_family = "wasm"))]
struct Worker {
    thread: JoinHandle<()>,
    stage: Arc<Mutex<Stage>>,
}

/// Where the worker's configure is, as the thread that draws sees it.
#[cfg(not(target_family = "wasm"))]
enum Stage {
    /// Running, with the notify to call when it returns.
    Running(Option<Notify>),
    Returned,
}

/// Marks the configure returned, normally or by a panic, and calls the
/// notify registered while it ran.
#[cfg(not(target_family = "wasm"))]
struct Returned(Arc<Mutex<Stage>>);

#[cfg(not(target_family = "wasm"))]
impl Drop for Returned {
    fn drop(&mut self) {
        let stage = std::mem::replace(&mut *self.0.lock(), Stage::Returned);
        if let Stage::Running(Some(notify)) = stage {
            notify();
        }
    }
}

impl Configuring {
    /// A surface that needs no configure, or whose configure has returned.
    pub(super) const fn done() -> Self {
        Self {
            #[cfg(not(target_family = "wasm"))]
            worker: None,
        }
    }

    /// Configures `surface` for `device` on a worker thread. Where no thread
    /// can be started, and on the web, the configure runs here.
    pub(super) fn start(
        surface: &Arc<wgpu::Surface<'static>>,
        device: &Arc<wgpu::Device>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        #[cfg(not(target_family = "wasm"))]
        {
            let (worker_surface, worker_device, worker_config) =
                (Arc::clone(surface), Arc::clone(device), config.clone());
            match Self::spawn(move || worker_surface.configure(&worker_device, &worker_config)) {
                Ok(configuring) => return configuring,
                Err(err) => {
                    log::warn!("no thread for the surface configure ({err}); configuring here");
                }
            }
        }
        surface.configure(device, config);
        Self::done()
    }

    /// Runs `configure` on a new thread.
    #[cfg(not(target_family = "wasm"))]
    fn spawn(configure: impl FnOnce() + Send + 'static) -> std::io::Result<Self> {
        let stage = Arc::new(Mutex::new(Stage::Running(None)));
        let returned = Returned(Arc::clone(&stage));
        let thread = std::thread::Builder::new()
            .name("surface-configure".into())
            .spawn(move || {
                let _returned = returned;
                configure();
            })?;
        Ok(Self {
            worker: Some(Worker { thread, stage }),
        })
    }

    /// Calls `notify` on the worker when the configure returns, normally or
    /// by a panic. A configure that has already returned calls nothing.
    pub(super) fn when_done(&self, notify: Notify) {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = &self.worker
            && let Stage::Running(slot) = &mut *worker.stage.lock()
        {
            *slot = Some(notify);
        }
        #[cfg(target_family = "wasm")]
        drop(notify);
    }

    /// Whether the configure has returned. The worker is joined once it has:
    /// it marks the configure returned and calls the notify as it exits, so
    /// the thread that handles the notify finds the configure done.
    pub(super) fn is_done(&mut self) -> bool {
        #[cfg(not(target_family = "wasm"))]
        if self
            .worker
            .as_ref()
            .is_some_and(|worker| matches!(*worker.stage.lock(), Stage::Running(_)))
        {
            return false;
        }
        self.finish();
        true
    }

    /// Waits for the configure to return. A panic in the worker resumes
    /// here, where a configure on this thread would have raised it.
    pub(super) fn finish(&mut self) {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = self.worker.take()
            && let Err(panic) = worker.thread.join()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Drop for Configuring {
    /// A renderer dropped without `destroy` still waits for its worker, so
    /// no swapchain is created for a window that is already gone.
    fn drop(&mut self) {
        #[cfg(not(target_family = "wasm"))]
        if let Some(worker) = self.worker.take()
            && worker.thread.join().is_err()
        {
            log::error!("the surface configure panicked");
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;
