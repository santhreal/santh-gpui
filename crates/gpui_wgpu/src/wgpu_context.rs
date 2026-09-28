use crate::wgpu_renderer::PipelineCache;
#[cfg(not(target_family = "wasm"))]
use anyhow::Context as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::TextureFormat;

#[cfg(not(target_family = "wasm"))]
mod adapter_selection;
#[cfg(not(target_family = "wasm"))]
use adapter_selection::BackendTier;

mod instance;
#[cfg(not(target_family = "wasm"))]
pub use instance::DisplayInstances;
pub(crate) use instance::create_instance;

#[cfg(test)]
mod keepalive;
#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    backend: WgpuBackend,
    dual_source_blending: bool,
    color_texture_format: wgpu::TextureFormat,
    device_lost: Arc<AtomicBool>,
    /// The device has configured a window surface of the display: its
    /// adapter was selected by configuring one, or a renderer's first
    /// configure on it succeeded.
    surface_tested: AtomicBool,
    /// The shader modules, bind group layouts, and render pipelines of the
    /// renderers on `device`, dropped with the context.
    pub(crate) pipeline_cache: Arc<PipelineCache>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuBackend {
    BrowserWebGpu,
    Gl,
    Native(wgpu::Backend),
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WebBackendPreference {
    #[default]
    Auto,
    WebGpu,
    WebGl,
}

#[cfg(target_family = "wasm")]
pub struct PreparedWebGraphics {
    pub context: WgpuContext,
    pub surface: wgpu::Surface<'static>,
}

/// wgpu-core refuses to create a surface when neither the instance nor the surface
/// target carries a display handle, and `SurfaceTarget::Canvas` always passes `None`.
/// The WebGL2 backend never reads the handle (WebGPU bypasses wgpu-core entirely), so
/// a unit web display handle on the instance satisfies the check.
#[cfg(target_family = "wasm")]
#[derive(Debug)]
struct WebDisplaySource;

#[cfg(target_family = "wasm")]
impl raw_window_handle::HasDisplayHandle for WebDisplaySource {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::web())
    }
}

#[derive(Clone, Copy)]
pub struct CompositorGpuHint {
    pub vendor_id: u32,
    pub device_id: u32,
}

impl WgpuContext {
    /// Creates the context of `window`'s display from `instances` and the
    /// surface of `window`, selecting an adapter whose device configures
    /// the surface. `reject_software` excludes CPU adapters. The context
    /// comes from the first backend tier that selects an adapter.
    #[cfg(not(target_family = "wasm"))]
    pub fn for_window<W>(
        instances: &DisplayInstances,
        window: &W,
        compositor_gpu: Option<CompositorGpuHint>,
        reject_software: bool,
    ) -> anyhow::Result<(Self, wgpu::Surface<'static>)>
    where
        W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    {
        Self::by_backend_tier(instances, |tier, candidates| {
            let mut surfaces = Vec::with_capacity(candidates.len());
            let mut errors = Vec::new();
            for &instance in candidates {
                match create_surface(instance, window) {
                    Ok(surface) => surfaces.push((instance, surface)),
                    Err(error) => errors.push(format!("{error:#}")),
                }
            }
            anyhow::ensure!(
                !surfaces.is_empty(),
                "Failed to create surface: {}",
                errors.join("; ")
            );
            let sources: Vec<_> = surfaces
                .iter()
                .map(|(instance, surface)| (*instance, Some(surface)))
                .collect();
            let (context, source) =
                Self::new_with_options(&sources, tier, compositor_gpu, reject_software)?;
            Ok((context, surfaces.swap_remove(source).1))
        })
    }

    /// Creates the context of the windows of the display of `instances`,
    /// selecting the adapter without a surface. The context comes from the
    /// first backend tier that selects an adapter.
    #[cfg(not(target_family = "wasm"))]
    pub fn for_display(
        instances: &DisplayInstances,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self> {
        Self::by_backend_tier(instances, |tier, candidates| {
            let sources: Vec<_> = candidates
                .iter()
                .map(|&instance| (instance, None))
                .collect();
            Ok(Self::new_with_options(&sources, tier, compositor_gpu, false)?.0)
        })
    }

    /// Creates a surfaceless WgpuContext with no OS display surface.
    ///
    /// Uses deterministic adapter selection and the same device limits and features
    /// as the windowed path.
    #[cfg(not(target_family = "wasm"))]
    pub fn new_surfaceless(
        instance: wgpu::Instance,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self> {
        let sources = [(&instance, None)];
        Ok(Self::new_with_options(&sources, BackendTier::VulkanAndGl, compositor_gpu, false)?.0)
    }

    /// Creates a surfaceless WgpuContext rejecting software adapters.
    #[cfg(not(target_family = "wasm"))]
    pub fn new_surfaceless_rejecting_software(
        instance: wgpu::Instance,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self> {
        let sources = [(&instance, None)];
        Ok(Self::new_with_options(&sources, BackendTier::VulkanAndGl, compositor_gpu, true)?.0)
    }

    #[cfg(target_family = "wasm")]
    pub async fn new_web(
        canvas: &web_sys::HtmlCanvasElement,
        preference: WebBackendPreference,
    ) -> anyhow::Result<PreparedWebGraphics> {
        Self::new_web_with_backend(canvas, preference).await
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    async fn new_web_with_backend(
        canvas: &web_sys::HtmlCanvasElement,
        preference: WebBackendPreference,
    ) -> anyhow::Result<PreparedWebGraphics> {
        let backends = match preference {
            WebBackendPreference::Auto => wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            WebBackendPreference::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
            WebBackendPreference::WebGl => wgpu::Backends::GL,
        };
        let descriptor = wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: Some(Box::new(WebDisplaySource)),
        };
        let instance = if preference == WebBackendPreference::Auto {
            wgpu::util::new_instance_with_webgpu_detection(descriptor).await
        } else {
            create_instance(descriptor)
        };
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| {
                anyhow::anyhow!("Failed to create browser graphics surface: {error}")
            })?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "Failed to request a {preference:?} adapter compatible with the canvas: {error}"
                )
            })?;
        let adapter_info = adapter.get_info();
        let backend = match adapter_info.backend {
            wgpu::Backend::BrowserWebGpu => WgpuBackend::BrowserWebGpu,
            wgpu::Backend::Gl => WgpuBackend::Gl,
            backend => {
                anyhow::bail!(
                    "Browser graphics initialization selected unexpected backend {backend:?}"
                )
            }
        };

        let device_lost = Arc::new(AtomicBool::new(false));
        let (device, queue, dual_source_blending, color_texture_format) =
            Self::create_device(&adapter).await?;
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });
        log::info!(
            "Browser graphics initialized: requested={preference:?}, selected={backend:?}, \
             adapter={:?}, limits={:?}, dual_source_blending={dual_source_blending}",
            adapter_info.name,
            device.limits(),
        );

        let context = Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            backend,
            dual_source_blending,
            color_texture_format,
            device_lost,
            surface_tested: AtomicBool::new(false),
            pipeline_cache: Arc::default(),
        };
        Ok(PreparedWebGraphics { context, surface })
    }

    #[allow(clippy::disallowed_methods)]
    async fn create_device(
        adapter: &wgpu::Adapter,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool, TextureFormat)> {
        let dual_source_blending = adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);

        let mut required_features = wgpu::Features::empty();
        if dual_source_blending {
            required_features |= wgpu::Features::DUAL_SOURCE_BLENDING;
        } else {
            log::warn!(
                "Dual-source blending not available on this GPU. \
                Subpixel text antialiasing will be disabled."
            );
        }

        let color_atlas_texture_format = Self::select_color_texture_format(adapter)?;
        #[cfg(target_family = "wasm")]
        let required_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits())
        } else {
            wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits())
        };
        #[cfg(not(target_family = "wasm"))]
        let required_limits = wgpu::Limits::downlevel_defaults()
            .using_resolution(adapter.limits())
            .using_alignment(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpui_device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create wgpu device: {e}"))?;

        Ok((
            device,
            queue,
            dual_source_blending,
            color_atlas_texture_format,
        ))
    }

    /// Creates a wgpu::Instance with no display handle.
    ///
    /// `ZED_BACKEND`, or `WGPU_BACKEND` if it is unset, selects the backends
    /// of the instance: `vulkan`, `gl` or `opengl`, `metal`, `dx12`, or
    /// `all`. Otherwise the instance is Vulkan alone if one of its adapters
    /// ranks above every GL adapter, and Vulkan and GL if none does.
    /// Creating a GL instance loads every installed EGL vendor library, and
    /// the NVIDIA driver (580) deadlocks when one thread destroys an EGL
    /// context while another destroys a Vulkan device.
    #[cfg(not(target_family = "wasm"))]
    pub fn surfaceless_instance() -> wgpu::Instance {
        let selected = std::env::var("ZED_BACKEND")
            .or_else(|_| std::env::var("WGPU_BACKEND"))
            .ok()
            .and_then(|backend| match backend.to_lowercase().as_str() {
                "vulkan" => Some(wgpu::Backends::VULKAN),
                "gl" | "opengl" => Some(wgpu::Backends::GL),
                "metal" => Some(wgpu::Backends::METAL),
                "dx12" => Some(wgpu::Backends::DX12),
                "all" => Some(wgpu::Backends::all()),
                _ => None,
            });
        let surfaceless = |backends| {
            create_instance(wgpu::InstanceDescriptor {
                backends,
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            })
        };
        if let Some(backends) = selected {
            return surfaceless(backends);
        }
        let vulkan = surfaceless(wgpu::Backends::VULKAN);
        if BackendTier::Vulkan.selects_from(&vulkan) {
            return vulkan;
        }
        drop(vulkan);
        surfaceless(wgpu::Backends::VULKAN | wgpu::Backends::GL)
    }

    pub fn check_compatible_with_surface(&self, surface: &wgpu::Surface<'_>) -> anyhow::Result<()> {
        let caps = surface.get_capabilities(&self.adapter);
        if caps.formats.is_empty() {
            let info = self.adapter.get_info();
            anyhow::bail!(
                "Adapter {:?} (backend={:?}, device={:#06x}) is not compatible with the \
                 display surface for this window.",
                info.name,
                info.backend,
                info.device,
            );
        }
        Ok(())
    }

    fn select_color_texture_format(adapter: &wgpu::Adapter) -> anyhow::Result<wgpu::TextureFormat> {
        let required_usages = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let bgra_features = adapter.get_texture_format_features(wgpu::TextureFormat::Bgra8Unorm);
        let rgba_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm);
        #[cfg(target_family = "wasm")]
        if adapter.get_info().backend == wgpu::Backend::Gl
            && rgba_features.allowed_usages.contains(required_usages)
        {
            return Ok(wgpu::TextureFormat::Rgba8Unorm);
        }
        if bgra_features.allowed_usages.contains(required_usages) {
            return Ok(wgpu::TextureFormat::Bgra8Unorm);
        }
        if rgba_features.allowed_usages.contains(required_usages) {
            let info = adapter.get_info();
            log::warn!(
                "Adapter {} ({:?}) does not support Bgra8Unorm atlas textures with usages {:?}; \
                 falling back to Rgba8Unorm atlas textures.",
                info.name,
                info.backend,
                required_usages,
            );
            return Ok(wgpu::TextureFormat::Rgba8Unorm);
        }

        let info = adapter.get_info();
        Err(anyhow::anyhow!(
            "Adapter {} ({:?}, device={:#06x}) does not support a usable color atlas texture \
             format with usages {:?}. Bgra8Unorm allowed usages: {:?}; \
             Rgba8Unorm allowed usages: {:?}.",
            info.name,
            info.backend,
            info.device,
            required_usages,
            bgra_features.allowed_usages,
            rgba_features.allowed_usages,
        ))
    }
    pub fn backend(&self) -> WgpuBackend {
        self.backend
    }

    pub fn uses_webgl_instance_data(&self) -> bool {
        matches!(self.backend, WgpuBackend::Gl) && cfg!(target_family = "wasm")
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn color_texture_format(&self) -> wgpu::TextureFormat {
        self.color_texture_format
    }

    /// Returns true if the GPU device was lost (e.g., due to driver crash, suspend/resume).
    /// When this returns true, the context should be recreated.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// Returns a clone of the device_lost flag for sharing with renderers.
    pub(crate) fn device_lost_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.device_lost)
    }

    /// Whether the device has configured a window surface of the display.
    /// A context whose adapter was selected without a surface has not,
    /// until a renderer's first configure on it succeeds.
    pub fn surface_tested(&self) -> bool {
        self.surface_tested.load(Ordering::Relaxed)
    }

    /// Records that the device configured a window surface of the display.
    pub(crate) fn pass_surface_test(&self) {
        self.surface_tested.store(true, Ordering::Relaxed);
    }
}

/// Creates the surface of `window` on `instance`.
///
/// The caller keeps the window alive while the surface exists.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn create_surface<W>(
    instance: &wgpu::Instance,
    window: &W,
) -> anyhow::Result<wgpu::Surface<'static>>
where
    W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
{
    let raw_window_handle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?
        .as_raw();
    // A Vulkan instance of `DisplayInstances` has no display of its own, so
    // the surface names the window's.
    let raw_display_handle = window
        .display_handle()
        .map_err(|e| anyhow::anyhow!("Failed to get display handle: {e}"))?
        .as_raw();
    // Safety: the caller keeps the window alive while the surface exists.
    unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_display_handle),
            raw_window_handle,
        })
    }
    .context("Failed to create surface")
}
