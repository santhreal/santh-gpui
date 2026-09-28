//! The order adapters are tried in, and the instances they are enumerated
//! from.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{CompositorGpuHint, DisplayInstances, WgpuBackend, WgpuContext};
use anyhow::Context as _;
use gpui_util::ResultExt;
use wgpu::TextureFormat;

/// An instance adapters are enumerated from, and the surface of the window
/// a context is created for, created on that instance.
pub(super) type Source<'a> = (&'a wgpu::Instance, Option<&'a wgpu::Surface<'static>>);

/// The instances a native context is selected from. A context is selected
/// from a Vulkan tier, and from the Vulkan and GL tier only when the Vulkan
/// tier selects no adapter.
///
/// Creating the GL instance loads every installed EGL vendor library and
/// initializes an EGL display on each, which takes tens of milliseconds;
/// the Vulkan tiers do not create it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BackendTier {
    /// Vulkan alone. It selects an adapter only if the adapter ranks above
    /// every adapter the GL backend reports, which the Vulkan and GL tier
    /// would then select as well.
    Vulkan,
    /// Vulkan alone, selecting any adapter: the first tier of a display
    /// whose GL instance is a last resort (see [`DisplayInstances`]).
    EveryVulkan,
    /// Vulkan and GL, selecting among the adapters of both.
    VulkanAndGl,
}

impl BackendTier {
    /// Whether the tier may select an adapter of `rank`.
    fn selects(self, rank: AdapterRank) -> bool {
        match self {
            Self::Vulkan => rank < AdapterRank::BEST_GL,
            Self::EveryVulkan | Self::VulkanAndGl => true,
        }
    }

    /// Whether the tier may select an adapter `instance` enumerates, at the
    /// rank the adapter has without `ZED_DEVICE_ID` and a compositor hint.
    pub(super) fn selects_from(self, instance: &wgpu::Instance) -> bool {
        gpui::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .any(|adapter| self.selects(AdapterRank::new(&adapter.get_info(), None, None)))
    }
}

/// The position of an adapter in selection order. Adapters are tried from
/// the lowest rank up; the fields compare in declaration order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AdapterRank {
    /// 0 for the device `ZED_DEVICE_ID` selects.
    user_override: u8,
    /// 0 for the GPU the display server renders on.
    compositor_match: u8,
    /// Discrete, integrated, other, virtual, CPU. OpenGL reports a GPU it
    /// does not recognize as other, which ranks it above a virtual GPU.
    device_type: u8,
    /// 0 for Vulkan, Metal, and DX12.
    backend: u8,
}

impl AdapterRank {
    /// The lowest rank of an adapter of the GL backend. It reports the
    /// device id as 0, which neither `ZED_DEVICE_ID` nor the compositor
    /// hint match, and the device type as integrated, other, or CPU.
    const BEST_GL: Self = Self {
        user_override: 1,
        compositor_match: 1,
        device_type: device_type_rank(wgpu::DeviceType::IntegratedGpu),
        backend: backend_rank(wgpu::Backend::Gl),
    };

    fn new(
        info: &wgpu::AdapterInfo,
        device_id_filter: Option<u32>,
        compositor_gpu: Option<&CompositorGpuHint>,
    ) -> Self {
        // The GL backend reports device 0 for every adapter, so a device
        // id matches only when it is not 0.
        let device_known = info.device != 0;
        let user_override = match device_id_filter {
            Some(id) if device_known && info.device == id => 0,
            _ => 1,
        };
        let compositor_match = match compositor_gpu {
            Some(hint)
                if device_known
                    && info.vendor == hint.vendor_id
                    && info.device == hint.device_id =>
            {
                0
            }
            _ => 1,
        };
        Self {
            user_override,
            compositor_match,
            device_type: device_type_rank(info.device_type),
            backend: backend_rank(info.backend),
        }
    }
}

const fn device_type_rank(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

const fn backend_rank(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
        _ => 1,
    }
}

/// The adapter a context is created on, and its device.
struct Selected {
    /// The index of the source the adapter was enumerated from.
    source: usize,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    dual_source_blending: bool,
    color_texture_format: TextureFormat,
}

impl WgpuContext {
    /// The result of `create` on the Vulkan tier's instance of `instances`,
    /// or, if that fails, on the Vulkan and GL tier's instances.
    pub(super) fn by_backend_tier<T>(
        instances: &DisplayInstances,
        mut create: impl FnMut(BackendTier, &[&wgpu::Instance]) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let vulkan_tier = if instances.gl_last_resort() {
            BackendTier::EveryVulkan
        } else {
            BackendTier::Vulkan
        };
        match instances.vulkan() {
            Ok(vulkan) => match create(vulkan_tier, &[vulkan]) {
                Ok(created) => return Ok(created),
                Err(error) => log::info!(
                    "No Vulkan adapter selected ({error:#}); selecting from the Vulkan and GL \
                     adapters"
                ),
            },
            Err(error) => {
                log::info!("No Vulkan instance ({error:#}); selecting from the GL adapters")
            }
        }
        let candidates: Vec<&wgpu::Instance> = instances
            .vulkan()
            .ok()
            .into_iter()
            .chain([instances.gl()])
            .collect();
        create(BackendTier::VulkanAndGl, &candidates)
    }

    /// Creates the context on the adapter of `sources` that `tier` selects,
    /// and returns it with the index of the adapter's source.
    pub(super) fn new_with_options(
        sources: &[Source<'_>],
        tier: BackendTier,
        compositor_gpu: Option<CompositorGpuHint>,
        reject_software: bool,
    ) -> anyhow::Result<(Self, usize)> {
        let device_id_filter = match std::env::var("ZED_DEVICE_ID") {
            Ok(val) => parse_pci_id(&val)
                .context("Failed to parse device ID from `ZED_DEVICE_ID` environment variable")
                .log_err(),
            Err(std::env::VarError::NotPresent) => None,
            err => {
                err.context("Failed to read value of `ZED_DEVICE_ID` environment variable")
                    .log_err();
                None
            }
        };

        let selected = gpui::block_on(Self::select_adapter_and_device(
            sources,
            tier,
            device_id_filter,
            compositor_gpu.as_ref(),
            reject_software,
        ))?;

        let device_lost = Arc::new(AtomicBool::new(false));
        selected.device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });

        let info = selected.adapter.get_info();
        log::info!("Selected GPU adapter: {:?} ({:?})", info.name, info.backend);

        let context = Self {
            instance: sources[selected.source].0.clone(),
            adapter: selected.adapter,
            device: Arc::new(selected.device),
            queue: Arc::new(selected.queue),
            backend: WgpuBackend::Native(info.backend),
            dual_source_blending: selected.dual_source_blending,
            color_texture_format: selected.color_texture_format,
            device_lost,
            // The adapter of a source with a surface is selected by
            // configuring the surface.
            surface_tested: AtomicBool::new(sources[selected.source].1.is_some()),
            pipeline_cache: Arc::default(),
        };
        Ok((context, selected.source))
    }

    /// Selects an adapter of `sources` that `tier` may select and creates
    /// its device, testing that the device configures the source's surface.
    /// That test is the only reliable one on hybrid GPU systems, where an
    /// adapter can report a surface as compatible and fail to configure it
    /// (NVIDIA reports Vulkan Wayland support and fails where the compositor
    /// runs on the Intel GPU).
    async fn select_adapter_and_device(
        sources: &[Source<'_>],
        tier: BackendTier,
        device_id_filter: Option<u32>,
        compositor_gpu: Option<&CompositorGpuHint>,
        reject_software: bool,
    ) -> anyhow::Result<Selected> {
        let mut adapters = Vec::new();
        for (source, (instance, _)) in sources.iter().enumerate() {
            let enumerated = instance.enumerate_adapters(wgpu::Backends::all()).await;
            adapters.extend(enumerated.into_iter().map(|adapter| (source, adapter)));
        }

        if adapters.is_empty() {
            anyhow::bail!("No GPU adapters found");
        }

        if let Some(device_id) = device_id_filter {
            log::info!("ZED_DEVICE_ID filter: {:#06x}", device_id);
        }

        // The backend, vendor, device, and name order adapters of one rank
        // deterministically.
        adapters.sort_by_key(|(_, adapter)| {
            let info = adapter.get_info();
            (
                AdapterRank::new(&info, device_id_filter, compositor_gpu),
                info.backend as u8,
                info.vendor,
                info.device,
                info.name,
            )
        });

        log::info!("Found {} GPU adapter(s):", adapters.len());
        for (_, adapter) in &adapters {
            let info = adapter.get_info();
            log::info!(
                "  - {} (vendor={:#06x}, device={:#06x}, backend={:?}, type={:?})",
                info.name,
                info.vendor,
                info.device,
                info.backend,
                info.device_type,
            );
        }

        for (source, adapter) in adapters {
            let info = adapter.get_info();

            if reject_software && info.device_type == wgpu::DeviceType::Cpu {
                log::info!(
                    "Skipping software renderer: {} ({:?})",
                    info.name,
                    info.backend
                );
                continue;
            }

            // Adapters are sorted, so no later adapter ranks higher.
            if !tier.selects(AdapterRank::new(&info, device_id_filter, compositor_gpu)) {
                anyhow::bail!(
                    "{} ({:?}, {:?}) and every adapter after it rank below the GL adapters",
                    info.name,
                    info.backend,
                    info.device_type
                );
            }

            log::info!("Testing adapter: {} ({:?})...", info.name, info.backend);

            let result = if let Some(surface) = sources[source].1 {
                Self::try_adapter_with_surface(&adapter, surface).await
            } else {
                Self::create_device(&adapter).await
            };

            match result {
                Ok((device, queue, dual_source_blending, color_texture_format)) => {
                    log::info!(
                        "Selected GPU (passed configuration test): {} ({:?})",
                        info.name,
                        info.backend
                    );
                    return Ok(Selected {
                        source,
                        adapter,
                        device,
                        queue,
                        dual_source_blending,
                        color_texture_format,
                    });
                }
                Err(e) => {
                    log::info!(
                        "  Adapter {} ({:?}) failed: {}, trying next...",
                        info.name,
                        info.backend,
                        e
                    );
                }
            }
        }

        anyhow::bail!("No GPU adapter found that can configure the display surface")
    }

    /// Creates the device of `adapter` and configures `surface` with it.
    /// Returns the device and queue if the configuration succeeds, so they
    /// can be reused.
    async fn try_adapter_with_surface(
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface<'_>,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool, TextureFormat)> {
        let caps = surface.get_capabilities(adapter);
        if caps.formats.is_empty() {
            anyhow::bail!("no compatible surface formats");
        }
        if caps.alpha_modes.is_empty() {
            anyhow::bail!("no compatible alpha modes");
        }

        let (device, queue, dual_source_blending, color_atlas_texture_format) =
            Self::create_device(adapter).await?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let test_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &test_config);

        let error = error_scope.pop().await;
        if let Some(e) = error {
            anyhow::bail!("surface configuration failed: {e}");
        }

        Ok((
            device,
            queue,
            dual_source_blending,
            color_atlas_texture_format,
        ))
    }
}

pub(super) fn parse_pci_id(id: &str) -> anyhow::Result<u32> {
    let mut id = id.trim();

    if id.starts_with("0x") || id.starts_with("0X") {
        id = &id[2..];
    }
    let is_hex_string = id.chars().all(|c| c.is_ascii_hexdigit());
    let is_4_chars = id.len() == 4;
    anyhow::ensure!(
        is_4_chars && is_hex_string,
        "Expected a 4 digit PCI ID in hexadecimal format"
    );

    u32::from_str_radix(id, 16).context("parsing PCI ID as hex")
}

#[cfg(test)]
mod tests;
