mod configure;
#[cfg(test)]
mod layer_mask_tests;
mod layers;
#[cfg(test)]
mod path_mask_tests;
mod pipeline_cache;
mod pipelines;
mod trial;

#[cfg(not(target_family = "wasm"))]
use crate::wgpu_context::create_surface;
use crate::{CompositorGpuHint, GpuContext, WgpuAtlas, WgpuContext};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use configure::{Configuring, Notify};
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, EdgeFadeMask, GpuSpecs,
    LayerMask, Path, Point, PrimitiveBatch, ScaledPixels, Scene, Size, TransformationMatrix,
    get_gamma_correction_ratios,
};
use layers::{LayerTextures, OpenLayer};
use log::warn;
use parking_lot::Mutex;
pub(crate) use pipeline_cache::PipelineCache;
use pipeline_cache::{ModuleKey, PipelineKey};
use pipelines::WgpuPipelines;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use trial::Trial;

const MAX_INSTANCE_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

const INSTANCE_TEXTURE_TEXEL_SIZE: u64 = 16;

/// Shader variant for backends with storage buffer support: the shared shader
/// logic plus the storage-buffer instance transport (recompiled).
const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
);

/// Shader variant for WebGL2, which has no storage buffers: the shared shader
/// logic plus the texture-based instance transport.
const WEBGL_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_webgl.wgsl"),
);

/// Subpixel text rendering requires dual-source blending, which WebGL2 lacks, so
/// this variant only ever runs with the storage-buffer transport. The `enable`
/// directive must precede all declarations.
const SUBPIXEL_SHADERS: &str = concat!(
    "enable dual_source_blending;\n",
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_subpixel.wgsl"),
);

fn least_common_multiple(left: u64, right: u64) -> u64 {
    let mut first = left;
    let mut second = right;
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    left / first * right
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodCorners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodContentMask {
    bounds: PodBounds,
    corner_radii: PodCorners,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodContentMask,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    /// The content mask of the path, whose rounded corners clip it.
    content_mask: ContentMask<ScaledPixels>,
    transformation: TransformationMatrix,
    /// The bounds of the path after its transformation, clipped to its
    /// content mask. A gradient spans them.
    bounds: Bounds<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

/// One frame allocation of instance data, ready to bind.
struct InstanceBinding {
    bind_group: wgpu::BindGroup,
    /// Index of the allocation's first instance within the bound data. Always
    /// zero on the storage-buffer path, where the binding offset already
    /// positions the array; on the WebGL texture path the shader indexes the
    /// shared instance texture absolutely, so draws must offset their
    /// instance (or vertex) ranges by this value.
    first_instance: u32,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    subpixel_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    backdrop_blurs: InstanceBinding,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
}

enum InstanceData {
    Storage(wgpu::Buffer),
    // WebGL2 has no storage buffers. A uint texture keeps the records available to both shader
    // stages while preserving integer and floating-point bit patterns exactly.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    },
}

/// The render target backing a WgpuRenderer (either an on-screen window surface or an offscreen texture).
enum WgpuRenderTarget {
    /// Shared with the worker that configures the surface first.
    Surface(Arc<wgpu::Surface<'static>>),
    Offscreen {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        format: wgpu::TextureFormat,
    },
}

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    target: WgpuRenderTarget,
    /// The pipelines of the target's key, shared with the renderers of the
    /// same key on `device`.
    pipelines: Arc<WgpuPipelines>,
    /// The cache of `device` that `pipelines` come from.
    pipeline_cache: Arc<PipelineCache>,
    atlas_sampler: wgpu::Sampler,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_data: InstanceData,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    /// The layers masked subtrees draw into, one per nesting depth.
    layers: LayerTextures,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    backdrop_texture: Option<wgpu::Texture>,
    backdrop_view: Option<wgpu::TextureView>,
    /// The complete frame, retained independently of swapchain copy support
    /// so backdrop blur never reads a non-copyable surface texture.
    retained_texture: Option<wgpu::Texture>,
    retained_view: Option<wgpu::TextureView>,
    retained_present_binding: Option<wgpu::BindGroup>,
    /// Whether `retained_texture` holds a complete frame at the current size.
    retained_valid: bool,
}

impl WgpuResources {
    fn invalidate_intermediate_textures(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.layers.clear();
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
        self.backdrop_texture = None;
        self.backdrop_view = None;
        self.retained_texture = None;
        self.retained_view = None;
        self.retained_present_binding = None;
        self.retained_valid = false;
    }

    /// Takes the pipelines of a target configured as `config`, rasterizing
    /// paths with `path_sample_count` samples, from the cache.
    fn select_pipelines(&mut self, config: &wgpu::SurfaceConfiguration, path_sample_count: u32) {
        let key = PipelineKey::new(self.pipelines.key().modules, config, path_sample_count);
        self.pipelines = self.pipeline_cache.pipelines(&self.device, key);
    }
}

/// The device-pixel rectangle a partial frame is limited to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scissor {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Scissor {
    /// Snaps `damage` outward to whole device pixels and clips it to the
    /// target. `None` when nothing inside the target is damaged.
    fn from_damage(damage: Bounds<ScaledPixels>, width: u32, height: u32) -> Option<Self> {
        let left = damage.origin.x.0.max(0.0).floor();
        let top = damage.origin.y.0.max(0.0).floor();
        let right = (damage.origin.x.0 + damage.size.width.0)
            .min(width as f32)
            .ceil();
        let bottom = (damage.origin.y.0 + damage.size.height.0)
            .min(height as f32)
            .ceil();
        if right <= left || bottom <= top {
            return None;
        }
        Some(Self {
            x: left as u32,
            y: top as u32,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
        })
    }

    fn apply(self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_scissor_rect(self.x, self.y, self.width, self.height);
    }
}

/// How one frame is drawn: the whole target, the declared region of the
/// retained frame, or nothing because the declared region is empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameExtent {
    Whole,
    Partial(Scissor),
    Nothing,
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    resources: Option<WgpuResources>,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_data_capacity: u64,
    max_instance_data_size: u64,
    instance_data_alignment: u64,
    uses_webgl_instance_data: bool,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    last_error: Arc<Mutex<Option<String>>>,
    failed_frame_count: u32,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// How much of the target the last `draw` repainted.
    last_frame_extent: FrameExtent,
    surface_configured: bool,
    /// The surface's first configure, which `draw` waits out.
    configuring: Configuring,
    /// Called on the configure worker when a first configure returns.
    configured_notify: Option<Notify>,
    /// Whether the surface's first configure keeps the context.
    trial: Trial,
    needs_redraw: bool,
}

impl WgpuRenderer {
    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let mut ctx_ref = gpu_context.borrow_mut();
        let (context, surface) = match ctx_ref.as_mut() {
            Some(context) => {
                // The caller keeps the window alive while the renderer exists,
                // and the renderer drops the surface before the window.
                let surface = create_surface(&context.instance, window)?;
                context.check_compatible_with_surface(&surface)?;
                (context, surface)
            }
            None => {
                let (context, surface) = WgpuContext::for_window(
                    gpu_context.instances(window),
                    window,
                    compositor_gpu,
                    false,
                )?;
                (ctx_ref.insert(context), surface)
            }
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(
            Some(gpu_context.clone()),
            context,
            WgpuRenderTarget::Surface(Arc::new(surface)),
            config,
            compositor_gpu,
            atlas,
        )
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_from_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_from_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(
            None,
            context,
            WgpuRenderTarget::Surface(Arc::new(surface)),
            config,
            None,
            atlas,
        )
    }

    /// Creates a new offscreen WgpuRenderer with an owned texture render target.
    // wgpu objects are neither `Send` nor `Sync` on the web, which runs the
    // renderer on one thread.
    #[cfg_attr(target_family = "wasm", allow(clippy::arc_with_non_send_sync))]
    pub fn new_offscreen(context: &WgpuContext, size: Size<DevicePixels>) -> anyhow::Result<Self> {
        let format = context.color_texture_format();
        let width = (size.width.0 as u32).max(1);
        let height = (size.height.0 as u32).max(1);

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_render_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let target = WgpuRenderTarget::Offscreen {
            texture,
            view,
            format,
        };

        let config = WgpuSurfaceConfig {
            size,
            transparent: true,
            preferred_present_mode: None,
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(None, context, target, config, None, atlas)
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        target: WgpuRenderTarget,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        // One capability query per surface: each is a round trip to the
        // window system.
        let (surface_format, transparent_alpha_mode, opaque_alpha_mode, surface_caps) =
            match &target {
                WgpuRenderTarget::Surface(surface) => {
                    let surface_caps = surface.get_capabilities(&context.adapter);
                    let preferred_formats = [
                        wgpu::TextureFormat::Bgra8Unorm,
                        wgpu::TextureFormat::Rgba8Unorm,
                    ];
                    let surface_format = preferred_formats
                        .iter()
                        .find(|f| surface_caps.formats.contains(f))
                        .copied()
                        .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
                        .or_else(|| surface_caps.formats.first().copied())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Surface reports no supported texture formats for adapter {:?}",
                                context.adapter.get_info().name
                            )
                        })?;

                    let pick_alpha_mode =
                    |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                        preferences
                            .iter()
                            .find(|p| surface_caps.alpha_modes.contains(p))
                            .copied()
                            .or_else(|| surface_caps.alpha_modes.first().copied())
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "Surface reports no supported alpha modes for adapter {:?}",
                                    context.adapter.get_info().name
                                )
                            })
                    };

                    let transparent_alpha_mode = pick_alpha_mode(&[
                        wgpu::CompositeAlphaMode::PreMultiplied,
                        wgpu::CompositeAlphaMode::Inherit,
                    ])?;

                    let opaque_alpha_mode = pick_alpha_mode(&[
                        wgpu::CompositeAlphaMode::Opaque,
                        wgpu::CompositeAlphaMode::Inherit,
                    ])?;
                    (
                        surface_format,
                        transparent_alpha_mode,
                        opaque_alpha_mode,
                        Some(surface_caps),
                    )
                }
                WgpuRenderTarget::Offscreen { format, .. } => {
                    // Note: For offscreen rendering on this platform, we explicitly use the color texture format
                    // selected by the WgpuContext (typically Bgra8Unorm on native platforms), matching the windowed path.
                    // A format change here changes the readback byte order.
                    // For offscreen rendering without a compositor, we use Auto/PostMultiplied for straight alpha.
                    (
                        *format,
                        wgpu::CompositeAlphaMode::Auto,
                        wgpu::CompositeAlphaMode::Opaque,
                        None,
                    )
                }
            };

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        // Prefer texture copies for presentation when supported; otherwise
        // a fullscreen pass presents the retained frame.
        let surface_config = wgpu::SurfaceConfiguration {
            usage: match &surface_caps {
                Some(surface_caps) => {
                    let copyable = surface_caps.usages.intersection(
                        wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                    );
                    wgpu::TextureUsages::RENDER_ATTACHMENT | copyable
                }
                None => {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                }
            },
            format: surface_format,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: match &surface_caps {
                Some(surface_caps) => config
                    .preferred_present_mode
                    .filter(|mode| surface_caps.present_modes.contains(mode))
                    .unwrap_or(wgpu::PresentMode::Fifo),
                None => wgpu::PresentMode::Fifo,
            },
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        let queue = Arc::clone(&context.queue);
        let rendering_params = RenderingParameters::new(&context.adapter, surface_format);
        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        let dual_source_blending =
            context.supports_dual_source_blending() && !uses_webgl_instance_data;
        let pipeline_cache = Arc::clone(&context.pipeline_cache);
        let pipelines = pipeline_cache.pipelines(
            &device,
            PipelineKey::new(
                ModuleKey {
                    dual_source_blending,
                    uses_webgl_instance_data,
                },
                &surface_config,
                rendering_params.path_sample_count,
            ),
        );
        let bind_group_layouts = pipelines.layouts();

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (
            instance_data,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
        ) = if uses_webgl_instance_data {
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            let max_instance_data_size = (u64::from(max_texture_dimension).pow(2)
                * INSTANCE_TEXTURE_TEXEL_SIZE)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_instance_data_size);
            let (instance_data, capacity) =
                Self::create_instance_texture(&device, initial_capacity, max_texture_dimension);
            (
                instance_data,
                capacity,
                max_instance_data_size,
                INSTANCE_TEXTURE_TEXEL_SIZE,
            )
        } else {
            // Every frame allocation is exposed as one storage-buffer binding, so
            // its backing buffer must satisfy both the allocation and binding limits.
            let max_buffer_size = device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_buffer_size);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: initial_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                InstanceData::Storage(buffer),
                initial_capacity,
                max_buffer_size,
                device.limits().min_storage_buffer_offset_alignment as u64,
            )
        };

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = context.adapter.get_info();

        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let last_error_clone = Arc::clone(&last_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            *last_error_clone.lock() = Some(error.to_string());
        }));
        // After the error handler: a configure error is a frame error, not a
        // panic on the worker. After the pipelines: NVIDIA's X11 driver
        // creates a swapchain and pipelines at the same time slower than one
        // after the other, by 36 ms to a cold start's first frame.
        let configuring = match &target {
            WgpuRenderTarget::Surface(surface) => {
                Configuring::start(surface, &device, &surface_config)
            }
            WgpuRenderTarget::Offscreen { .. } => Configuring::done(),
        };
        // A window's renderer shares its context with the display's other
        // windows through `gpu_context`.
        let trial = Trial::new(
            gpu_context.is_some() && matches!(target, WgpuRenderTarget::Surface(_)),
            context.surface_tested(),
        );

        let resources = WgpuResources {
            device,
            queue,
            target,
            pipelines,
            pipeline_cache,
            atlas_sampler,
            globals_buffer,
            globals_bind_group,
            path_globals_bind_group,
            instance_data,
            // Defer intermediate texture creation to first draw call via ensure_intermediate_textures().
            // This avoids panics when the device/surface is in an invalid state during initialization.
            path_intermediate_texture: None,
            path_intermediate_view: None,
            layers: LayerTextures::default(),
            path_msaa_texture: None,
            path_msaa_view: None,
            backdrop_texture: None,
            backdrop_view: None,
            retained_texture: None,
            retained_view: None,
            retained_present_binding: None,
            retained_valid: false,
        };

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            resources: Some(resources),
            surface_config,
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
            uses_webgl_instance_data,
            rendering_params,
            is_bgr: match surface_format {
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
                _ => false,
            },
            dual_source_blending,
            adapter_info,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            last_error,
            failed_frame_count: 0,
            device_lost: context.device_lost_flag(),
            surface_configured: true,
            configuring,
            configured_notify: None,
            trial,
            needs_redraw: false,
            last_frame_extent: FrameExtent::Whole,
        })
    }

    fn create_bind_group_layouts(
        device: &wgpu::Device,
        uses_webgl_instance_data: bool,
    ) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let instance_data_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: if uses_webgl_instance_data {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[instance_data_entry],
        });

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            texture,
            surfaces,
        }
    }

    fn create_instance_texture(
        device: &wgpu::Device,
        requested_capacity: u64,
        max_texture_dimension: u32,
    ) -> (InstanceData, u64) {
        let texel_count = requested_capacity.div_ceil(INSTANCE_TEXTURE_TEXEL_SIZE);
        let width = texel_count.min(u64::from(max_texture_dimension)).max(1) as u32;
        let height = texel_count
            .div_ceil(u64::from(width))
            .min(u64::from(max_texture_dimension))
            .max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("instance_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let capacity = u64::from(width) * u64::from(height) * INSTANCE_TEXTURE_TEXEL_SIZE;
        (
            InstanceData::Texture {
                texture,
                view,
                width,
                height,
            },
            capacity,
        )
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width != self.surface_config.width || height != self.surface_config.height {
            self.configuring.finish();
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            self.surface_config.width = clamped_width.max(1);
            self.surface_config.height = clamped_height.max(1);
            let surface_config = self.surface_config.clone();

            let Some(resources) = self.resources.as_mut() else {
                return;
            };

            // Wait for any in-flight GPU work to complete before destroying textures
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                warn!("Failed to poll device during resize: {e:?}");
            }

            // Destroy old textures before allocating new ones to avoid GPU memory spikes
            if let Some(ref texture) = resources.path_intermediate_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.path_msaa_texture {
                texture.destroy();
            }

            match &mut resources.target {
                WgpuRenderTarget::Surface(surface) => {
                    surface.configure(&resources.device, &surface_config);
                }
                WgpuRenderTarget::Offscreen {
                    texture,
                    view,
                    format,
                } => {
                    texture.destroy();
                    let new_texture = resources.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("offscreen_render_target"),
                        size: wgpu::Extent3d {
                            width: surface_config.width,
                            height: surface_config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: *format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let new_view = new_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    *texture = new_texture;
                    *view = new_view;
                }
            }
            // Invalidate intermediate textures - they will be lazily recreated
            // in draw() after we confirm the surface is healthy. This avoids
            // panics when the device/surface is in an invalid state during resize.
            resources.invalidate_intermediate_textures();
        }
    }

    fn ensure_intermediate_textures(&mut self) {
        if self.resources().path_intermediate_texture.is_some() {
            return;
        }

        let format = self.surface_config.format;
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let path_sample_count = self.rendering_params.path_sample_count;
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(t);
        resources.path_intermediate_view = Some(v);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;

        let (backdrop_texture, backdrop_view) = {
            let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("backdrop_texture"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        resources.backdrop_texture = Some(backdrop_texture);
        resources.backdrop_view = Some(backdrop_view);

        {
            let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("retained_frame"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            resources.retained_present_binding =
                resources
                    .pipelines
                    .present_retained
                    .as_ref()
                    .map(|pipeline| {
                        resources
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("retained_present_binding"),
                                layout: &pipeline.get_bind_group_layout(0),
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&view),
                                }],
                            })
                    });
            resources.retained_texture = Some(texture);
            resources.retained_view = Some(view);
            resources.retained_valid = false;
        }
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.is_bgr = is_bgr;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        if new_alpha_mode != self.surface_config.alpha_mode {
            self.configuring.finish();
            self.surface_config.alpha_mode = new_alpha_mode;
            let surface_config = self.surface_config.clone();
            let path_sample_count = self.rendering_params.path_sample_count;
            let Some(resources) = self.resources.as_mut() else {
                return;
            };
            if let WgpuRenderTarget::Surface(surface) = &mut resources.target {
                surface.configure(&resources.device, &surface_config);
            }
            resources.select_pipelines(&surface_config, path_sample_count);
            resources.invalidate_intermediate_textures();
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) -> bool {
        #[cfg(target_family = "wasm")]
        if self.device_lost() {
            if self.surface_configured {
                log::error!(
                    "Browser graphics context was lost; rendering has stopped. Reload the page to recover."
                );
                self.surface_configured = false;
            }
            return false;
        }

        // Bail out early if the surface has been unconfigured (e.g. during
        // Android background/rotation transitions).  Attempting to acquire
        // a texture from an unconfigured surface can block indefinitely on
        // some drivers (Adreno).
        if !self.surface_configured {
            return false;
        }
        // The first configure is still running on its worker. The frame is
        // not presented; the platform presents the same scene again once
        // the configure returns.
        if !self.configuring.is_done() {
            return false;
        }

        let last_error = self.last_error.lock().take();
        if let Some(error) = last_error {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );

            // TBD. Does retrying more actually help?
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            } else if self.failed_frame_count > 5 {
                if let Some(res) = self.resources.as_mut() {
                    res.invalidate_intermediate_textures();
                }
                self.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        enum AcquiredFrame {
            Surface(wgpu::SurfaceTexture),
            Offscreen,
        }

        let (acquired_frame, frame_view) = match &self.resources().target {
            WgpuRenderTarget::Surface(surface) => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame) => {
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    (AcquiredFrame::Surface(frame), view)
                }
                wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                    // Textures must be destroyed before the surface can be reconfigured.
                    drop(frame);
                    let surface_config = self.surface_config.clone();
                    let resources = self.resources_mut();
                    if let WgpuRenderTarget::Surface(s) = &resources.target {
                        s.configure(&resources.device, &surface_config);
                    }
                    return false;
                }
                wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                    let surface_config = self.surface_config.clone();
                    let resources = self.resources_mut();
                    if let WgpuRenderTarget::Surface(s) = &resources.target {
                        s.configure(&resources.device, &surface_config);
                    }
                    return false;
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    return false;
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    *self.last_error.lock() = Some("Surface texture validation error".to_string());
                    return false;
                }
            },
            WgpuRenderTarget::Offscreen { view, .. } => (AcquiredFrame::Offscreen, view.clone()),
        };

        // Now that we know the surface is healthy, ensure intermediate textures exist
        self.ensure_intermediate_textures();

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };

        let globals = GlobalParams {
            viewport_size: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            premultiplied_alpha: if self.surface_config.alpha_mode
                == wgpu::CompositeAlphaMode::PreMultiplied
            {
                1
            } else {
                0
            },
            pad: 0,
        };

        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };

        {
            let resources = self.resources();
            resources.queue.write_buffer(
                &resources.globals_buffer,
                0,
                bytemuck::bytes_of(&globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.path_globals_offset,
                bytemuck::bytes_of(&path_globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.gamma_offset,
                bytemuck::bytes_of(&gamma_params),
            );
        }

        let surface_texture = match &acquired_frame {
            AcquiredFrame::Surface(frame) => Some(&frame.texture),
            AcquiredFrame::Offscreen => None,
        };

        if let Err(error) = self.record_frame(scene, &frame_view, surface_texture) {
            log::error!("{error:#}");
            self.resources().queue.submit(std::iter::empty());
            return false;
        }

        match acquired_frame {
            AcquiredFrame::Surface(frame) => frame.present(),
            AcquiredFrame::Offscreen => {}
        }
        true
    }
    fn record_frame(
        &mut self,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
        surface_texture: Option<&wgpu::Texture>,
    ) -> Result<()> {
        let extent = match scene.damage {
            Some(damage) if self.resources().retained_valid => Scissor::from_damage(
                damage,
                self.surface_config.width,
                self.surface_config.height,
            )
            .map_or(FrameExtent::Nothing, FrameExtent::Partial),
            _ => FrameExtent::Whole,
        };
        let scissor = match extent {
            FrameExtent::Partial(scissor) => Some(scissor),
            FrameExtent::Whole | FrameExtent::Nothing => None,
        };

        // Handles are cloned (they are reference counted) so the draw loop
        // below can borrow `self` mutably.
        let retained_texture = self.resources().retained_texture.clone();
        let retained_view = self.resources().retained_view.clone();
        let offscreen_texture = match &self.resources().target {
            WgpuRenderTarget::Offscreen { texture, .. } => Some(texture.clone()),
            WgpuRenderTarget::Surface(_) => None,
        };
        // Backdrop blur and damage reuse require a readable retained frame,
        // even when the acquired surface supports only render attachments.
        let target_view = retained_view.as_ref().unwrap_or(frame_view);
        let frame_texture = surface_texture.or(offscreen_texture.as_ref());
        // The texture holding the frame drawn so far, which a backdrop blur
        // samples.
        let frame_texture_so_far = retained_texture.as_ref().or(frame_texture);

        let mut encoder =
            self.resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main_encoder"),
                });

        if extent != FrameExtent::Nothing {
            let mut instance_offset = 0;
            let instance_bindings = self
                .write_instances(scene, &mut instance_offset)
                .with_context(|| {
                    format!(
                        "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} monochrome sprites, {} subpixel sprites, {} polychrome sprites",
                        scene.paths.len(),
                        scene.shadows.len(),
                        scene.quads.len(),
                        scene.underlines.len(),
                        scene.monochrome_sprites.len(),
                        scene.subpixel_sprites.len(),
                        scene.polychrome_sprites.len(),
                    )
                })?;

            let load = match extent {
                FrameExtent::Whole => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                FrameExtent::Partial(_) | FrameExtent::Nothing => wgpu::LoadOp::Load,
            };
            let mut pass =
                Self::begin_frame_pass(&mut encoder, target_view, "main_pass", load, scissor);
            if scissor.is_some() {
                pass.set_pipeline(&self.resources().pipelines.clear);
                pass.draw(0..3, 0..1);
            }

            // The masks open at this point of the frame, innermost last. A
            // frame without masks leaves it empty and creates no layer.
            let mut open_layers: Vec<OpenLayer> = Vec::new();
            let mut deepest_layer = 0;
            for batch in scene.batches() {
                match batch {
                    PrimitiveBatch::Quads(range) => self.draw_instances(
                        &instance_bindings.quads,
                        &self.resources().pipelines.quads,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Shadows(range) => self.draw_instances(
                        &instance_bindings.shadows,
                        &self.resources().pipelines.shadows,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Paths(range) => {
                        let paths = &scene.paths[range];
                        if paths.is_empty() {
                            continue;
                        }

                        drop(pass);
                        let rasterized = self.draw_paths_to_intermediate(
                            &mut encoder,
                            paths,
                            &mut instance_offset,
                        )?;

                        pass = Self::begin_frame_pass(
                            &mut encoder,
                            layers::draw_target(&open_layers, target_view),
                            "main_pass_continued",
                            wgpu::LoadOp::Load,
                            scissor,
                        );

                        if rasterized {
                            self.draw_paths_from_intermediate(
                                paths,
                                &mut instance_offset,
                                &mut pass,
                            )?;
                        }
                    }
                    PrimitiveBatch::Underlines(range) => self.draw_instances(
                        &instance_bindings.underlines,
                        &self.resources().pipelines.underlines,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => self.draw_sprites(
                        &instance_bindings.monochrome_sprites,
                        texture_id,
                        &self.resources().pipelines.mono_sprites,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                        let resources = self.resources();
                        self.draw_sprites(
                            &instance_bindings.subpixel_sprites,
                            texture_id,
                            resources
                                .pipelines
                                .subpixel_sprites
                                .as_ref()
                                .unwrap_or(&resources.pipelines.mono_sprites),
                            instance_range(range),
                            &mut pass,
                        );
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => self.draw_sprites(
                        &instance_bindings.polychrome_sprites,
                        texture_id,
                        &self.resources().pipelines.poly_sprites,
                        instance_range(range),
                        &mut pass,
                    ),
                    // Surfaces are macOS-only for video playback and are not
                    // implemented by the WGPU renderer.
                    PrimitiveBatch::Surfaces(_surfaces) => {}
                    PrimitiveBatch::StartLayerMask(mask) => {
                        drop(pass);
                        let view = self.layer_view(open_layers.len());
                        pass = Self::begin_frame_pass(
                            &mut encoder,
                            &view,
                            "layer_mask_pass",
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            scissor,
                        );
                        open_layers.push(OpenLayer { mask, view });
                        deepest_layer = deepest_layer.max(open_layers.len());
                    }
                    PrimitiveBatch::EndLayerMask => {
                        // An end without a start has no layer to composite.
                        let Some(layer) = open_layers.pop() else {
                            continue;
                        };
                        drop(pass);
                        pass = self.composite_layer(
                            &mut encoder,
                            layer,
                            layers::draw_target(&open_layers, target_view),
                            scissor,
                            &mut instance_offset,
                        )?;
                    }
                    PrimitiveBatch::BackdropBlurs(range) => {
                        let backdrop_blurs = &scene.backdrop_blurs[range.clone()];
                        if backdrop_blurs.is_empty() {
                            continue;
                        }

                        drop(pass);

                        let resources = self.resources();
                        if let (Some(frame_tex), Some(backdrop_tex), Some(backdrop_view)) = (
                            frame_texture_so_far,
                            resources.backdrop_texture.as_ref(),
                            resources.backdrop_view.as_ref(),
                        ) {
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: frame_tex,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: backdrop_tex,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: self.surface_config.width,
                                    height: self.surface_config.height,
                                    depth_or_array_layers: 1,
                                },
                            );

                            let backdrop_bind_group = self
                                .create_texture_bind_group("backdrop_bind_group", backdrop_view);

                            pass = Self::begin_frame_pass(
                                &mut encoder,
                                layers::draw_target(&open_layers, target_view),
                                "main_pass_backdrop_blur",
                                wgpu::LoadOp::Load,
                                scissor,
                            );

                            let instances = &instance_bindings.backdrop_blurs;
                            let pipeline = &self.resources().pipelines.backdrop_blur;
                            let range = instance_range(range);
                            if !range.is_empty() {
                                pass.set_pipeline(pipeline);
                                pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
                                pass.set_bind_group(1, &instances.bind_group, &[]);
                                pass.set_bind_group(2, &backdrop_bind_group, &[]);
                                pass.draw(
                                    0..4,
                                    instances.first_instance + range.start
                                        ..instances.first_instance + range.end,
                                );
                            }
                        } else {
                            pass = Self::begin_frame_pass(
                                &mut encoder,
                                layers::draw_target(&open_layers, target_view),
                                "main_pass_continued",
                                wgpu::LoadOp::Load,
                                scissor,
                            );
                        }
                    }
                }
            }
            // A mask the scene left open closes at the end of the frame.
            while let Some(layer) = open_layers.pop() {
                drop(pass);
                pass = self.composite_layer(
                    &mut encoder,
                    layer,
                    layers::draw_target(&open_layers, target_view),
                    scissor,
                    &mut instance_offset,
                )?;
            }
            drop(pass);
            self.resources_mut().layers.end_frame(deepest_layer);
        }

        // Swapchain contents are undefined, so present the complete retained
        // frame even when only part of it was repainted.
        if let (Some(pipeline), Some(binding)) = (
            self.resources().pipelines.present_retained.as_ref(),
            self.resources().retained_present_binding.as_ref(),
        ) {
            let mut pass = Self::begin_frame_pass(
                &mut encoder,
                frame_view,
                "present_retained",
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                None,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, binding, &[]);
            pass.draw(0..3, 0..1);
        } else if let (Some(retained_texture), Some(target_texture)) =
            (retained_texture.as_ref(), frame_texture)
        {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: retained_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: target_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.resources()
            .queue
            .submit(std::iter::once(encoder.finish()));
        self.last_frame_extent = extent;
        if retained_texture.is_some() {
            self.resources_mut().retained_valid = true;
        }
        Ok(())
    }

    /// Begins a pass onto the frame target. A partial frame re-applies its
    /// scissor here because scissor state does not outlive a pass.
    fn begin_frame_pass<'encoder>(
        encoder: &'encoder mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        label: &'static str,
        load: wgpu::LoadOp<wgpu::Color>,
        scissor: Option<Scissor>,
    ) -> wgpu::RenderPass<'encoder> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        if let Some(scissor) = scissor {
            scissor.apply(&mut pass);
        }
        pass
    }

    fn write_instances(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> Result<InstanceBindings> {
        Ok(InstanceBindings {
            quads: self.write_instance_binding(
                "quads_bind_group",
                instance_offset,
                &scene.quads,
            )?,
            shadows: self.write_instance_binding(
                "shadows_bind_group",
                instance_offset,
                &scene.shadows,
            )?,
            underlines: self.write_instance_binding(
                "underlines_bind_group",
                instance_offset,
                &scene.underlines,
            )?,
            monochrome_sprites: self.write_instance_binding(
                "monochrome_sprites_bind_group",
                instance_offset,
                &scene.monochrome_sprites,
            )?,
            subpixel_sprites: self.write_instance_binding(
                "subpixel_sprites_bind_group",
                instance_offset,
                &scene.subpixel_sprites,
            )?,
            polychrome_sprites: self.write_instance_binding(
                "polychrome_sprites_bind_group",
                instance_offset,
                &scene.polychrome_sprites,
            )?,
            backdrop_blurs: self.write_instance_binding(
                "backdrop_blurs_bind_group",
                instance_offset,
                &scene.backdrop_blurs,
            )?,
        })
    }

    fn create_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.pipelines.layouts().texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            })
    }

    fn draw_instances(
        &self,
        instances: &InstanceBinding,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.draw(
            0..4,
            instances.first_instance + range.start..instances.first_instance + range.end,
        );
    }

    fn draw_sprites(
        &self,
        sprite_instances: &InstanceBinding,
        texture_id: AtlasTextureId,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        let texture_info = self.atlas.get_texture_info(texture_id);
        let texture =
            self.create_texture_bind_group("atlas_texture_bind_group", &texture_info.view);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &sprite_instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            sprite_instances.first_instance + range.start
                ..sprite_instances.first_instance + range.end,
        );
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        let first_path = &paths[0];
        let sprites: Vec<PathSprite> = if paths.last().map(|p| &p.order) == Some(&first_path.order)
        {
            paths
                .iter()
                .map(|p| PathSprite {
                    bounds: p.transformation.apply_to_bounds(p.clipped_bounds()),
                })
                .collect()
        } else {
            let mut bounds = first_path
                .transformation
                .apply_to_bounds(first_path.clipped_bounds());
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.transformation.apply_to_bounds(path.clipped_bounds()));
            }
            vec![PathSprite { bounds }]
        };

        let Some(path_intermediate_view) = self.resources().path_intermediate_view.clone() else {
            return Ok(());
        };
        let instances =
            self.write_instance_binding("path_sprites_bind_group", instance_offset, &sprites)?;
        let texture = self.create_texture_bind_group(
            "path_intermediate_texture_bind_group",
            &path_intermediate_view,
        );
        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.paths);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            instances.first_instance..instances.first_instance + sprites.len() as u32,
        );
        Ok(())
    }

    fn draw_paths_to_intermediate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
    ) -> Result<bool> {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path
                .transformation
                .apply_to_bounds(path.bounds)
                .intersect(&path.content_mask.bounds);
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                content_mask: path.content_mask,
                transformation: path.transformation,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return Ok(false);
        }

        let vertex_binding = self.write_instance_binding(
            "path_rasterization_bind_group",
            instance_offset,
            &vertices,
        )?;

        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Ok(false);
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
            // The path rasterization shader loads records by vertex index
            // rather than instance index, so the allocation's base shifts the
            // vertex range here.
            pass.draw(
                vertex_binding.first_instance
                    ..vertex_binding.first_instance + vertices.len() as u32,
                0..1,
            );
        }

        Ok(true)
    }

    /// The layer a mask opened at nesting `depth` draws into, sized to the
    /// frame.
    fn layer_view(&mut self, depth: usize) -> wgpu::TextureView {
        let format = self.surface_config.format;
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let resources = self.resources_mut();
        resources
            .layers
            .view(&resources.device, depth, format, width, height)
    }

    /// Composites the closed `layer` onto `parent` through its mask, inside
    /// the mask's bounds and `scissor`, and returns the pass that continues
    /// drawing onto `parent`.
    fn composite_layer<'encoder>(
        &mut self,
        encoder: &'encoder mut wgpu::CommandEncoder,
        layer: OpenLayer,
        parent: &wgpu::TextureView,
        scissor: Option<Scissor>,
        instance_offset: &mut u64,
    ) -> Result<wgpu::RenderPass<'encoder>> {
        match layer.mask {
            LayerMask::Path(path) => {
                // Paths inside the subtree reuse the path intermediate
                // texture, so the mask path is rasterized when it closes.
                let rasterized = self.draw_paths_to_intermediate(
                    encoder,
                    std::slice::from_ref(&path),
                    instance_offset,
                )?;
                let mut pass = Self::begin_frame_pass(
                    encoder,
                    parent,
                    "path_mask_composite",
                    wgpu::LoadOp::Load,
                    scissor,
                );
                // A path without vertices covers nothing, so its layer
                // contributes nothing.
                if rasterized {
                    self.draw_path_mask_composite(&path, &layer.view, instance_offset, &mut pass)?;
                }
                Ok(pass)
            }
            LayerMask::EdgeFade(fade) => {
                let mut pass = Self::begin_frame_pass(
                    encoder,
                    parent,
                    "edge_fade_composite",
                    wgpu::LoadOp::Load,
                    scissor,
                );
                self.draw_edge_fade_composite(&fade, &layer.view, instance_offset, &mut pass)?;
                Ok(pass)
            }
        }
    }

    /// Composites `layer` through the coverage of `path`, which the path
    /// intermediate texture holds.
    fn draw_path_mask_composite(
        &mut self,
        path: &Path<ScaledPixels>,
        layer: &wgpu::TextureView,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        let sprite = PathSprite {
            bounds: path.transformation.apply_to_bounds(path.clipped_bounds()),
        };
        let Some(path_intermediate_view) = self.resources().path_intermediate_view.clone() else {
            return Ok(());
        };

        let instances = self.write_instance_binding(
            "path_mask_composite_bind_group",
            instance_offset,
            &[sprite],
        )?;
        let layer_bind = self.create_texture_bind_group("path_mask_layer_bind_group", layer);
        let path_mask_bind =
            self.create_texture_bind_group("path_mask_texture_bind_group", &path_intermediate_view);

        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.path_mask_composite);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &layer_bind, &[]);
        pass.set_bind_group(3, &path_mask_bind, &[]);
        pass.draw(0..4, instances.first_instance..instances.first_instance + 1);
        Ok(())
    }

    /// Composites `layer` through the edge fade `fade`.
    fn draw_edge_fade_composite(
        &mut self,
        fade: &EdgeFadeMask,
        layer: &wgpu::TextureView,
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        let instances = self.write_instance_binding(
            "edge_fade_composite_bind_group",
            instance_offset,
            std::slice::from_ref(fade),
        )?;
        let layer_bind = self.create_texture_bind_group("edge_fade_layer_bind_group", layer);

        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.edge_fade_composite);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &layer_bind, &[]);
        pass.draw(0..4, instances.first_instance..instances.first_instance + 1);
        Ok(())
    }

    fn write_instance_binding<T>(
        &mut self,
        label: &str,
        instance_offset: &mut u64,
        instances: &[T],
    ) -> Result<InstanceBinding> {
        let data = unsafe { Self::instance_bytes(instances) };
        // wgpu rejects zero-sized bindings, so empty primitive arrays still
        // reserve the 16-byte minimum.
        let size = (data.len() as u64).max(16);
        let stride = (std::mem::size_of::<T>() as u64).max(1);
        let (alignment, allocation_size) = if self.uses_webgl_instance_data {
            // The texture transport has no binding offset: the shader indexes
            // the instance texture absolutely, so each allocation must start on
            // a whole instance (a stride multiple) and a whole texel, and must
            // end on a texel boundary so the zero padding of its final partial
            // texel cannot overlap the next allocation.
            (
                least_common_multiple(self.instance_data_alignment, stride),
                size.next_multiple_of(INSTANCE_TEXTURE_TEXEL_SIZE),
            )
        } else {
            (self.instance_data_alignment.max(1), size)
        };
        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + allocation_size > self.instance_data_capacity {
            self.grow_instance_data(allocation_size)?;
            offset = 0;
        }
        *instance_offset = offset + allocation_size;

        let first_instance = if self.uses_webgl_instance_data {
            u32::try_from(offset / stride).context("instance index exceeds u32 range")?
        } else {
            0
        };

        let resources = self.resources();
        if !data.is_empty() {
            match &resources.instance_data {
                InstanceData::Storage(buffer) => resources.queue.write_buffer(buffer, offset, data),
                InstanceData::Texture { .. } => {
                    Self::write_instance_texture(resources, offset, data)
                }
            }
        }
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.pipelines.layouts().instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match &resources.instance_data {
                        InstanceData::Storage(buffer) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset,
                                size: NonZeroU64::new(size),
                            })
                        }
                        InstanceData::Texture { view, .. } => {
                            wgpu::BindingResource::TextureView(view)
                        }
                    },
                }],
            });
        Ok(InstanceBinding {
            bind_group,
            first_instance,
        })
    }

    fn write_instance_texture(resources: &WgpuResources, offset: u64, data: &[u8]) {
        let InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } = &resources.instance_data
        else {
            return;
        };
        let mut byte_offset = 0usize;
        let mut texel_offset = offset / INSTANCE_TEXTURE_TEXEL_SIZE;
        while byte_offset < data.len() {
            let x = (texel_offset % u64::from(*width)) as u32;
            let y = (texel_offset / u64::from(*width)) as u32;
            if y >= *height {
                // The capacity check in write_instance_binding should make this
                // unreachable. Truncating silently would leave stale bytes in the
                // texture and draw garbage for the remaining instances.
                debug_assert!(
                    false,
                    "instance texture write out of bounds: row {y} >= height {}",
                    *height
                );
                log::error!(
                    "instance texture write out of bounds; dropping {} bytes of instance data",
                    data.len() - byte_offset
                );
                return;
            }
            let available_texels = u64::from(*width - x);
            let remaining_bytes = data.len() - byte_offset;
            let complete_texels = remaining_bytes as u64 / INSTANCE_TEXTURE_TEXEL_SIZE;
            let texels = complete_texels.min(available_texels);
            if texels > 0 {
                let byte_count = (texels * INSTANCE_TEXTURE_TEXEL_SIZE) as usize;
                resources.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[byte_offset..byte_offset + byte_count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(byte_count as u32),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: texels as u32,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                byte_offset += byte_count;
                texel_offset += texels;
                continue;
            }

            let mut final_texel = [0; INSTANCE_TEXTURE_TEXEL_SIZE as usize];
            final_texel[..remaining_bytes].copy_from_slice(&data[byte_offset..]);
            resources.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &final_texel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            break;
        }
    }

    fn grow_instance_data(&mut self, required: u64) -> Result<()> {
        let capacity = (self.instance_data_capacity * 2)
            .max(required.next_power_of_two())
            .min(self.max_instance_data_size);
        anyhow::ensure!(
            capacity >= required,
            "instance data needs {required} bytes, above the maximum of {}",
            self.max_instance_data_size
        );
        anyhow::ensure!(
            capacity > self.instance_data_capacity,
            "frame instance data exceeds the {}-byte maximum",
            self.max_instance_data_size
        );
        log::debug!(
            "instance data grown from {} to {capacity}",
            self.instance_data_capacity
        );
        // Bind groups created earlier in the frame keep the previous buffer or
        // texture alive, so allocations written before the grow remain valid;
        // only subsequent writes land in the new allocation.
        let uses_webgl_instance_data = self.uses_webgl_instance_data;
        let resources = self.resources_mut();
        if uses_webgl_instance_data {
            let max_texture_dimension = resources.device.limits().max_texture_dimension_2d;
            let (instance_data, actual_capacity) =
                Self::create_instance_texture(&resources.device, capacity, max_texture_dimension);
            resources.instance_data = instance_data;
            self.instance_data_capacity = actual_capacity;
        } else {
            resources.instance_data =
                InstanceData::Storage(resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance_buffer"),
                    size: capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instance_data_capacity = capacity;
        }
        Ok(())
    }

    /// Mark the surface as unconfigured so rendering is skipped until a new
    /// surface is provided via [`replace_surface`](Self::replace_surface).
    ///
    /// This does **not** drop the renderer — the device, queue, atlas, and
    /// pipelines stay alive.  Use this when the native window is destroyed
    /// (e.g. Android `TerminateWindow`) but you intend to re-create the
    /// surface later without losing cached atlas textures.
    pub fn unconfigure_surface(&mut self) {
        self.configuring.finish();
        self.surface_configured = false;
        // Drop intermediate textures since they reference the old surface size.
        if let Some(res) = self.resources.as_mut() {
            res.invalidate_intermediate_textures();
        }
    }

    /// Replace the wgpu surface with a new one (e.g. after Android destroys
    /// and recreates the native window).  Keeps the device, queue, atlas, and
    /// all pipelines intact so cached `AtlasTextureId`s remain valid.
    ///
    /// The `instance` **must** be the same [`wgpu::Instance`] that was used to
    /// create the adapter and device (i.e. from the [`WgpuContext`]).  Using a
    /// different instance will cause a "Device does not exist" panic because
    /// the wgpu device is bound to its originating instance.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle + HasDisplayHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        self.configuring.finish();
        let surface = create_surface(instance, window)?;

        let width = (config.size.width.0 as u32).max(1);
        let height = (config.size.height.0 as u32).max(1);

        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface_config.alpha_mode = alpha_mode;
        if let Some(mode) = config.preferred_present_mode {
            self.surface_config.present_mode = mode;
        }

        {
            let res = self
                .resources
                .as_mut()
                .expect("GPU resources not available");
            surface.configure(&res.device, &self.surface_config);
            res.target = WgpuRenderTarget::Surface(Arc::new(surface));

            // Invalidate intermediate textures — they'll be recreated lazily.
            res.invalidate_intermediate_textures();
        }

        self.surface_configured = true;

        Ok(())
    }

    pub fn destroy(&mut self) {
        // Release surface-bound GPU resources eagerly so the underlying native
        // window can be destroyed before the renderer itself is dropped. No
        // swapchain is created for it after this returns.
        self.configuring.finish();
        self.resources.take();
    }

    /// Waits for the surface's first configure, which runs on a worker, and
    /// reports the error the device raised while it ran. The error stays for
    /// the next `draw` to count as a failed frame.
    pub fn finish_configure(&mut self) -> anyhow::Result<()> {
        self.configuring.finish();
        match self.last_error.lock().as_deref() {
            Some(error) => Err(anyhow::anyhow!("configuring the surface failed: {error}")),
            None => Ok(()),
        }
    }

    /// Whether the surface's first configure is still running on its worker.
    /// A `draw` until then presents nothing.
    pub fn is_configuring(&mut self) -> bool {
        !self.configuring.is_done()
    }

    /// Calls `notify` on the configure worker each time a first configure of
    /// the surface returns: the one running now, and the one `recover`
    /// starts. A configure that has already returned calls nothing.
    pub fn notify_configured(&mut self, notify: Arc<dyn Fn() + Send + Sync>) {
        self.configuring.when_done(Arc::clone(&notify));
        self.configured_notify = Some(notify);
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Whether `recover` must run before the next `draw`: the device was
    /// lost, or the surface's first configure failed on a context that has
    /// configured no window surface. Reads the first configure's result
    /// once the configure has returned.
    pub fn needs_recovery(&mut self) -> bool {
        if self.trial == Trial::Pending && self.configuring.is_done() {
            self.settle_trial();
        }
        self.device_lost() || self.trial == Trial::Rejected
    }

    /// Settles the trial of the renderer's context by the surface's first
    /// configure, which has returned. A configure that succeeded on the
    /// display's context records that the context configured a surface;
    /// one that failed on a context that has configured none rejects the
    /// context, and its error goes with it instead of failing a frame.
    fn settle_trial(&mut self) {
        let failed = self.last_error.lock().is_some();
        let surface_tested = {
            let shared = self.context.as_ref().map(GpuContext::borrow);
            let own = shared
                .as_deref()
                .and_then(Option::as_ref)
                .filter(|context| {
                    self.resources
                        .as_ref()
                        .is_some_and(|resources| Arc::ptr_eq(&resources.device, &context.device))
                });
            if let Some(context) = own
                && !failed
            {
                context.pass_surface_test();
            }
            own.is_some_and(WgpuContext::surface_tested)
        };
        self.trial = self.trial.settle(failed, surface_tested);
        if self.trial == Trial::Rejected {
            let error = self.last_error.lock().take().unwrap_or_default();
            log::info!(
                "The GPU context selected without a surface cannot configure this window's \
                 surface ({error}); creating one selected by configuring the surface"
            );
        }
    }

    /// Returns true if a redraw is needed because GPU state was cleared.
    /// Calling this method clears the flag.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Recreates the renderer on a new context: after its device was lost,
    /// or after its surface's first configure rejected its context.
    ///
    /// Call this after `needs_recovery()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        self.configuring.finish();
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");
        let lost = self.device_lost();
        let rejected = self.trial == Trial::Rejected;

        // Another window may have recovered the context already. A rejected
        // context is replaced only while it is still the display's.
        let needs_new_context = gpu_context.borrow().as_ref().is_none_or(|ctx| {
            ctx.device_lost()
                || rejected
                    && self
                        .resources
                        .as_ref()
                        .is_some_and(|resources| Arc::ptr_eq(&resources.device, &ctx.device))
        });

        let surface = if needs_new_context {
            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.resources = None;
            *gpu_context.borrow_mut() = None;

            if lost {
                log::warn!("GPU device lost, recreating context...");
                // Wait briefly for the GPU driver to stabilize, then try to
                // recreate the context without software renderers. If this
                // fails the caller should request another frame and retry —
                // the real GPU may need more time to come back (e.g. after
                // suspend/resume).
                std::thread::sleep(std::time::Duration::from_millis(350));
            }

            let (new_context, surface) = WgpuContext::for_window(
                gpu_context.instances(window),
                window,
                self.compositor_gpu,
                lost,
            )?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, window)?
        };

        let config = WgpuSurfaceConfig {
            size: gpui::Size {
                width: gpui::DevicePixels(self.surface_config.width as i32),
                height: gpui::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let gpu_context = gpu_context.clone();
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.resources = None;
        self.atlas.handle_device_lost(context);

        let recovered = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            WgpuRenderTarget::Surface(Arc::new(surface)),
            config,
            self.compositor_gpu,
            self.atlas.clone(),
        )?;
        let configured_notify = self.configured_notify.take();
        *self = recovered;
        if let Some(notify) = configured_notify {
            self.notify_configured(notify);
        }

        log::info!("GPU recovery complete");
        Ok(())
    }

    /// Reads back pixels from the offscreen render target as tightly packed RGBA8 rows.
    ///
    /// Copies the rendered texture to a host-visible staging buffer, waits for GPU completion,
    /// and strips any row padding required by wgpu (256-byte row alignment).
    /// If the texture format is Bgra8Unorm, pixels are swizzled to RGBA8 so the returned bytes
    /// are consistently RGBA.
    pub fn read_pixels(&self) -> anyhow::Result<Vec<u8>> {
        let resources = self
            .resources
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU resources not available"))?;
        let (texture, format) = match &resources.target {
            WgpuRenderTarget::Offscreen {
                texture, format, ..
            } => (texture, *format),
            WgpuRenderTarget::Surface(_) => {
                anyhow::bail!("read_pixels is only supported on offscreen renderers")
            }
        };

        let width = self.surface_config.width;
        let height = self.surface_config.height;
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let output_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_readback_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("readback_encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        resources.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        resources.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        match receiver.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(anyhow::anyhow!("Failed to map readback buffer: {e:?}")),
            Err(e) => Err(anyhow::anyhow!("Readback channel dropped: {e:?}")),
        }?;
        let mapped_data = buffer_slice.get_mapped_range();
        let is_bgra = format == wgpu::TextureFormat::Bgra8Unorm
            || format == wgpu::TextureFormat::Bgra8UnormSrgb;

        let mut result = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            let row_bytes = &mapped_data[start..end];
            if is_bgra {
                for chunk in row_bytes.chunks_exact(4) {
                    result.extend_from_slice(&[chunk[2], chunk[1], chunk[0], chunk[3]]);
                }
            } else {
                result.extend_from_slice(row_bytes);
            }
        }

        drop(mapped_data);
        output_buffer.unmap();

        Ok(result)
    }
}

/// A headless WGPU renderer implementing `PlatformHeadlessRenderer`. The web
/// has no surfaceless context, so the renderer exists on native targets only.
#[cfg(not(target_family = "wasm"))]
pub struct WgpuHeadlessRenderer {
    #[allow(dead_code)]
    context: WgpuContext,
    renderer: WgpuRenderer,
}

#[cfg(not(target_family = "wasm"))]
impl WgpuHeadlessRenderer {
    /// Creates a new headless WGPU renderer for offscreen rasterization.
    pub fn new(size: Size<DevicePixels>) -> anyhow::Result<Self> {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None)?;
        let renderer = WgpuRenderer::new_offscreen(&context, size)?;
        Ok(Self { context, renderer })
    }

    /// Reads back pixels from the renderer as RGBA8 bytes.
    pub fn read_pixels(&self) -> anyhow::Result<Vec<u8>> {
        self.renderer.read_pixels()
    }
}

#[cfg(not(target_family = "wasm"))]
impl gpui::PlatformHeadlessRenderer for WgpuHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<gpui::RgbaImage> {
        self.render_scene(scene, size)?;
        let bytes = self.renderer.read_pixels()?;
        gpui::RgbaImage::from_raw(size.width.0 as u32, size.height.0 as u32, bytes)
            .ok_or_else(|| anyhow::anyhow!("Failed to construct RgbaImage from rendered buffer"))
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.update_drawable_size(size);
        if !self.renderer.draw(scene) {
            anyhow::bail!("Offscreen WGPU render failed");
        }
        Ok(())
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

fn instance_range(range: Range<usize>) -> Range<u32> {
    range.start as u32..range.end as u32
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{MonochromeSprite, PolychromeSprite, Quad, Shadow, SubpixelSprite, Underline};

    #[test]
    fn webgl_shader_is_valid_wgsl_without_storage_buffers() {
        assert!(!WEBGL_SHADERS.contains("var<storage"));
        validate_wgsl(WEBGL_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn storage_buffer_shader_is_valid_wgsl() {
        validate_wgsl(STORAGE_BUFFER_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn subpixel_shader_is_valid_wgsl() {
        validate_wgsl(
            SUBPIXEL_SHADERS,
            naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        );
    }

    fn validate_wgsl(source: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .expect("shader should validate");
    }

    #[test]
    fn webgl_record_sizes_match_shader_word_strides() {
        assert_eq!(std::mem::size_of::<Quad>(), 50 * 4);
        assert_eq!(std::mem::size_of::<Shadow>(), 38 * 4);
        assert_eq!(std::mem::size_of::<PathRasterizationVertex>(), 40 * 4);
        assert_eq!(std::mem::size_of::<PathSprite>(), 4 * 4);
        assert_eq!(std::mem::size_of::<Underline>(), 26 * 4);
        assert_eq!(std::mem::size_of::<MonochromeSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<SubpixelSprite>(), 32 * 4);
        assert_eq!(std::mem::size_of::<PolychromeSprite>(), 34 * 4);
        assert_eq!(std::mem::size_of::<EdgeFadeMask>(), 12 * 4);
    }

    /// Every record a storage buffer holds has the layout of its Rust type
    /// in the shaders: the same members in the same order at the same byte
    /// offsets, and a span equal to the size of the Rust type. Catches a
    /// field added, removed, or moved on one side only, which makes the
    /// shader read a member from bytes of another. A storage buffer of a
    /// record type not listed here fails the test until it is listed.
    #[test]
    fn storage_records_have_the_layout_of_their_rust_types() {
        use gpui::BackdropBlur;
        use std::collections::BTreeSet;
        use std::mem::{offset_of, size_of};

        let module = naga::front::wgsl::parse_str(SUBPIXEL_SHADERS).expect("shader should parse");
        let struct_name = |ty: naga::Handle<naga::Type>| module.types[ty].name.clone();
        let stored: BTreeSet<String> = module
            .global_variables
            .iter()
            .filter(|(_, var)| matches!(var.space, naga::AddressSpace::Storage { .. }))
            .filter_map(|(_, var)| match module.types[var.ty].inner {
                naga::TypeInner::Array { base, .. } => struct_name(base),
                _ => None,
            })
            .collect();
        let layout = |name: &str| -> (Vec<(String, u32)>, u32) {
            module
                .types
                .iter()
                .find_map(|(_, ty)| match &ty.inner {
                    naga::TypeInner::Struct { members, span }
                        if ty.name.as_deref() == Some(name) =>
                    {
                        let members = members
                            .iter()
                            .map(|member| (member.name.clone().unwrap_or_default(), member.offset))
                            .collect();
                        Some((members, *span))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("the shaders declare no struct {name}"))
        };

        let mut checked = BTreeSet::new();
        macro_rules! assert_layout {
            ($ty:ident { $($field:ident),* $(,)? }) => {{
                let name = stringify!($ty);
                let rust: Vec<(String, u32)> =
                    vec![$((stringify!($field).to_string(), offset_of!($ty, $field) as u32)),*];
                let (shader, span) = layout(name);
                assert_eq!(shader, rust, "members and byte offsets of {name}");
                assert_eq!(span as usize, size_of::<$ty>(), "size of {name}");
                checked.insert(name.to_string());
            }};
        }
        assert_layout!(Quad {
            order,
            border_style,
            bounds,
            content_mask,
            background,
            border_color,
            corner_radii,
            border_widths,
            transformation,
        });
        assert_layout!(Shadow {
            order,
            blur_radius,
            bounds,
            corner_radii,
            content_mask,
            color,
            element_bounds,
            element_corner_radii,
            inset,
            pad,
            transformation,
        });
        assert_layout!(PathRasterizationVertex {
            xy_position,
            st_position,
            color,
            content_mask,
            transformation,
            bounds,
        });
        assert_layout!(PathSprite { bounds });
        assert_layout!(Underline {
            order,
            pad,
            bounds,
            content_mask,
            color,
            thickness,
            wavy,
            transformation,
        });
        assert_layout!(MonochromeSprite {
            order,
            pad,
            bounds,
            content_mask,
            color,
            tile,
            transformation,
        });
        assert_layout!(SubpixelSprite {
            order,
            pad,
            bounds,
            content_mask,
            color,
            tile,
            transformation,
        });
        assert_layout!(PolychromeSprite {
            order,
            pad,
            grayscale,
            opacity,
            bounds,
            content_mask,
            corner_radii,
            tile,
            transformation,
        });
        assert_layout!(BackdropBlur {
            order,
            pad,
            bounds,
            content_mask,
            corner_radii,
            blur_radius,
            saturation,
            tint,
            transformation,
        });
        assert_layout!(EdgeFadeMask {
            bounds,
            fade_bounds,
            bands,
        });
        assert_eq!(
            checked, stored,
            "the records the storage buffers hold and the records checked here"
        );
    }

    fn build_test_quad_scene(x: f32, width: f32, height: f32) -> Scene {
        build_test_quad_scene_with_hue(x, width, height, 0.0)
    }

    fn build_test_quad_scene_with_hue(x: f32, width: f32, height: f32, hue: f32) -> Scene {
        use gpui::{
            Bounds, ContentMask, Corners, Edges, Hsla, Point, ScaledPixels, Size, solid_background,
        };

        let mut scene = Scene::default();
        scene.insert_primitive(Quad {
            order: 0,
            border_style: Default::default(),
            bounds: Bounds {
                origin: Point {
                    x: ScaledPixels(x),
                    y: ScaledPixels(0.0),
                },
                size: Size {
                    width: ScaledPixels(width),
                    height: ScaledPixels(height),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(x + width + 100.0),
                        height: ScaledPixels(height + 100.0),
                    },
                },
                corner_radii: Default::default(),
            },
            background: solid_background(Hsla {
                h: hue,
                s: 1.0,
                l: 0.5,
                a: 1.0,
            }),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transformation: TransformationMatrix::unit(),
        });
        scene
    }

    fn pixel(bytes: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * width + x) * 4) as usize;
        [
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]
    }

    fn set_test_target_usage(
        renderer: &mut WgpuRenderer,
        usage: wgpu::TextureUsages,
        configure_pipeline: bool,
    ) {
        renderer.surface_config.usage = usage;
        let config = renderer.surface_config.clone();
        let path_sample_count = renderer.rendering_params.path_sample_count;
        let resources = renderer.resources_mut();
        let texture = resources.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("restricted_test_target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        resources.target = WgpuRenderTarget::Offscreen {
            texture,
            view,
            format: config.format,
        };
        if configure_pipeline {
            resources.select_pipelines(&config, path_sample_count);
            resources.invalidate_intermediate_textures();
        }
    }

    fn blurred_test_frame(size: f32, hue: f32) -> Scene {
        let mut scene = build_test_quad_scene_with_hue(0.0, size, size, hue);
        let bounds = gpui::Bounds::new(
            gpui::Point::default(),
            gpui::size(ScaledPixels(size), ScaledPixels(size)),
        );
        scene.insert_primitive(gpui::BackdropBlur {
            order: 1,
            pad: 0,
            bounds,
            content_mask: gpui::ContentMask {
                bounds,
                corner_radii: Default::default(),
            },
            corner_radii: Default::default(),
            blur_radius: ScaledPixels(2.0),
            saturation: 1.0,
            tint: Default::default(),
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();
        scene
    }

    /// WHY: swapchain copy flags are optional. Blur must never copy from a
    /// render-only target, and presentation must still replace stale pixels.
    /// Exercises every COPY_SRC/COPY_DST combination, partial and empty damage,
    /// and resize. Checks target pixels where readback is permitted; render-only
    /// targets are checked for GPU validation failures and covered by native capture.
    #[test]
    fn backdrop_and_presentation_respect_target_copy_capabilities() {
        let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
            .expect("surfaceless context");
        let copy_flags = [wgpu::TextureUsages::COPY_SRC, wgpu::TextureUsages::COPY_DST];
        for combination in 0..(1 << copy_flags.len()) {
            let usage = copy_flags.iter().enumerate().fold(
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                |usage, (index, flag)| {
                    if combination & (1 << index) != 0 {
                        usage | *flag
                    } else {
                        usage
                    }
                },
            );
            let mut renderer = WgpuRenderer::new_offscreen(
                &context,
                gpui::size(gpui::DevicePixels(32), gpui::DevicePixels(32)),
            )
            .expect("renderer");
            set_test_target_usage(&mut renderer, usage, true);
            let mut previous_pixels = None;
            for step in 0..4 {
                let mut scene = match step {
                    0 => blurred_test_frame(32.0, 0.0),
                    1 => blurred_test_frame(32.0, 2.0 / 3.0),
                    2 => Scene::default(),
                    _ => {
                        renderer.update_drawable_size(gpui::size(
                            gpui::DevicePixels(16),
                            gpui::DevicePixels(16),
                        ));
                        set_test_target_usage(&mut renderer, usage, false);
                        blurred_test_frame(16.0, 1.0 / 3.0)
                    }
                };
                if step == 1 {
                    scene.damage = damage(8.0, 8.0, 16.0, 16.0);
                }
                if step == 2 {
                    scene.damage = damage(0.0, 0.0, 0.0, 0.0);
                }
                if step == 3 {
                    scene.damage = damage(6.0, 6.0, 4.0, 4.0);
                }
                // Acquired swapchain contents are undefined, not the previous frame.
                let resources = renderer.resources();
                let WgpuRenderTarget::Offscreen { view, .. } = &resources.target else {
                    panic!("test requires an offscreen target");
                };
                let mut encoder =
                    resources
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("discard_acquired_target"),
                        });
                drop(WgpuRenderer::begin_frame_pass(
                    &mut encoder,
                    view,
                    "discard_acquired_target",
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    None,
                ));
                resources.queue.submit(std::iter::once(encoder.finish()));
                assert!(
                    renderer.draw(&scene),
                    "{usage:?}, step {step}: submission failed"
                );
                renderer
                    .resources()
                    .device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(std::time::Duration::from_secs(10)),
                    })
                    .expect("bounded GPU completion");
                assert_eq!(
                    renderer.last_error.lock().take(),
                    None,
                    "{usage:?}, step {step}",
                );
                if usage.contains(wgpu::TextureUsages::COPY_SRC) {
                    let pixels = renderer.read_pixels().expect("target readback");
                    match step {
                        0 => assert_eq!(pixel(&pixels, 32, 16, 16), [255, 0, 0, 255]),
                        1 => {
                            assert_eq!(pixel(&pixels, 32, 16, 16), [0, 0, 255, 255]);
                            assert_eq!(pixel(&pixels, 32, 2, 2), [255, 0, 0, 255]);
                        }
                        2 => assert_eq!(Some(&pixels), previous_pixels.as_ref()),
                        _ => {
                            assert_eq!(pixel(&pixels, 16, 8, 8), [0, 255, 0, 255]);
                            assert_eq!(pixel(&pixels, 16, 2, 2), [0, 255, 0, 255]);
                        }
                    }
                    previous_pixels = Some(pixels);
                }
            }
        }
    }

    fn damage(x: f32, y: f32, width: f32, height: f32) -> Option<gpui::Bounds<gpui::ScaledPixels>> {
        Some(gpui::Bounds {
            origin: gpui::Point {
                x: gpui::ScaledPixels(x),
                y: gpui::ScaledPixels(y),
            },
            size: gpui::Size {
                width: gpui::ScaledPixels(width),
                height: gpui::ScaledPixels(height),
            },
        })
    }

    /// WHY: a frame that declares damage must replace every pixel inside the
    /// rect and keep every pixel outside it, through the retained texture, the
    /// scissored clear, and the copy to the target. Covers a whole frame, a
    /// partial frame whose scene no longer contains what lies outside the
    /// rect, an empty rect, a rect with fractional edges, and a resize, which
    /// invalidates the retained frame. Does not cover a surface that lacks
    /// `COPY_DST`, where every frame is whole.
    #[test]
    fn a_partial_frame_repaints_only_its_damage_and_keeps_the_rest() {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
        let width = 200;
        let size = gpui::size(gpui::DevicePixels(width as i32), gpui::DevicePixels(100));
        let mut renderer = WgpuRenderer::new_offscreen(&context, size).expect("renderer");
        let clear = [0, 0, 0, 0];
        let is_red = |pixel: [u8; 4]| pixel[0] > 200 && pixel[2] < 50 && pixel[3] == 255;
        let is_blue = |pixel: [u8; 4]| pixel[2] > 200 && pixel[0] < 50 && pixel[3] == 255;

        // A red quad over the left half, drawn whole.
        let mut red = build_test_quad_scene(0.0, 100.0, 100.0);
        red.damage = damage(0.0, 0.0, 10.0, 10.0);
        assert!(renderer.draw(&red));
        assert_eq!(
            renderer.last_frame_extent,
            FrameExtent::Whole,
            "the first frame is whole even when the scene declares damage"
        );
        let bytes = renderer.read_pixels().expect("read pixels");
        assert!(is_red(pixel(&bytes, width, 50, 50)));
        assert_eq!(pixel(&bytes, width, 150, 50), clear);

        // A scene holding only a blue quad on the right half, with damage
        // limited to that half: the red quad is gone from the scene but its
        // pixels stay.
        let mut blue = build_test_quad_scene_with_hue(100.0, 100.0, 100.0, 240.0 / 360.0);
        blue.damage = damage(100.0, 0.0, 100.0, 100.0);
        assert!(renderer.draw(&blue));
        assert_eq!(
            renderer.last_frame_extent,
            FrameExtent::Partial(Scissor {
                x: 100,
                y: 0,
                width: 100,
                height: 100
            })
        );
        let bytes = renderer.read_pixels().expect("read pixels");
        assert!(is_red(pixel(&bytes, width, 50, 50)), "outside the rect");
        assert!(is_red(pixel(&bytes, width, 99, 50)), "last column outside");
        assert!(
            is_blue(pixel(&bytes, width, 100, 50)),
            "first column inside"
        );
        assert!(is_blue(pixel(&bytes, width, 150, 50)), "inside the rect");

        // An empty rect draws nothing and changes nothing.
        let mut empty = Scene::default();
        empty.damage = damage(300.0, 0.0, 10.0, 10.0);
        assert!(renderer.draw(&empty));
        assert_eq!(renderer.last_frame_extent, FrameExtent::Nothing);
        let bytes = renderer.read_pixels().expect("read pixels");
        assert!(is_red(pixel(&bytes, width, 50, 50)));
        assert!(is_blue(pixel(&bytes, width, 150, 50)));

        // A fractional rect snaps outward: the partially covered columns are
        // repainted, their neighbours are kept.
        let mut nothing_inside = Scene::default();
        nothing_inside.damage = damage(10.5, 0.0, 20.2, 100.0);
        assert!(renderer.draw(&nothing_inside));
        assert_eq!(
            renderer.last_frame_extent,
            FrameExtent::Partial(Scissor {
                x: 10,
                y: 0,
                width: 21,
                height: 100
            })
        );
        let bytes = renderer.read_pixels().expect("read pixels");
        assert!(is_red(pixel(&bytes, width, 9, 50)));
        assert_eq!(pixel(&bytes, width, 10, 50), clear);
        assert_eq!(pixel(&bytes, width, 30, 50), clear);
        assert!(is_red(pixel(&bytes, width, 31, 50)));
        assert!(is_blue(pixel(&bytes, width, 150, 50)));

        // A resize discards the retained frame: the next frame is whole.
        renderer.update_drawable_size(gpui::size(
            gpui::DevicePixels(width as i32),
            gpui::DevicePixels(120),
        ));
        let mut after_resize = Scene::default();
        after_resize.damage = damage(0.0, 0.0, 1.0, 1.0);
        assert!(renderer.draw(&after_resize));
        assert_eq!(renderer.last_frame_extent, FrameExtent::Whole);
        let bytes = renderer.read_pixels().expect("read pixels");
        assert_eq!(bytes.len(), (width * 120 * 4) as usize);
        assert_eq!(pixel(&bytes, width, 50, 50), clear);
        assert_eq!(pixel(&bytes, width, 150, 50), clear);
    }

    #[test]
    fn a_scissor_snaps_outward_and_clips_to_the_target() {
        assert_eq!(
            Scissor::from_damage(damage(10.5, 0.2, 20.2, 99.7).unwrap(), 200, 100),
            Some(Scissor {
                x: 10,
                y: 0,
                width: 21,
                height: 100
            })
        );
        assert_eq!(
            Scissor::from_damage(damage(-5.0, -5.0, 10.0, 10.0).unwrap(), 200, 100),
            Some(Scissor {
                x: 0,
                y: 0,
                width: 5,
                height: 5
            })
        );
        assert_eq!(
            Scissor::from_damage(damage(190.0, 90.0, 50.0, 50.0).unwrap(), 200, 100),
            Some(Scissor {
                x: 190,
                y: 90,
                width: 10,
                height: 10
            })
        );
        assert_eq!(
            Scissor::from_damage(damage(200.0, 0.0, 10.0, 10.0).unwrap(), 200, 100),
            None
        );
        assert_eq!(
            Scissor::from_damage(damage(0.0, 0.0, 0.0, 0.0).unwrap(), 200, 100),
            None
        );
    }

    #[test]
    fn test_scene_renders_expected_geometry_at_scale_factors() {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");

        // Scale factor 1.0: 120x80 logical = 120x80 device
        let size_1x = gpui::size(gpui::DevicePixels(120), gpui::DevicePixels(80));
        let mut renderer_1x = WgpuRenderer::new_offscreen(&context, size_1x).expect("renderer 1x");
        let scene_1x = build_test_quad_scene(10.0, 50.0, 50.0);
        assert!(renderer_1x.draw(&scene_1x));
        let bytes_1x = renderer_1x.read_pixels().expect("read pixels 1x");
        assert_eq!(bytes_1x.len(), 120 * 80 * 4);

        // Scale factor 2.0: 120x80 logical @ 2.0 = 240x160 device
        let size_2x = gpui::size(gpui::DevicePixels(240), gpui::DevicePixels(160));
        let mut renderer_2x = WgpuRenderer::new_offscreen(&context, size_2x).expect("renderer 2x");
        let scene_2x = build_test_quad_scene(20.0, 100.0, 100.0);
        assert!(renderer_2x.draw(&scene_2x));
        let bytes_2x = renderer_2x.read_pixels().expect("read pixels 2x");
        assert_eq!(bytes_2x.len(), 240 * 160 * 4);
    }

    #[test]
    fn test_row_unpadding_at_non_multiple_of_64() {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");

        // Width 300 is not a multiple of 64. 300 * 4 = 1200 bytes per row, padded to 1280.
        let width = 300;
        let height = 100;
        let size = gpui::size(gpui::DevicePixels(width), gpui::DevicePixels(height));
        let mut renderer = WgpuRenderer::new_offscreen(&context, size).expect("offscreen renderer");

        // Render a solid red quad from x = 50 to x = 100, across all y = 0..100
        let scene = build_test_quad_scene(50.0, 50.0, 100.0);
        assert!(renderer.draw(&scene));
        let bytes = renderer.read_pixels().expect("read pixels");
        assert_eq!(bytes.len(), (width * height * 4) as usize);

        // Verify the vertical edge: column 49 is clear, column 50 is red, column 99 is red, column 100 is clear.
        // If rows were sheared due to stride padding mismatch, these column positions would drift per row.
        for y in 0..height {
            let row_offset = (y * width * 4) as usize;

            // Column 49: transparent black (0, 0, 0, 0)
            let col_49 = row_offset + 49 * 4;
            assert_eq!(
                &bytes[col_49..col_49 + 4],
                &[0, 0, 0, 0],
                "column 49 should be transparent on row {y}"
            );

            // Column 50: red pixel (255, 0, 0, 255)
            let col_50 = row_offset + 50 * 4;
            assert_eq!(
                bytes[col_50 + 3],
                255,
                "column 50 alpha should be 255 on row {y}"
            );
            assert!(
                bytes[col_50] > 200,
                "column 50 red should be > 200 on row {y}"
            );

            // Column 99: red pixel (255, 0, 0, 255)
            let col_99 = row_offset + 99 * 4;
            assert_eq!(
                bytes[col_99 + 3],
                255,
                "column 99 alpha should be 255 on row {y}"
            );

            // Column 100: transparent black (0, 0, 0, 0)
            let col_100 = row_offset + 100 * 4;
            assert_eq!(
                &bytes[col_100..col_100 + 4],
                &[0, 0, 0, 0],
                "column 100 should be transparent on row {y}"
            );
        }
    }

    #[test]
    fn test_determinism_in_one_process() {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
        let size = gpui::size(gpui::DevicePixels(200), gpui::DevicePixels(100));

        let mut renderer1 = WgpuRenderer::new_offscreen(&context, size).expect("renderer 1");
        let scene = build_test_quad_scene(30.0, 80.0, 60.0);
        assert!(renderer1.draw(&scene));
        let bytes1 = renderer1.read_pixels().expect("read pixels 1");

        let mut renderer2 = WgpuRenderer::new_offscreen(&context, size).expect("renderer 2");
        assert!(renderer2.draw(&scene));
        let bytes2 = renderer2.read_pixels().expect("read pixels 2");

        assert_eq!(
            bytes1, bytes2,
            "offscreen render must be deterministic within one process"
        );
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn test_determinism_across_processes() {
        if std::env::var("P10_CHILD_DETERMINISM").is_ok() {
            let instance = WgpuContext::surfaceless_instance();
            let context =
                WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
            let size = gpui::size(gpui::DevicePixels(200), gpui::DevicePixels(100));
            let mut renderer = WgpuRenderer::new_offscreen(&context, size).expect("renderer");
            let scene = build_test_quad_scene(30.0, 80.0, 60.0);
            assert!(renderer.draw(&scene));
            let bytes = renderer.read_pixels().expect("read pixels");
            use std::io::Write as _;
            let mut stdout = std::io::stdout();
            stdout.write_all(b"P10_BYTES:").unwrap();
            for b in &bytes {
                write!(stdout, "{:02x}", b).unwrap();
            }
            writeln!(stdout).unwrap();
            return;
        }

        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
        let size = gpui::size(gpui::DevicePixels(200), gpui::DevicePixels(100));
        let mut renderer = WgpuRenderer::new_offscreen(&context, size).expect("renderer");
        let scene = build_test_quad_scene(30.0, 80.0, 60.0);
        assert!(renderer.draw(&scene));
        let parent_bytes = renderer.read_pixels().expect("read pixels");

        // Spawn second process
        if let Ok(current_exe) = std::env::current_exe() {
            let output = std::process::Command::new(current_exe)
                .arg("wgpu_renderer::tests::test_determinism_across_processes")
                .arg("--exact")
                .arg("--nocapture")
                .env("P10_CHILD_DETERMINISM", "1")
                .output();

            if let Ok(output) = output {
                let stdout_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout_str.lines().find(|l| l.starts_with("P10_BYTES:")) {
                    let hex_str = line.trim_start_matches("P10_BYTES:");
                    let child_bytes: Vec<u8> = (0..hex_str.len())
                        .step_by(2)
                        .filter_map(|i| u8::from_str_radix(&hex_str[i..i + 2], 16).ok())
                        .collect();
                    assert_eq!(
                        parent_bytes, child_bytes,
                        "render output must match exactly across processes"
                    );
                }
            }
        }
    }

    #[test]
    fn test_text_is_really_rasterized() {
        use crate::cosmic_text_system::CosmicTextSystem;
        use gpui::{
            AppContext, Context, HeadlessAppContext, IntoElement, ParentElement,
            PlatformHeadlessRenderer, Render, Styled, Window, div, px, size,
        };
        use std::sync::Arc;

        struct TextView;
        impl Render for TextView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div()
                    .text_color(gpui::rgb(0xffffff))
                    .child("Hello, Headless Rasterizer!")
            }
        }

        struct EmptyView;
        impl Render for EmptyView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div()
            }
        }

        let text_system = Arc::new(CosmicTextSystem::new("sans-serif"));
        let mut cx = HeadlessAppContext::with_platform(text_system, Arc::new(()), || {
            WgpuHeadlessRenderer::new(size(gpui::DevicePixels(200), gpui::DevicePixels(100)))
                .ok()
                .map(|r| Box::new(r) as Box<dyn PlatformHeadlessRenderer>)
        });

        let text_frame = cx
            .render_frame(size(px(200.0), px(100.0)), 1.0, |_window, cx| {
                cx.new(|_| TextView)
            })
            .expect("render text frame");

        let empty_frame = cx
            .render_frame(size(px(200.0), px(100.0)), 1.0, |_window, cx| {
                cx.new(|_| EmptyView)
            })
            .expect("render empty frame");

        assert_ne!(
            text_frame.as_bytes(),
            empty_frame.as_bytes(),
            "frame with text must differ from frame without text"
        );

        // Verify text frame is not a uniform color (contains non-zero alpha / color text pixels)
        let text_bytes = text_frame.as_bytes();
        let has_text_pixels = text_bytes.chunks_exact(4).any(|p| p[3] > 0 || p[0] > 0);
        assert!(
            has_text_pixels,
            "text frame must contain rasterized glyph pixels"
        );
    }

    #[test]
    fn test_cleared_frame_is_exactly_clear_color() {
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
        let size = gpui::size(gpui::DevicePixels(100), gpui::DevicePixels(100));
        let mut renderer = WgpuRenderer::new_offscreen(&context, size).expect("renderer");

        let empty_scene = Scene::default();
        assert!(renderer.draw(&empty_scene));
        let bytes = renderer.read_pixels().expect("read pixels");

        assert_eq!(bytes.len(), 100 * 100 * 4);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "cleared frame must have all zero bytes (transparent black), proving no garbage initialization"
        );
    }

    #[test]
    // P1 golden test for affine transforms
    fn test_headless_affine_transforms_all_primitives_at_scale_factors() {
        use crate::cosmic_text_system::CosmicTextSystem;
        use gpui::{
            AppContext, Context, HeadlessAppContext, IntoElement, ParentElement,
            PlatformHeadlessRenderer, Render, Styled, Window, div, px, rgb, size,
        };
        struct QuadView(bool);
        impl Render for QuadView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let quad = if self.0 {
                    div()
                        .absolute()
                        .top(px(10.0))
                        .left(px(10.0))
                        .w(px(40.0))
                        .h(px(40.0))
                        .bg(rgb(0xff0000))
                        .translate_x(px(50.0))
                } else {
                    div()
                        .absolute()
                        .top(px(10.0))
                        .left(px(10.0))
                        .w(px(40.0))
                        .h(px(40.0))
                        .bg(rgb(0xff0000))
                };
                div().size_full().child(quad)
            }
        }

        struct ShadowView(bool);
        impl Render for ShadowView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let shadow = if self.0 {
                    div()
                        .absolute()
                        .top(px(20.0))
                        .left(px(20.0))
                        .w(px(40.0))
                        .h(px(40.0))
                        .shadow_lg()
                        .translate_x(px(50.0))
                } else {
                    div()
                        .absolute()
                        .top(px(20.0))
                        .left(px(20.0))
                        .w(px(40.0))
                        .h(px(40.0))
                        .shadow_lg()
                };
                div().size_full().child(shadow)
            }
        }

        struct TextView(bool);
        impl Render for TextView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let text = if self.0 {
                    div()
                        .absolute()
                        .top(px(10.0))
                        .left(px(10.0))
                        .translate_y(px(30.0))
                        .text_color(rgb(0xffffff))
                        .child("Affine Transform")
                } else {
                    div()
                        .absolute()
                        .top(px(10.0))
                        .left(px(10.0))
                        .text_color(rgb(0xffffff))
                        .child("Affine Transform")
                };
                div().size_full().child(text)
            }
        }

        struct UnderlineView(bool);
        impl Render for UnderlineView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                let underline = if self.0 {
                    div()
                        .absolute()
                        .top(px(20.0))
                        .left(px(10.0))
                        .w(px(80.0))
                        .h(px(20.0))
                        .translate_y(px(30.0))
                        .text_color(rgb(0xffffff))
                        .underline()
                        .child("Underline")
                } else {
                    div()
                        .absolute()
                        .top(px(20.0))
                        .left(px(10.0))
                        .w(px(80.0))
                        .h(px(20.0))
                        .text_color(rgb(0xffffff))
                        .underline()
                        .child("Underline")
                };
                div().size_full().child(underline)
            }
        }

        for scale in [1.0, 2.0] {
            let text_system = Arc::new(CosmicTextSystem::new("sans-serif"));
            let width = (160.0 * scale) as i32;
            let height = (100.0 * scale) as i32;
            let mut cx = HeadlessAppContext::with_platform(text_system, Arc::new(()), move || {
                WgpuHeadlessRenderer::new(size(
                    gpui::DevicePixels(width),
                    gpui::DevicePixels(height),
                ))
                .ok()
                .map(|r| Box::new(r) as Box<dyn PlatformHeadlessRenderer>)
            });

            // 1. Quad affine transform validation
            let frame_quad_untrans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| QuadView(false))
                })
                .expect("render quad untransformed");
            let frame_quad_trans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| QuadView(true))
                })
                .expect("render quad transformed");

            assert_ne!(
                frame_quad_untrans.as_bytes(),
                frame_quad_trans.as_bytes(),
                "transformed quad must differ from untransformed quad at scale {scale}"
            );
            let q_untrans_bytes = frame_quad_untrans.as_bytes();
            let q_trans_bytes = frame_quad_trans.as_bytes();
            let stride = (160.0 * scale) as usize * 4;
            let px_x = 60;
            let px_y = 60;
            let idx_untrans = px_y * stride + px_x * 4;
            assert!(
                q_untrans_bytes[idx_untrans] > 200 && q_untrans_bytes[idx_untrans + 3] == 255,
                "untransformed quad should be solid red at device ({px_x}, {px_y}) for scale {scale}"
            );
            assert_eq!(
                &q_trans_bytes[idx_untrans..idx_untrans + 4],
                &[0, 0, 0, 0],
                "transformed quad must be clear at original device position ({px_x}, {px_y}) for scale {scale}"
            );

            let trans_x = 140;
            let idx_trans = px_y * stride + trans_x * 4;
            assert!(
                q_trans_bytes[idx_trans] > 200 && q_trans_bytes[idx_trans + 3] == 255,
                "transformed quad must be solid red at new device position ({trans_x}, {px_y}) for scale {scale}"
            );
            assert_eq!(
                &q_untrans_bytes[idx_trans..idx_trans + 4],
                &[0, 0, 0, 0],
                "untransformed quad must be clear at translated device position ({trans_x}, {px_y}) for scale {scale}"
            );
            // 2. Shadow affine transform validation
            let frame_shadow_untrans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| ShadowView(false))
                })
                .expect("render shadow untransformed");
            let frame_shadow_trans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| ShadowView(true))
                })
                .expect("render shadow transformed");

            assert_ne!(
                frame_shadow_untrans.as_bytes(),
                frame_shadow_trans.as_bytes(),
                "transformed shadow must differ from untransformed shadow at scale {scale}"
            );

            // 3. Text (glyph run) affine transform validation
            let frame_text_untrans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| TextView(false))
                })
                .expect("render text untransformed");
            let frame_text_trans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| TextView(true))
                })
                .expect("render text transformed");

            assert_ne!(
                frame_text_untrans.as_bytes(),
                frame_text_trans.as_bytes(),
                "transformed text must differ from untransformed text at scale {scale}"
            );

            // 4. Underline (path) affine transform validation
            let frame_underline_untrans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| UnderlineView(false))
                })
                .expect("render underline untransformed");
            let frame_underline_trans = cx
                .render_frame(size(px(160.0), px(100.0)), scale, |_window, cx| {
                    cx.new(|_| UnderlineView(true))
                })
                .expect("render underline transformed");

            assert_ne!(
                frame_underline_untrans.as_bytes(),
                frame_underline_trans.as_bytes(),
                "transformed underline must differ from untransformed underline at scale {scale}"
            );
        }

        // 5. Scene primitive affine transform validation for Path and Sprite
        let instance = WgpuContext::surfaceless_instance();
        let context = WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
        let dev_size = size(gpui::DevicePixels(160), gpui::DevicePixels(100));
        let mut renderer = WgpuRenderer::new_offscreen(&context, dev_size).expect("renderer");

        let mut scene_untrans = Scene::default();
        let mut path_untrans = gpui::Path::new(gpui::point(px(10.0), px(10.0)));
        path_untrans.line_to(gpui::point(px(50.0), px(10.0)));
        path_untrans.line_to(gpui::point(px(50.0), px(50.0)));
        path_untrans.line_to(gpui::point(px(10.0), px(50.0)));
        path_untrans.color = gpui::solid_background(rgb(0x00ff00));
        path_untrans.bounds = gpui::Bounds::new(
            gpui::point(px(10.0), px(10.0)),
            gpui::size(px(40.0), px(40.0)),
        );
        path_untrans.content_mask = gpui::ContentMask {
            bounds: gpui::Bounds::new(
                gpui::point(px(0.0), px(0.0)),
                gpui::size(px(160.0), px(100.0)),
            ),
            corner_radii: Default::default(),
        };
        scene_untrans.insert_primitive(path_untrans.scale(1.0));

        let mut scene_trans = Scene::default();
        let mut path_trans = path_untrans.scale(1.0);
        path_trans.transformation = gpui::TransformationMatrix::unit().translate(gpui::point(
            gpui::ScaledPixels(50.0),
            gpui::ScaledPixels(0.0),
        ));
        scene_trans.insert_primitive(path_trans);
        assert!(renderer.draw(&scene_untrans));
        let path_untrans_bytes = renderer.read_pixels().expect("read path untrans");
        assert!(renderer.draw(&scene_trans));
        let path_trans_bytes = renderer.read_pixels().expect("read path trans");

        assert_ne!(
            path_untrans_bytes, path_trans_bytes,
            "transformed path must differ from untransformed path"
        );
        assert!(
            path_trans_bytes.chunks_exact(4).any(|p| p[1] > 200),
            "transformed path frame must contain rasterized green path pixels at new position"
        );

        // 6. Direct Underline primitive affine transform validation
        let mut underline_scene_untrans = Scene::default();
        underline_scene_untrans.insert_primitive(gpui::Underline {
            order: 0,
            pad: 0,
            bounds: gpui::Bounds::new(
                gpui::point(gpui::ScaledPixels(10.0), gpui::ScaledPixels(20.0)),
                gpui::size(gpui::ScaledPixels(80.0), gpui::ScaledPixels(4.0)),
            ),
            content_mask: gpui::ContentMask {
                bounds: gpui::Bounds::new(
                    gpui::point(gpui::ScaledPixels(0.0), gpui::ScaledPixels(0.0)),
                    gpui::size(gpui::ScaledPixels(160.0), gpui::ScaledPixels(100.0)),
                ),
                corner_radii: Default::default(),
            },
            color: gpui::rgb(0xffffff).into(),
            thickness: gpui::ScaledPixels(2.0),
            wavy: false.into(),
            transformation: gpui::TransformationMatrix::unit(),
        });

        let mut underline_scene_trans = Scene::default();
        underline_scene_trans.insert_primitive(gpui::Underline {
            order: 0,
            pad: 0,
            bounds: gpui::Bounds::new(
                gpui::point(gpui::ScaledPixels(10.0), gpui::ScaledPixels(20.0)),
                gpui::size(gpui::ScaledPixels(80.0), gpui::ScaledPixels(4.0)),
            ),
            content_mask: gpui::ContentMask {
                bounds: gpui::Bounds::new(
                    gpui::point(gpui::ScaledPixels(0.0), gpui::ScaledPixels(0.0)),
                    gpui::size(gpui::ScaledPixels(160.0), gpui::ScaledPixels(100.0)),
                ),
                corner_radii: Default::default(),
            },
            color: gpui::rgb(0xffffff).into(),
            thickness: gpui::ScaledPixels(2.0),
            wavy: false.into(),
            transformation: gpui::TransformationMatrix::unit().translate(gpui::point(
                gpui::ScaledPixels(50.0),
                gpui::ScaledPixels(0.0),
            )),
        });

        assert!(renderer.draw(&underline_scene_untrans));
        let direct_underline_untrans_bytes =
            renderer.read_pixels().expect("read underline untrans");
        assert!(renderer.draw(&underline_scene_trans));
        let direct_underline_trans_bytes = renderer.read_pixels().expect("read underline trans");

        assert_ne!(
            direct_underline_untrans_bytes, direct_underline_trans_bytes,
            "transformed underline primitive must differ from untransformed underline primitive"
        );
    }
    #[test]
    fn test_headless_frame_metadata_text_runs_and_hitboxes_at_scale_factors() {
        use crate::cosmic_text_system::CosmicTextSystem;
        use gpui::{
            AppContext, Context, HeadlessAppContext, InteractiveElement, IntoElement,
            ParentElement, PlatformHeadlessRenderer, Render, Styled, Window, div, px, rgb, size,
        };
        use std::collections::HashSet;
        struct MultiSizeTextView;
        impl Render for MultiSizeTextView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div()
                    .size_full()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(0xffffff))
                            .child("Heading 14px"),
                    )
                    .child(
                        div()
                            .text_size(px(24.0))
                            .text_color(rgb(0xffffff))
                            .child("Title 24px"),
                    )
            }
        }

        struct InteractiveAndInertView;
        impl Render for InteractiveAndInertView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div()
                    .size_full()
                    // 1. Inert element: just background quad, no id or mouse listeners
                    .child(div().w(px(50.0)).h(px(50.0)).bg(rgb(0xff0000)))
                    // 2. Interactive element: carries an id and mouse handler
                    .child(
                        div()
                            .id("clickable-button")
                            .w(px(60.0))
                            .h(px(30.0))
                            .bg(rgb(0x00ff00))
                            .on_mouse_down(gpui::MouseButton::Left, |_ev, _window, _cx| {}),
                    )
            }
        }

        struct EmptyView;
        impl Render for EmptyView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut Context<Self>,
            ) -> impl IntoElement {
                div().size_full()
            }
        }

        for scale in [1.0, 2.0] {
            let width = (200.0 * scale) as i32;
            let height = (150.0 * scale) as i32;
            let text_system = Arc::new(CosmicTextSystem::new("sans-serif"));
            let mut cx = HeadlessAppContext::with_platform(text_system, Arc::new(()), move || {
                WgpuHeadlessRenderer::new(size(
                    gpui::DevicePixels(width),
                    gpui::DevicePixels(height),
                ))
                .ok()
                .map(|r| Box::new(r) as Box<dyn PlatformHeadlessRenderer>)
            });

            // 1. Text metadata validation: exactly 2 distinct font sizes
            let text_frame = cx
                .render_frame(size(px(200.0), px(150.0)), scale, |_window, cx| {
                    cx.new(|_| MultiSizeTextView)
                })
                .expect("render text frame");

            let runs = text_frame.text_runs();
            assert!(
                !runs.is_empty(),
                "text runs must be populated at scale {scale}"
            );
            let distinct_sizes: HashSet<_> = runs.iter().map(|r| r.font_size).collect();
            assert_eq!(
                distinct_sizes.len(),
                2,
                "must report exactly 2 distinct text sizes, found: {distinct_sizes:?}"
            );
            assert!(distinct_sizes.contains(&px(14.0)));
            assert!(distinct_sizes.contains(&px(24.0)));

            // 2. Hitbox metadata validation: exactly 1 interactive hitbox
            let interactive_frame = cx
                .render_frame(size(px(200.0), px(150.0)), scale, |_window, cx| {
                    cx.new(|_| InteractiveAndInertView)
                })
                .expect("render interactive frame");

            let hitboxes = interactive_frame.hitboxes();
            assert_eq!(
                hitboxes.len(),
                1,
                "must report exactly 1 hitbox for 1 interactive and 1 inert element, found {hitboxes:?}"
            );
            assert!(
                hitboxes[0].size.width == px(60.0) && hitboxes[0].size.height == px(30.0),
                "hitbox bounds must match interactive element, got {:?}",
                hitboxes[0]
            );

            // 3. Empty frame validation: empty lists distinguishable from unpopulated
            let empty_frame = cx
                .render_frame(size(px(200.0), px(150.0)), scale, |_window, cx| {
                    cx.new(|_| EmptyView)
                })
                .expect("render empty frame");

            assert_eq!(empty_frame.text_runs().len(), 0);
            assert_eq!(empty_frame.hitboxes().len(), 0);
        }
    }
    #[test]
    fn test_per_corner_radii_on_single_quad_at_scale_factors() {
        use gpui::{Bounds, ContentMask, Corners, Edges, Hsla, Point, ScaledPixels, Scene, Size};

        for scale in [1.0, 2.0] {
            let width = (100.0 * scale) as i32;
            let height = (100.0 * scale) as i32;

            let instance = WgpuContext::surfaceless_instance();
            let context =
                WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
            let size_device = gpui::size(gpui::DevicePixels(width), gpui::DevicePixels(height));
            let mut renderer =
                WgpuRenderer::new_offscreen(&context, size_device).expect("renderer");

            let mut scene = Scene::default();
            scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                background: gpui::solid_background(Hsla {
                    h: 0.0,
                    s: 1.0,
                    l: 0.5,
                    a: 1.0,
                }),
                border_color: Hsla::default(),
                corner_radii: Corners {
                    top_left: ScaledPixels(20.0 * scale),
                    top_right: ScaledPixels(0.0),
                    bottom_right: ScaledPixels(30.0 * scale),
                    bottom_left: ScaledPixels(10.0 * scale),
                },
                border_widths: Edges::default(),
                transformation: TransformationMatrix::unit(),
            });

            assert!(renderer.draw(&scene));
            let bytes = renderer.read_pixels().expect("read pixels");
            let pitch = (100.0 * scale) as usize * 4;
            let pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]
            };
            // Corner 1: Top-Left (R = 20px)
            // (2, 2) is outside the 20px radius arc (distance to (20,20) = ~25.46 > 20) -> alpha 0
            let tl_outside = pixel_at(2.0, 2.0);
            assert!(
                tl_outside[3] < 30,
                "TL corner outside radius (2, 2) must be unpainted (alpha 0) at scale {scale}, got {tl_outside:?}"
            );
            // (16, 16) is inside the 20px radius arc (distance to (20,20) = ~5.66 < 20) -> alpha 255
            let tl_inside = pixel_at(16.0, 16.0);
            assert!(
                tl_inside[3] > 220 && tl_inside[0] > 220,
                "TL corner inside radius (16, 16) must be painted at scale {scale}, got {tl_inside:?}"
            );
            // (5, 25) is beyond the 20px corner on the straight edge -> alpha 255
            let tl_straight = pixel_at(5.0, 25.0);
            assert!(
                tl_straight[3] > 220 && tl_straight[0] > 220,
                "TL straight edge (5, 25) must be painted at scale {scale}, got {tl_straight:?}"
            );

            // Corner 2: Top-Right (R = 0px, sharp corner)
            // (98, 2) is in the geometric corner and must be fully painted -> alpha 255
            let tr_corner = pixel_at(98.0, 2.0);
            assert!(
                tr_corner[3] > 220 && tr_corner[0] > 220,
                "TR sharp corner (98, 2) must be painted at scale {scale}, got {tr_corner:?}"
            );

            // Corner 3: Bottom-Right (R = 30px)
            // (98, 98) is outside the 30px radius arc (distance to (70,70) = ~39.6 > 30) -> alpha 0
            let br_outside = pixel_at(98.0, 98.0);
            assert!(
                br_outside[3] < 30,
                "BR corner outside radius (98, 98) must be unpainted (alpha 0) at scale {scale}, got {br_outside:?}"
            );
            // (92, 92) is also outside the 30px radius arc (distance to (70,70) = ~31.1 > 30) -> alpha 0
            let br_outside2 = pixel_at(92.0, 92.0);
            assert!(
                br_outside2[3] < 30,
                "BR corner outside radius (92, 92) must be unpainted (alpha 0) at scale {scale}, got {br_outside2:?}"
            );
            // (75, 75) is inside the 30px radius arc (distance to (70,70) = ~7.07 < 30) -> alpha 255
            let br_inside = pixel_at(75.0, 75.0);
            assert!(
                br_inside[3] > 220 && br_inside[0] > 220,
                "BR corner inside radius (75, 75) must be painted at scale {scale}, got {br_inside:?}"
            );

            // Corner 4: Bottom-Left (R = 10px)
            // (2, 98) is outside the 10px radius arc (distance to (10,90) = ~11.31 > 10) -> alpha 0
            let bl_outside = pixel_at(2.0, 98.0);
            assert!(
                bl_outside[3] < 30,
                "BL corner outside radius (2, 98) must be unpainted (alpha 0) at scale {scale}, got {bl_outside:?}"
            );
            // (2, 85) is past the 10px radius vertically (y=85 < 90) -> alpha 255
            let bl_straight = pixel_at(2.0, 85.0);
            assert!(
                bl_straight[3] > 220 && bl_straight[0] > 220,
                "BL straight edge (2, 85) must be painted at scale {scale}, got {bl_straight:?}"
            );
            // (8, 92) is inside the 10px radius arc (distance to (10,90) = ~2.83 < 10) -> alpha 255
            let bl_inside = pixel_at(8.0, 92.0);
            assert!(
                bl_inside[3] > 220 && bl_inside[0] > 220,
                "BL corner inside radius (8, 92) must be painted at scale {scale}, got {bl_inside:?}"
            );
        }
    }
    #[test]
    fn test_inset_shadow_and_inset_hairline_on_rounded_rect_at_scale_factors() {
        use gpui::{Bounds, ContentMask, Corners, Hsla, Point, ScaledPixels, Scene, Shadow, Size};

        for scale in [1.0, 2.0] {
            let width = (100.0 * scale) as i32;
            let height = (100.0 * scale) as i32;

            let instance = WgpuContext::surfaceless_instance();
            let context =
                WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
            let size_device = gpui::size(gpui::DevicePixels(width), gpui::DevicePixels(height));
            let mut renderer =
                WgpuRenderer::new_offscreen(&context, size_device).expect("renderer");

            // 1. Inset Hairline (spread = 2px, blur = 0px)
            // Element: 100x100 at (0, 0), corner radius 20px
            // Hole: 96x96 at (2, 2), corner radius 18px
            let mut scene = Scene::default();
            scene.insert_primitive(Shadow {
                order: 0,
                blur_radius: ScaledPixels(0.0),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(2.0 * scale),
                        y: ScaledPixels(2.0 * scale),
                    },
                    size: Size {
                        width: ScaledPixels(96.0 * scale),
                        height: ScaledPixels(96.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                corner_radii: Corners::all(ScaledPixels(18.0 * scale)),
                color: Hsla {
                    h: 0.0,
                    s: 1.0,
                    l: 0.5,
                    a: 1.0,
                },
                element_bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                element_corner_radii: Corners::all(ScaledPixels(20.0 * scale)),
                inset: 1,
                pad: 0,
                transformation: TransformationMatrix::unit(),
            });

            assert!(renderer.draw(&scene));
            let bytes = renderer.read_pixels().expect("read pixels");
            let pitch = (100.0 * scale) as usize * 4;
            let pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]
            };

            // Geometric corner outside radius: (2, 2) has distance ~25.46 to (20, 20) > 20 -> alpha 0
            let outside = pixel_at(2.0, 2.0);
            assert!(
                outside[3] < 30,
                "geometric corner outside radius must be untouched (alpha 0) at scale {scale}, got {outside:?}"
            );

            // Pixel just inside the outer rounded corner edge on the 45-degree diagonal:
            // Arc center (20, 20), radius 19.0 -> (20 - 13.44, 20 - 13.44) ~ (6.56, 6.56)
            let corner_hairline = pixel_at(6.5, 6.5);
            assert!(
                corner_hairline[3] > 220 && corner_hairline[0] > 220,
                "pixel just inside rounded corner arc must be hairline color at scale {scale}, got {corner_hairline:?}"
            );

            // Pixel just inside the straight border edge: (1.0, 50.0) -> alpha 255
            let straight_hairline = pixel_at(1.0, 50.0);
            assert!(
                straight_hairline[3] > 220 && straight_hairline[0] > 220,
                "pixel just inside straight edge must be hairline color at scale {scale}, got {straight_hairline:?}"
            );

            // Pixel deep inside the interior hole: (50.0, 50.0) -> alpha 0
            let interior = pixel_at(50.0, 50.0);
            assert!(
                interior[3] < 30,
                "interior hole inside hairline must be untouched (alpha 0) at scale {scale}, got {interior:?}"
            );

            // 2. Blurred Inset Shadow (spread = 0px, blur = 8px)
            let mut blur_scene = Scene::default();
            blur_scene.insert_primitive(Shadow {
                order: 0,
                blur_radius: ScaledPixels(8.0 * scale),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                corner_radii: Corners::all(ScaledPixels(20.0 * scale)),
                color: Hsla {
                    h: 0.0,
                    s: 1.0,
                    l: 0.5,
                    a: 1.0,
                },
                element_bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                element_corner_radii: Corners::all(ScaledPixels(20.0 * scale)),
                inset: 1,
                pad: 0,
                transformation: TransformationMatrix::unit(),
            });

            assert!(renderer.draw(&blur_scene));
            let blur_bytes = renderer.read_pixels().expect("read blur pixels");
            let blur_pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    blur_bytes[offset],
                    blur_bytes[offset + 1],
                    blur_bytes[offset + 2],
                    blur_bytes[offset + 3],
                ]
            };

            // Blurred inset shadow: geometric corner outside radius remains clipped/untouched
            let blur_outside = blur_pixel_at(2.0, 2.0);
            assert!(
                blur_outside[3] < 30,
                "blurred inset shadow must stay clipped to element bounds at geometric corner, got {blur_outside:?}"
            );

            // Near the boundary inside the element: shadow is active
            let blur_edge = blur_pixel_at(3.0, 50.0);
            assert!(
                blur_edge[3] > 50,
                "blurred inset shadow must be non-zero near edge, got {blur_edge:?}"
            );

            // Deep interior: blurred shadow attenuates to zero
            let blur_interior = blur_pixel_at(50.0, 50.0);
            assert!(
                blur_interior[3] < 30,
                "blurred inset shadow must attenuate in deep interior, got {blur_interior:?}"
            );
        }
    }

    #[test]
    fn test_backdrop_blur_primitive_and_sampling_behind_element_at_scale_factors() {
        use gpui::{
            BackdropBlur, Bounds, ContentMask, Corners, Edges, Hsla, Point, ScaledPixels, Scene,
            Size,
        };

        for scale in [1.0, 2.0] {
            let width = (100.0 * scale) as i32;
            let height = (100.0 * scale) as i32;

            let instance = WgpuContext::surfaceless_instance();
            let context =
                WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
            let size_device = gpui::size(gpui::DevicePixels(width), gpui::DevicePixels(height));
            let mut renderer =
                WgpuRenderer::new_offscreen(&context, size_device).expect("renderer");

            // Build scene with high-frequency vertical stripes (2px wide) from x=0 to x=100
            let mut scene = Scene::default();
            for i in 0..50 {
                let x = (i * 2) as f32;
                let color = if i % 2 == 0 {
                    Hsla {
                        h: 0.0,
                        s: 0.0,
                        l: 1.0,
                        a: 1.0,
                    } // White
                } else {
                    Hsla {
                        h: 0.0,
                        s: 0.0,
                        l: 0.0,
                        a: 1.0,
                    } // Black
                };
                scene.insert_primitive(Quad {
                    order: 0,
                    border_style: Default::default(),
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(x * scale),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(2.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    content_mask: ContentMask {
                        bounds: Bounds {
                            origin: Point {
                                x: ScaledPixels(0.0),
                                y: ScaledPixels(0.0),
                            },
                            size: Size {
                                width: ScaledPixels(100.0 * scale),
                                height: ScaledPixels(100.0 * scale),
                            },
                        },
                        corner_radii: Default::default(),
                    },
                    background: gpui::solid_background(color),
                    border_color: Hsla::default(),
                    corner_radii: Corners::default(),
                    border_widths: Edges::default(),
                    transformation: TransformationMatrix::unit(),
                });
            }

            // Overlay a backdrop blur pane on the right half (x: 50..100)
            // with 10px logical blur radius (which scales to 10*scale physical pixels)
            scene.insert_primitive(BackdropBlur {
                order: 1,
                pad: 0,
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(50.0 * scale),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(50.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                corner_radii: Corners::default(),
                blur_radius: ScaledPixels(10.0 * scale),
                saturation: 1.0,
                tint: Hsla::default(),
                transformation: TransformationMatrix::unit(),
            });
            scene.finish();
            assert!(renderer.draw(&scene));
            let bytes = renderer.read_pixels().expect("read pixels");
            let pitch = (100.0 * scale) as usize * 4;
            let pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]
            };

            // 1. Uncovered left half: high-frequency pattern remains exact and unblurred
            // At x=12 (even stripe index 6 -> white) and x=10 (odd stripe index 5 -> black)
            let unblurred_white = pixel_at(12.0, 50.0);
            let unblurred_black = pixel_at(10.0, 50.0);
            assert!(
                unblurred_white[0] > 230 && unblurred_white[1] > 230 && unblurred_white[2] > 230,
                "uncovered white stripe must stay white at scale {scale}, got {unblurred_white:?}"
            );
            assert!(
                unblurred_black[0] < 25 && unblurred_black[1] < 25 && unblurred_black[2] < 25,
                "uncovered black stripe must stay black at scale {scale}, got {unblurred_black:?}"
            );

            // 2. Blurred right half: pixels in interior are smoothed average of pattern underneath
            // Low variance: every pixel in the interior of the blurred pane should be ~128 (gray)
            let mut blurred_vals = Vec::new();
            for lx in 65..85 {
                let p = pixel_at(lx as f32, 50.0);
                blurred_vals.push(p[0] as f32);
            }
            let mean: f32 = blurred_vals.iter().sum::<f32>() / (blurred_vals.len() as f32);
            let variance: f32 = blurred_vals
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f32>()
                / (blurred_vals.len() as f32);
            let mut unblurred_vals = Vec::new();
            for lx in 15..35 {
                let p = pixel_at(lx as f32, 50.0);
                unblurred_vals.push(p[0] as f32);
            }
            let unblurred_mean: f32 =
                unblurred_vals.iter().sum::<f32>() / (unblurred_vals.len() as f32);
            let unblurred_variance: f32 = unblurred_vals
                .iter()
                .map(|v| (v - unblurred_mean) * (v - unblurred_mean))
                .sum::<f32>()
                / (unblurred_vals.len() as f32);

            assert!(
                unblurred_variance > 15000.0,
                "unblurred region variance must be high (>15000) across stripes at scale {scale}, got {unblurred_variance}"
            );
            assert!(
                variance < unblurred_variance * 0.10,
                "blurred region variance ({variance}) must be a small fraction (<10%) of unblurred variance ({unblurred_variance}) at scale {scale}"
            );

            // 3. Test saturation: with colored background, saturation=0.0 desaturates to grayscale
            let mut sat_scene = Scene::default();
            // Solid red background
            sat_scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                background: gpui::solid_background(Hsla {
                    h: 0.0,
                    s: 1.0,
                    l: 0.5,
                    a: 1.0,
                }),
                border_color: Hsla::default(),
                corner_radii: Corners::default(),
                border_widths: Edges::default(),
                transformation: TransformationMatrix::unit(),
            });

            // Grayscale blur pane over right half with saturation=0.0
            sat_scene.insert_primitive(BackdropBlur {
                order: 1,
                pad: 0,
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(50.0 * scale),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(50.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                corner_radii: Corners::default(),
                blur_radius: ScaledPixels(5.0 * scale),
                saturation: 0.0,
                tint: Hsla::default(),
                transformation: TransformationMatrix::unit(),
            });

            assert!(renderer.draw(&sat_scene));
            let sat_bytes = renderer.read_pixels().expect("read pixels");
            let sat_pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    sat_bytes[offset],
                    sat_bytes[offset + 1],
                    sat_bytes[offset + 2],
                    sat_bytes[offset + 3],
                ]
            };

            // Uncovered left half remains vivid red
            let vivid_red = sat_pixel_at(25.0, 50.0);
            assert!(
                vivid_red[0] > 200 && vivid_red[1] < 30 && vivid_red[2] < 30,
                "uncovered red background must stay red at scale {scale}, got {vivid_red:?}"
            );

            // Blurred right half with saturation=0.0 must be grayscale (R == G == B)
            let desat = sat_pixel_at(75.0, 50.0);
            let diff_rg = (desat[0] as i32 - desat[1] as i32).abs();
            let diff_gb = (desat[1] as i32 - desat[2] as i32).abs();
            assert!(
                diff_rg < 5 && diff_gb < 5 && desat[0] > 30,
                "saturation=0.0 blur pane must convert colored background to grayscale at scale {scale}, got {desat:?}"
            );
        }
    }

    #[test]
    fn test_path_clipped_subtree_at_scale_factors() {
        use gpui::{
            Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Path, Point, ScaledPixels,
            Scene, Size, TransformationMatrix, px, rgb, size,
        };

        for scale in [1.0, 2.0] {
            let width = (100.0 * scale) as i32;
            let height = (100.0 * scale) as i32;
            let instance = WgpuContext::surfaceless_instance();
            let context =
                WgpuContext::new_surfaceless(instance, None).expect("surfaceless context");
            let size_device = size(DevicePixels(width), DevicePixels(height));
            let mut renderer =
                WgpuRenderer::new_offscreen(&context, size_device).expect("renderer");

            let mut scene = Scene::default();

            // Base black background quad (0, 0, 100, 100)
            scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                background: gpui::solid_background(rgb(0x000000)),
                border_color: Hsla::default(),
                corner_radii: Corners::default(),
                border_widths: Edges::default(),
                transformation: TransformationMatrix::unit(),
            });

            // Triangular clip path: (10, 10) -> (90, 10) -> (10, 90) -> (10, 10)
            let mut path = Path::new(Point::new(px(10.0), px(10.0)));
            path.line_to(Point::new(px(90.0), px(10.0)));
            path.line_to(Point::new(px(10.0), px(90.0)));
            path.line_to(Point::new(px(10.0), px(10.0)));
            path.color = gpui::solid_background(rgb(0xffffff));
            path.content_mask = ContentMask {
                bounds: Bounds::new(
                    Point::new(px(0.0), px(0.0)),
                    Size::new(px(100.0), px(100.0)),
                ),
                corner_radii: Default::default(),
            };
            let path_scaled = path.scale(scale);
            scene.push_layer_mask(gpui::LayerMask::Path(path_scaled));

            // Subtree quad: bright green covering (0, 0, 100, 100)
            scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: Bounds {
                    origin: Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: Size {
                        width: ScaledPixels(100.0 * scale),
                        height: ScaledPixels(100.0 * scale),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds {
                        origin: Point {
                            x: ScaledPixels(0.0),
                            y: ScaledPixels(0.0),
                        },
                        size: Size {
                            width: ScaledPixels(100.0 * scale),
                            height: ScaledPixels(100.0 * scale),
                        },
                    },
                    corner_radii: Default::default(),
                },
                background: gpui::solid_background(rgb(0x00ff00)),
                border_color: Hsla::default(),
                corner_radii: Corners::default(),
                border_widths: Edges::default(),
                transformation: TransformationMatrix::unit(),
            });

            scene.pop_layer_mask();
            scene.finish();

            assert!(renderer.draw(&scene));
            let bytes = renderer.read_pixels().expect("read rendered pixels");
            let device_w = (100.0 * scale) as usize;
            let pitch = device_w * 4;

            let pixel_at = |lx: f32, ly: f32| -> [u8; 4] {
                let px = (lx * scale) as usize;
                let py = (ly * scale) as usize;
                let offset = py * pitch + px * 4;
                [
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]
            };

            // Inside triangle: (25, 25) must be green
            let inside = pixel_at(25.0, 25.0);
            assert!(
                inside[1] > 200 && inside[0] < 30 && inside[2] < 30,
                "inside path clip pixel at (25,25) must be green at scale {scale}, got {inside:?}"
            );

            // Outside triangle: (75, 75) must be black
            let outside = pixel_at(75.0, 75.0);
            assert_eq!(
                outside[1], 0,
                "outside path clip pixel at (75,75) must be black at scale {scale}, got {outside:?}"
            );
        }
    }
}
