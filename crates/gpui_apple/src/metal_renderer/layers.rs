//! Offscreen targets and pipelines for backdrop blurs and layer masks.
//!
//! A frame with backdrop blurs draws into an offscreen frame texture, because
//! a blur samples the frame drawn so far and a framebuffer-only drawable is
//! not readable. Each blur batch copies that texture into the backdrop
//! texture, and the finished frame is copied into the drawable. A layer mask
//! draws its subtree into a cleared layer texture, one per nesting depth, and
//! composites the layer onto its parent scaled by the mask value: the clip
//! path's coverage or the square of the edge fade ratio.
//!
//! Frames without blurs or masks create none of these resources. Textures are
//! released after [`gpui::LAYER_IDLE_RELEASE_FRAMES`] consecutive frames
//! without a use, and pipelines are compiled on first use.

use super::{
    BackdropBlurInputIndex, EdgeFadeInputIndex, FrameCopyInputIndex, InstanceBinding,
    PathClipInputIndex, SpriteInputIndex,
};
use anyhow::{Context as _, Result};
use gpui::{DevicePixels, EdgeFadeMask, LayerIdleCounter, Size};
use metal::{MTLPixelFormat, MTLStorageMode, MTLTextureUsage};
use std::{mem, ops::Range};

pub(super) struct RenderLayers {
    library: metal::Library,
    pipelines: Option<LayerPipelines>,
    size: Size<DevicePixels>,
    frame: Option<metal::Texture>,
    backdrop: Option<metal::Texture>,
    mask_layers: Vec<metal::Texture>,
    blur_idle: LayerIdleCounter,
    mask_idle: LayerIdleCounter,
}

struct LayerPipelines {
    backdrop_blur: metal::RenderPipelineState,
    path_clip_composite: metal::RenderPipelineState,
    edge_fade_composite: metal::RenderPipelineState,
    frame_copy: metal::RenderPipelineState,
}

/// The textures a frame with backdrop blurs draws through.
pub(super) struct BackdropTargets {
    /// Offscreen texture the frame is drawn into.
    pub frame: metal::Texture,
    /// Copy of `frame` that blur batches sample.
    pub backdrop: metal::Texture,
}

/// The mask value a layer is composited through, as the composite pipelines
/// read it.
pub(super) enum CompositeMask<'a> {
    /// The clip path rasterized into this texture; its alpha is the mask.
    Path(&'a metal::TextureRef),
    /// The edge fade; the mask is the square of its fade ratio.
    EdgeFade(&'a EdgeFadeMask),
}

impl RenderLayers {
    pub fn new(library: metal::Library) -> Self {
        Self {
            library,
            pipelines: None,
            size: Size::default(),
            frame: None,
            backdrop: None,
            mask_layers: Vec::new(),
            blur_idle: LayerIdleCounter::default(),
            mask_idle: LayerIdleCounter::default(),
        }
    }

    /// Returns the frame and backdrop textures at `size`, creating them when
    /// missing or sized differently.
    pub fn backdrop_targets(
        &mut self,
        device: &metal::DeviceRef,
        size: Size<DevicePixels>,
    ) -> Result<BackdropTargets> {
        self.match_size(size)?;
        let frame = self
            .frame
            .get_or_insert_with(|| new_target(device, size))
            .clone();
        let backdrop = self
            .backdrop
            .get_or_insert_with(|| new_target(device, size))
            .clone();
        Ok(BackdropTargets { frame, backdrop })
    }

    /// Returns the cleared-on-use layer texture for mask nesting `depth`,
    /// creating the textures of every shallower depth that is missing.
    pub fn mask_layer(
        &mut self,
        device: &metal::DeviceRef,
        size: Size<DevicePixels>,
        depth: usize,
    ) -> Result<metal::Texture> {
        self.match_size(size)?;
        while self.mask_layers.len() <= depth {
            self.mask_layers.push(new_target(device, size));
        }
        Ok(self.mask_layers[depth].clone())
    }

    /// The layer textures held for mask nesting depths, shallowest first.
    #[cfg(test)]
    pub fn mask_layers(&self) -> &[metal::Texture] {
        &self.mask_layers
    }

    /// Records whether the finished frame used blurs and masks, and releases
    /// the textures of a feature idle for
    /// [`gpui::LAYER_IDLE_RELEASE_FRAMES`] frames.
    pub fn end_frame(&mut self, used_blur: bool, used_mask: bool) {
        if self.blur_idle.tick(used_blur) {
            self.frame = None;
            self.backdrop = None;
        }
        if self.mask_idle.tick(used_mask) {
            self.mask_layers.clear();
        }
    }

    fn match_size(&mut self, size: Size<DevicePixels>) -> Result<()> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "cannot create a {size:?} offscreen layer"
        );
        if self.size != size {
            self.size = size;
            self.frame = None;
            self.backdrop = None;
            self.mask_layers.clear();
        }
        Ok(())
    }

    fn pipelines(&mut self, device: &metal::DeviceRef) -> Result<&LayerPipelines> {
        if self.pipelines.is_none() {
            self.pipelines = Some(LayerPipelines::new(device, &self.library)?);
        }
        self.pipelines
            .as_ref()
            .context("layer pipelines missing after creation")
    }

    /// Copies `targets.frame` into `targets.backdrop`.
    pub fn snapshot_frame(command_buffer: &metal::CommandBufferRef, targets: &BackdropTargets) {
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture(
            &targets.frame,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize {
                width: targets.frame.width(),
                height: targets.frame.height(),
                depth: 1,
            },
            &targets.backdrop,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();
    }

    pub fn draw_backdrop_blurs(
        &mut self,
        device: &metal::DeviceRef,
        blurs: Range<usize>,
        instances: &InstanceBinding,
        unit_vertices: &metal::BufferRef,
        backdrop: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        if blurs.is_empty() {
            return Ok(());
        }
        let pipelines = self.pipelines(device)?;
        command_encoder.set_render_pipeline_state(&pipelines.backdrop_blur);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instances.buffer),
            instances.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instances.buffer),
            instances.offset as u64,
        );
        let viewport_bytes = &viewport_size as *const Size<DevicePixels> as *const _;
        command_encoder.set_vertex_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            viewport_bytes,
        );
        command_encoder.set_fragment_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            viewport_bytes,
        );
        command_encoder
            .set_fragment_texture(BackdropBlurInputIndex::Backdrop as u64, Some(backdrop));
        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            blurs.len() as u64,
            blurs.start as u64,
        );
        Ok(())
    }

    /// Draws `layer` onto the encoder's target inside `sprite` (the mask's
    /// bounds), scaled by the value of `mask`.
    pub fn draw_layer_composite(
        &mut self,
        device: &metal::DeviceRef,
        sprite: &InstanceBinding,
        unit_vertices: &metal::BufferRef,
        layer: &metal::TextureRef,
        mask: CompositeMask<'_>,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let pipelines = self.pipelines(device)?;
        match mask {
            CompositeMask::Path(coverage) => {
                command_encoder.set_render_pipeline_state(&pipelines.path_clip_composite);
                command_encoder.set_fragment_texture(PathClipInputIndex::Layer as u64, Some(layer));
                command_encoder
                    .set_fragment_texture(PathClipInputIndex::Mask as u64, Some(coverage));
            }
            CompositeMask::EdgeFade(fade) => {
                command_encoder.set_render_pipeline_state(&pipelines.edge_fade_composite);
                command_encoder.set_fragment_texture(EdgeFadeInputIndex::Layer as u64, Some(layer));
                command_encoder.set_fragment_bytes(
                    EdgeFadeInputIndex::Mask as u64,
                    mem::size_of_val(fade) as u64,
                    fade as *const EdgeFadeMask as *const _,
                );
            }
        }
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite.buffer),
            sprite.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.draw_primitives_instanced(metal::MTLPrimitiveType::Triangle, 0, 6, 1);
        Ok(())
    }

    /// Replaces every pixel of `target` with the matching pixel of `frame`.
    pub fn copy_frame(
        &mut self,
        device: &metal::DeviceRef,
        command_buffer: &metal::CommandBufferRef,
        frame: &metal::TextureRef,
        target: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        let pipelines = self.pipelines(device)?;
        let command_encoder = super::new_command_encoder_for_texture(
            command_buffer,
            target,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., 0.)),
        );
        command_encoder.set_render_pipeline_state(&pipelines.frame_copy);
        command_encoder.set_fragment_texture(FrameCopyInputIndex::Frame as u64, Some(frame));
        command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
        command_encoder.end_encoding();
        Ok(())
    }
}

fn new_target(device: &metal::DeviceRef, size: Size<DevicePixels>) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_width(size.width.0 as u64);
    descriptor.set_height(size.height.0 as u64);
    descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    device.new_texture(&descriptor)
}

impl LayerPipelines {
    fn new(device: &metal::DeviceRef, library: &metal::LibraryRef) -> Result<Self> {
        Ok(Self {
            backdrop_blur: super::build_pipeline_state(
                device,
                library,
                "backdrop_blur",
                "backdrop_blur_vertex",
                "backdrop_blur_fragment",
                MTLPixelFormat::BGRA8Unorm,
            ),
            path_clip_composite: super::build_path_sprite_pipeline_state(
                device,
                library,
                "path_clip_composite",
                "path_sprite_vertex",
                "path_clip_composite_fragment",
                MTLPixelFormat::BGRA8Unorm,
            ),
            edge_fade_composite: super::build_path_sprite_pipeline_state(
                device,
                library,
                "edge_fade_composite",
                "path_sprite_vertex",
                "edge_fade_composite_fragment",
                MTLPixelFormat::BGRA8Unorm,
            ),
            frame_copy: build_copy_pipeline_state(device, library)?,
        })
    }
}

fn build_copy_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
) -> Result<metal::RenderPipelineState> {
    let vertex_fn = library
        .get_function("frame_copy_vertex", None)
        .map_err(anyhow::Error::msg)
        .context("locating frame_copy_vertex")?;
    let fragment_fn = library
        .get_function("frame_copy_fragment", None)
        .map_err(anyhow::Error::msg)
        .context("locating frame_copy_fragment")?;
    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("frame_copy");
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor
        .color_attachments()
        .object_at(0)
        .context("frame_copy pipeline has no color attachment")?;
    color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    color_attachment.set_blending_enabled(false);
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(anyhow::Error::msg)
        .context("creating the frame_copy pipeline")
}
