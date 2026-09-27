//! The offscreen layers that masked subtrees draw into.
//!
//! Each open layer mask draws its subtree into the layer texture of its
//! nesting depth. A layer texture is created the first time a frame opens a
//! mask at its depth, is reused by later frames, and is released after
//! [`gpui::LAYER_IDLE_RELEASE_FRAMES`] consecutive drawn frames that open no
//! mask at its depth. A frame without masks creates no texture.

use gpui::{LayerIdleCounter, LayerMask};

/// The layer textures of a renderer, one per nesting depth.
#[derive(Default)]
pub(super) struct LayerTextures {
    layers: Vec<LayerTexture>,
}

struct LayerTexture {
    /// The view of the texture; the view keeps the texture alive.
    view: wgpu::TextureView,
    idle: LayerIdleCounter,
}

impl LayerTextures {
    /// The view of the layer at `depth`, creating the texture when absent.
    /// A frame opens depth `n` only while depths `0..n` are open, so the
    /// layers at shallower depths already exist.
    pub(super) fn view(
        &mut self,
        device: &wgpu::Device,
        depth: usize,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> wgpu::TextureView {
        debug_assert!(depth <= self.layers.len(), "layer depth {depth} skipped");
        if depth == self.layers.len() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("layer_mask_layer"),
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
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.layers.push(LayerTexture {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                idle: LayerIdleCounter::default(),
            });
        }
        self.layers[depth].view.clone()
    }

    /// Records a drawn frame whose deepest nesting of open masks was
    /// `depth`, zero for a frame without masks, and releases every layer
    /// idle for [`gpui::LAYER_IDLE_RELEASE_FRAMES`] frames. A deeper layer
    /// is never used more recently than a shallower one, so the released
    /// layers are the deepest ones.
    pub(super) fn end_frame(&mut self, depth: usize) {
        let mut kept = self.layers.len();
        for (layer_depth, layer) in self.layers.iter_mut().enumerate() {
            if layer.idle.tick(layer_depth < depth) {
                kept = kept.min(layer_depth);
            }
        }
        self.layers.truncate(kept);
    }

    /// Releases every layer texture.
    pub(super) fn clear(&mut self) {
        self.layers.clear();
    }

    /// The number of layer textures held.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.layers.len()
    }
}

/// A layer mask open at the current point of a frame.
pub(super) struct OpenLayer {
    pub(super) mask: LayerMask,
    /// The layer the masked subtree draws into.
    pub(super) view: wgpu::TextureView,
}

/// The view primitives draw into: the innermost open layer, or `frame`
/// when no layer is open.
pub(super) fn draw_target<'a>(
    open: &'a [OpenLayer],
    frame: &'a wgpu::TextureView,
) -> &'a wgpu::TextureView {
    open.last().map_or(frame, |layer| &layer.view)
}
