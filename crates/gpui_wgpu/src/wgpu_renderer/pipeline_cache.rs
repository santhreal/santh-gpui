//! The shader modules, bind group layouts, and render pipelines the
//! renderers of one device share.
//!
//! A pipeline set depends on its target's format, on whether the target
//! composites premultiplied alpha, on the path sample count, and on whether
//! the target accepts copies. Its shader modules and bind group layouts
//! depend on dual-source blending and on the instance data transport. None
//! of these depends on the window, so the renderers of a device whose
//! targets have one [`PipelineKey`] draw with one set: the first renderer of
//! a key creates it, and every later renderer of the key takes it from the
//! cache.
//!
//! The cache is part of the [`WgpuContext`](crate::WgpuContext) of its
//! device. Device recovery replaces the context, which drops the cache and
//! every set of the lost device with it.

use super::pipelines::{WgpuModules, WgpuPipelines};
use parking_lot::Mutex;
use std::sync::Arc;

#[cfg(test)]
mod tests;

/// What the shader modules and bind group layouts depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ModuleKey {
    /// Subpixel text is drawn with dual-source blending.
    pub(super) dual_source_blending: bool,
    /// Instance data is read from a texture instead of a storage buffer.
    pub(super) uses_webgl_instance_data: bool,
}

/// What a pipeline set depends on, besides its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PipelineKey {
    pub(super) modules: ModuleKey,
    /// The format of the target.
    pub(super) format: wgpu::TextureFormat,
    /// The target composites premultiplied alpha, which selects the blend
    /// of the drawing pipelines. Every other alpha mode blends straight
    /// alpha, so targets of those modes share one set.
    pub(super) premultiplied_alpha: bool,
    /// The sample count of path rasterization.
    pub(super) path_sample_count: u32,
    /// The target accepts copies (`COPY_DST`). A target that does not is
    /// presented by a pipeline of its own.
    pub(super) copyable: bool,
}

impl PipelineKey {
    /// The key of the pipelines that draw to a target configured as
    /// `config`, rasterizing paths with `path_sample_count` samples.
    pub(super) fn new(
        modules: ModuleKey,
        config: &wgpu::SurfaceConfiguration,
        path_sample_count: u32,
    ) -> Self {
        Self {
            modules,
            format: config.format,
            premultiplied_alpha: config.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied,
            path_sample_count,
            copyable: config.usage.contains(wgpu::TextureUsages::COPY_DST),
        }
    }
}

/// The module sets and pipeline sets created on one device, at most one of
/// each per key. A set lives as long as the cache: a process draws with a
/// few keys, and a window that opens again takes its set instead of
/// compiling it.
#[derive(Default)]
pub(crate) struct PipelineCache {
    sets: Mutex<Sets>,
}

#[derive(Default)]
struct Sets {
    modules: Vec<Arc<WgpuModules>>,
    pipelines: Vec<Arc<WgpuPipelines>>,
}

impl PipelineCache {
    /// The pipelines of `key`. `device` is the device of the context that
    /// holds the cache.
    ///
    /// The first call for a key creates its pipelines, and the first call
    /// for a [`ModuleKey`] creates the modules they are created from; every
    /// later call returns the same set. The cache stays locked while a set is
    /// created, so concurrent calls for one key create one set.
    // wgpu objects are neither `Send` nor `Sync` on the web, which runs the
    // renderer on one thread.
    #[cfg_attr(target_family = "wasm", allow(clippy::arc_with_non_send_sync))]
    pub(super) fn pipelines(&self, device: &wgpu::Device, key: PipelineKey) -> Arc<WgpuPipelines> {
        let mut sets = self.sets.lock();
        if let Some(pipelines) = sets.pipelines.iter().find(|set| set.key() == key) {
            return Arc::clone(pipelines);
        }
        let modules = match sets.modules.iter().find(|set| set.key() == key.modules) {
            Some(modules) => Arc::clone(modules),
            None => {
                let modules = Arc::new(WgpuModules::new(device, key.modules));
                sets.modules.push(Arc::clone(&modules));
                modules
            }
        };
        let pipelines = Arc::new(WgpuPipelines::new(device, &modules, key));
        sets.pipelines.push(Arc::clone(&pipelines));
        pipelines
    }

    /// The number of module sets and of pipeline sets the cache holds.
    #[cfg(test)]
    fn set_counts(&self) -> (usize, usize) {
        let sets = self.sets.lock();
        (sets.modules.len(), sets.pipelines.len())
    }
}
