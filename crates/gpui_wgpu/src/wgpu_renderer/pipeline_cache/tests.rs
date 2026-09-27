//! WHY: every renderer of a device takes its shader modules, bind group
//! layouts, and render pipelines from the device's [`PipelineCache`]. These
//! tests close three defect classes of that sharing:
//!
//! - a renderer that creates a set of its own for a key the cache already
//!   holds, which recompiles every shader and pipeline on each window open;
//! - a key that leaves out something pipeline creation reads, which hands a
//!   renderer pipelines built for another target (another format, blend,
//!   sample count, or presentation path);
//! - a set that outlives the device it was created on, which hands a
//!   renderer of a recovered context pipelines of the lost device.
//!
//! They do not catch a pipeline that differs between keys in a way no key
//! field selects; `WgpuPipelines::new` reads nothing but its key and modules.

use super::{ModuleKey, PipelineCache, PipelineKey};
use crate::wgpu_renderer::{WgpuRenderTarget, WgpuRenderer, WgpuSurfaceConfig};
use crate::{GpuContext, WgpuAtlas, WgpuContext};
use gpui::{
    Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point, Quad, ScaledPixels, Scene,
    Size, TransformationMatrix, solid_background,
};
use std::sync::{Arc, Barrier};
use wgpu::CompositeAlphaMode;

const EXTENT: i32 = 16;

/// A context of the test's own, so its cache holds only the test's sets.
fn context() -> WgpuContext {
    WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context")
}

fn extent() -> Size<DevicePixels> {
    Size {
        width: DevicePixels(EXTENT),
        height: DevicePixels(EXTENT),
    }
}

fn renderer(context: &WgpuContext) -> WgpuRenderer {
    WgpuRenderer::new_offscreen(context, extent()).expect("offscreen renderer")
}

/// A renderer drawing to an offscreen target of `format`.
fn renderer_with_format(context: &WgpuContext, format: wgpu::TextureFormat) -> WgpuRenderer {
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pipeline_cache_test_target"),
        size: wgpu::Extent3d {
            width: EXTENT as u32,
            height: EXTENT as u32,
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
    WgpuRenderer::new_internal(
        None,
        context,
        WgpuRenderTarget::Offscreen {
            texture,
            view,
            format,
        },
        WgpuSurfaceConfig {
            size: extent(),
            transparent: true,
            preferred_present_mode: None,
        },
        None,
        Arc::new(WgpuAtlas::from_context(context)),
    )
    .expect("offscreen renderer")
}

fn pipelines(renderer: &WgpuRenderer) -> &Arc<super::WgpuPipelines> {
    &renderer.resources().pipelines
}

/// An opaque red quad over the whole target.
fn red_frame() -> Scene {
    let bounds = Bounds {
        origin: Point {
            x: ScaledPixels(0.0),
            y: ScaledPixels(0.0),
        },
        size: Size {
            width: ScaledPixels(EXTENT as f32),
            height: ScaledPixels(EXTENT as f32),
        },
    };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        order: 0,
        border_style: Default::default(),
        bounds,
        content_mask: ContentMask {
            bounds,
            corner_radii: Default::default(),
        },
        background: solid_background(Hsla {
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
    scene.finish();
    scene
}

/// Draws the red frame and asserts every pixel is opaque red.
fn assert_draws_red(renderer: &mut WgpuRenderer) {
    assert!(renderer.draw(&red_frame()), "the frame was not drawn");
    let pixels = renderer.read_pixels().expect("read pixels");
    for pixel in pixels.chunks_exact(4) {
        assert_eq!(pixel, [255, 0, 0, 255], "the frame is not opaque red");
    }
}

#[test]
fn renderers_of_one_device_share_one_set() {
    let context = context();
    let first = renderer(&context);
    let mut second = renderer(&context);
    assert!(
        Arc::ptr_eq(pipelines(&first), pipelines(&second)),
        "the second renderer created pipelines of its own"
    );
    assert!(
        std::ptr::eq(pipelines(&first).layouts(), pipelines(&second).layouts()),
        "the second renderer created bind group layouts of its own"
    );
    assert_eq!(context.pipeline_cache.set_counts(), (1, 1));
    assert_draws_red(&mut second);
}

#[test]
fn each_key_field_selects_its_own_set() {
    let context = context();
    let device = &context.device;
    let cache = PipelineCache::default();
    let base = PipelineKey {
        modules: ModuleKey {
            dual_source_blending: context.supports_dual_source_blending(),
            uses_webgl_instance_data: false,
        },
        format: wgpu::TextureFormat::Bgra8Unorm,
        premultiplied_alpha: false,
        path_sample_count: 1,
        copyable: true,
    };
    let multisampled = [4, 2]
        .into_iter()
        .find(|&count| {
            context
                .adapter
                .get_texture_format_features(base.format)
                .flags
                .sample_count_supported(count)
        })
        .expect("the adapter multisamples the target format");
    // Destructured without `..`: a new key field fails to compile until it
    // has a variant here.
    let PipelineKey {
        modules:
            ModuleKey {
                dual_source_blending,
                uses_webgl_instance_data,
            },
        format: _,
        premultiplied_alpha,
        path_sample_count: _,
        copyable,
    } = base;
    let variants = [
        (
            "dual_source_blending",
            PipelineKey {
                modules: ModuleKey {
                    dual_source_blending: !dual_source_blending,
                    ..base.modules
                },
                ..base
            },
        ),
        (
            "uses_webgl_instance_data",
            PipelineKey {
                modules: ModuleKey {
                    uses_webgl_instance_data: !uses_webgl_instance_data,
                    ..base.modules
                },
                ..base
            },
        ),
        (
            "format",
            PipelineKey {
                format: wgpu::TextureFormat::Rgba8Unorm,
                ..base
            },
        ),
        (
            "premultiplied_alpha",
            PipelineKey {
                premultiplied_alpha: !premultiplied_alpha,
                ..base
            },
        ),
        (
            "path_sample_count",
            PipelineKey {
                path_sample_count: multisampled,
                ..base
            },
        ),
        (
            "copyable",
            PipelineKey {
                copyable: !copyable,
                ..base
            },
        ),
    ];

    let base_set = cache.pipelines(device, base);
    for (field, key) in variants {
        let set = cache.pipelines(device, key);
        assert_eq!(set.key(), key, "{field}: the set of another key");
        assert!(
            !Arc::ptr_eq(&set, &base_set),
            "{field}: the key's set is the base key's"
        );
        assert!(
            Arc::ptr_eq(&set, &cache.pipelines(device, key)),
            "{field}: a second request created another set"
        );
        let shares_modules = key.modules == base.modules;
        assert_eq!(
            std::ptr::eq(set.layouts(), base_set.layouts()),
            shares_modules,
            "{field}: modules shared with the base key: {}",
            !shares_modules
        );
    }
    assert!(Arc::ptr_eq(&base_set, &cache.pipelines(device, base)));
    // One module set for the base key and one per module key field; one
    // pipeline set per key.
    assert_eq!(cache.set_counts(), (3, variants.len() + 1));
}

#[test]
fn a_key_derives_from_the_target_configuration() {
    let modules = ModuleKey {
        dual_source_blending: true,
        uses_webgl_instance_data: false,
    };
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: EXTENT as u32,
        height: EXTENT as u32,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: Vec::new(),
    };
    for alpha_mode in [
        CompositeAlphaMode::Auto,
        CompositeAlphaMode::Opaque,
        CompositeAlphaMode::PreMultiplied,
        CompositeAlphaMode::PostMultiplied,
        CompositeAlphaMode::Inherit,
    ] {
        // The match fails to compile when wgpu adds a mode, so the list
        // above stays complete.
        let premultiplied = match alpha_mode {
            CompositeAlphaMode::PreMultiplied => true,
            CompositeAlphaMode::Auto
            | CompositeAlphaMode::Opaque
            | CompositeAlphaMode::PostMultiplied
            | CompositeAlphaMode::Inherit => false,
        };
        config.alpha_mode = alpha_mode;
        for (usage, copyable) in [
            (wgpu::TextureUsages::RENDER_ATTACHMENT, false),
            (
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                false,
            ),
            (
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
                true,
            ),
        ] {
            config.usage = usage;
            assert_eq!(
                PipelineKey::new(modules, &config, 4),
                PipelineKey {
                    modules,
                    format: config.format,
                    premultiplied_alpha: premultiplied,
                    path_sample_count: 4,
                    copyable,
                },
                "{alpha_mode:?}, {usage:?}"
            );
        }
    }
}

#[test]
fn a_transparency_change_takes_the_set_of_its_key() {
    let context = context();
    let mut first = renderer(&context);
    let mut second = renderer(&context);
    let straight = Arc::clone(pipelines(&first));
    for renderer in [&mut first, &mut second] {
        renderer.transparent_alpha_mode = CompositeAlphaMode::PreMultiplied;
    }

    // Offscreen targets start transparent with the automatic alpha mode,
    // which blends as an opaque target does.
    first.update_transparency(false);
    assert!(
        Arc::ptr_eq(pipelines(&first), &straight),
        "an opaque target created pipelines of its own"
    );

    first.update_transparency(true);
    let premultiplied = Arc::clone(pipelines(&first));
    assert!(premultiplied.key().premultiplied_alpha);
    assert!(
        !Arc::ptr_eq(&premultiplied, &straight),
        "a premultiplied target draws with straight alpha pipelines"
    );
    assert_draws_red(&mut first);

    second.update_transparency(true);
    assert!(
        Arc::ptr_eq(pipelines(&second), &premultiplied),
        "the second premultiplied target created pipelines of its own"
    );

    first.update_transparency(false);
    assert!(
        Arc::ptr_eq(pipelines(&first), &straight),
        "the return to straight alpha created pipelines"
    );
    assert_draws_red(&mut first);
    assert_eq!(context.pipeline_cache.set_counts(), (1, 2));
}

#[test]
fn a_target_format_takes_the_set_of_its_format() {
    let context = context();
    let bgra = renderer_with_format(&context, wgpu::TextureFormat::Bgra8Unorm);
    let mut rgba = renderer_with_format(&context, wgpu::TextureFormat::Rgba8Unorm);
    let another_rgba = renderer_with_format(&context, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(
        pipelines(&rgba).key().format,
        wgpu::TextureFormat::Rgba8Unorm
    );
    assert!(
        !Arc::ptr_eq(pipelines(&bgra), pipelines(&rgba)),
        "an Rgba8Unorm target draws with Bgra8Unorm pipelines"
    );
    assert!(Arc::ptr_eq(pipelines(&rgba), pipelines(&another_rgba)));
    assert_eq!(context.pipeline_cache.set_counts(), (1, 2));
    assert_draws_red(&mut rgba);
}

#[test]
fn concurrent_requests_for_one_key_create_one_set() {
    const THREADS: usize = 8;
    let context = context();
    let key = pipelines(&renderer(&context)).key();
    let cache = PipelineCache::default();
    let barrier = Barrier::new(THREADS);
    let sets: Vec<_> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache.pipelines(&context.device, key)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().expect("request thread"))
            .collect()
    });
    for set in &sets {
        assert!(
            Arc::ptr_eq(set, &sets[0]),
            "concurrent requests created two sets"
        );
    }
    assert_eq!(cache.set_counts(), (1, 1));
}

/// Mirrors `WgpuRenderer::recover` on a lost device: the renderer drops its
/// resources, the GPU context drops the lost context, a replacement context
/// takes its place, and the renderer is created again on it.
#[test]
fn a_recovered_context_draws_with_sets_of_its_own_device() {
    let gpu_context = GpuContext::new();
    *gpu_context.borrow_mut() = Some(context());
    let mut lost = renderer(gpu_context.borrow().as_ref().expect("context"));
    let lost_pipelines = Arc::downgrade(pipelines(&lost));

    lost.resources = None;
    *gpu_context.borrow_mut() = None;
    assert!(
        lost_pipelines.upgrade().is_none(),
        "the pipelines of the lost device outlived its context"
    );

    *gpu_context.borrow_mut() = Some(context());
    let current = gpu_context.borrow();
    let replacement = current.as_ref().expect("context");
    let mut recovered = renderer(replacement);
    let key = pipelines(&recovered).key();
    assert!(
        Arc::ptr_eq(
            pipelines(&recovered),
            &replacement
                .pipeline_cache
                .pipelines(&replacement.device, key)
        ),
        "the recovered renderer's pipelines are not the replacement context's"
    );
    assert_eq!(replacement.pipeline_cache.set_counts(), (1, 1));
    assert_draws_red(&mut recovered);
}
