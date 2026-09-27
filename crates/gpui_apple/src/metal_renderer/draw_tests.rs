//! Draws layer masks with the Metal renderer into an offscreen texture at 1x
//! and 2x scale, and compares every pixel with the composite documented on
//! [`gpui::StartLayerMask`] and the fade documented on
//! [`gpui::EdgeFadeMask`], computed on the CPU. Also checks the mask layer
//! textures: none for a frame without masks, one per nesting depth, reused
//! across frames, and released after [`LAYER_IDLE_RELEASE_FRAMES`] idle
//! frames.
//!
//! The pixel checks do not catch an error below [`TOLERANCE`], and treat a
//! pixel within [`EDGE`] of a clip path outline as antialiased: it only has
//! to lie between the results for path coverage 0 and 1.

mod reference;

use super::{InstanceBufferPool, MetalRenderer};
use foreign_types::ForeignType;
use gpui::{
    BackdropBlur, Bounds, ContentMask, Corners, DevicePixels, EdgeFadeMask, Edges, Hsla,
    LAYER_IDLE_RELEASE_FRAMES, LayerMask, Pixels, Quad, ScaledPixels, Scene, black, hsla, point,
    px, size, white,
};
use parking_lot::Mutex;
use reference::{Image, Outline, Rgba, composite, fade_mask, over, rgba};
use std::sync::Arc;

/// Width and height of the render target, in logical pixels.
const LOGICAL_SIZE: f32 = 60.0;
/// The scale factors every pixel check runs at.
const SCALES: [f32; 2] = [1.0, 2.0];
/// Largest difference, in 8-bit steps, between a drawn channel and the
/// reference.
const TOLERANCE: f32 = 3.0;
/// Distance, in device pixels, from a clip outline within which a pixel is
/// antialiased.
const EDGE: f32 = 1.5;
/// Left edge, in logical pixels, of the translucent red quad of
/// [`draw_content`].
const RED_LEFT: f32 = 30.0;
/// Right edge, in logical pixels, of the translucent white quad of
/// [`draw_parent_content`].
const WHITE_RIGHT: f32 = 20.0;

/// A headless renderer drawing into a square target at `scale`.
struct Target {
    renderer: MetalRenderer,
    scale: f32,
}

impl Target {
    fn new(scale: f32) -> Self {
        assert!(
            metal::Device::system_default().is_some(),
            "the Metal draw tests need a Metal device"
        );
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        Self {
            renderer: MetalRenderer::new_headless(pool),
            scale,
        }
    }

    fn target_size(&self) -> gpui::Size<DevicePixels> {
        let side = DevicePixels((LOGICAL_SIZE * self.scale) as i32);
        size(side, side)
    }

    fn scene(&self, build: impl FnOnce(&mut Scene, f32)) -> Scene {
        let mut scene = Scene::default();
        build(&mut scene, self.scale);
        scene.finish();
        scene
    }

    fn draw(&mut self, build: impl FnOnce(&mut Scene, f32)) -> Image {
        let scene = self.scene(build);
        let image = self
            .renderer
            .render_scene_to_image(&scene, self.target_size())
            .expect("drawing the scene");
        Image {
            width: image.width() as i32,
            pixels: image
                .pixels()
                .map(|pixel| pixel.0.map(|channel| f32::from(channel) / 255.0))
                .collect(),
        }
    }

    /// Draws a frame without reading it back.
    fn draw_unread(&mut self, build: impl FnOnce(&mut Scene, f32)) {
        let scene = self.scene(build);
        self.renderer
            .render_scene(&scene, self.target_size())
            .expect("drawing the scene");
    }

    /// The mask layer textures the renderer holds, shallowest first.
    fn mask_layers(&self) -> Vec<*mut metal::MTLTexture> {
        self.renderer
            .layers
            .mask_layers()
            .iter()
            .map(|layer| layer.as_ptr())
            .collect()
    }
}

fn logical(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
    Bounds::new(point(px(x), px(y)), size(px(width), px(height)))
}

fn bands(top: f32, right: f32, bottom: f32, left: f32) -> Edges<Pixels> {
    Edges {
        top: px(top),
        right: px(right),
        bottom: px(bottom),
        left: px(left),
    }
}

fn window() -> Bounds<Pixels> {
    logical(0.0, 0.0, LOGICAL_SIZE, LOGICAL_SIZE)
}

fn quad(scale: f32, bounds: Bounds<Pixels>, color: Hsla) -> Quad {
    Quad {
        bounds: bounds.scale(scale),
        content_mask: window().scale(scale).into(),
        background: gpui::solid_background(color),
        ..Quad::default()
    }
}

fn blue() -> Hsla {
    hsla(2.0 / 3.0, 1.0, 0.5, 1.0)
}

fn green() -> Hsla {
    hsla(1.0 / 3.0, 1.0, 0.5, 1.0)
}

fn translucent_red() -> Hsla {
    hsla(0.0, 1.0, 0.5, 0.5)
}

fn translucent_white() -> Hsla {
    hsla(0.0, 0.0, 1.0, 0.5)
}

/// Whether device pixel column `x` has its center at or right of logical
/// column `left`.
fn right_of(scale: f32, x: i32, left: f32) -> bool {
    x as f32 + 0.5 >= left * scale
}

/// The subtree drawn inside the innermost mask: opaque green over the whole
/// target, then translucent red right of [`RED_LEFT`].
fn draw_content(scene: &mut Scene, scale: f32) {
    scene.insert_primitive(quad(scale, window(), green()));
    scene.insert_primitive(quad(
        scale,
        logical(RED_LEFT, 0.0, LOGICAL_SIZE - RED_LEFT, LOGICAL_SIZE),
        translucent_red(),
    ));
}

/// The premultiplied layer [`draw_content`] draws, at device pixel column `x`.
fn content_at(scale: f32, x: i32) -> Rgba {
    let layer = over([0.0; 4], rgba(green()));
    if right_of(scale, x, RED_LEFT) {
        over(layer, rgba(translucent_red()))
    } else {
        layer
    }
}

/// What an outer mask draws before its inner mask opens: translucent white
/// left of [`WHITE_RIGHT`].
fn draw_parent_content(scene: &mut Scene, scale: f32) {
    scene.insert_primitive(quad(
        scale,
        logical(0.0, 0.0, WHITE_RIGHT, LOGICAL_SIZE),
        translucent_white(),
    ));
}

/// The premultiplied layer [`draw_parent_content`] draws, at device pixel
/// column `x`.
fn parent_content_at(scale: f32, x: i32) -> Rgba {
    if right_of(scale, x, WHITE_RIGHT) {
        [0.0; 4]
    } else {
        over([0.0; 4], rgba(translucent_white()))
    }
}

/// A layer mask in logical pixels, drawn at any scale.
enum TestMask {
    Path(Outline),
    EdgeFade {
        bounds: Bounds<Pixels>,
        bands: Edges<Pixels>,
        clip: Bounds<Pixels>,
    },
}

impl TestMask {
    fn fade(bounds: Bounds<Pixels>, bands: Edges<Pixels>) -> Self {
        Self::EdgeFade {
            bounds,
            bands,
            clip: window(),
        }
    }

    /// The mask in device pixels at `scale`, built as the window builds it.
    fn layer_mask(&self, scale: f32) -> LayerMask {
        match self {
            Self::Path(outline) => LayerMask::Path(outline.clip_path(scale)),
            Self::EdgeFade {
                bounds,
                bands,
                clip,
            } => LayerMask::EdgeFade(edge_fade_mask(bounds, bands, clip, scale)),
        }
    }

    /// The mask value at device pixel `(x, y)`, or `None` when the pixel is
    /// antialiased by a clip path.
    fn value(&self, scale: f32, x: i32, y: i32) -> Option<f32> {
        match self {
            Self::Path(outline) => outline.coverage(scale, x, y),
            Self::EdgeFade {
                bounds,
                bands,
                clip,
            } => Some(fade_mask(&edge_fade_mask(bounds, bands, clip, scale), x, y)),
        }
    }
}

/// The edge fade of `bounds` across `bands` inside `clip`, in device pixels
/// at `scale`.
fn edge_fade_mask(
    bounds: &Bounds<Pixels>,
    bands: &Edges<Pixels>,
    clip: &Bounds<Pixels>,
    scale: f32,
) -> EdgeFadeMask {
    EdgeFadeMask::new(bounds.scale(scale), bands.scale(scale), clip.scale(scale))
}

/// The kind index of `mask`. Adding a [`LayerMask`] kind fails to compile
/// here until the draw tests cover it.
fn kind(mask: &LayerMask) -> usize {
    match mask {
        LayerMask::Path(_) => 0,
        LayerMask::EdgeFade(_) => 1,
    }
}
const KINDS: usize = 2;

/// Fails unless the masks cover every [`LayerMask`] kind.
fn assert_every_kind(masks: &[(&str, TestMask)]) {
    let mut covered = [false; KINDS];
    for (_, mask) in masks {
        covered[kind(&mask.layer_mask(1.0))] = true;
    }
    assert!(
        covered.iter().all(|covered| *covered),
        "the masks leave a layer mask kind untested: {covered:?}"
    );
}

/// The masks each drawn alone.
fn single_masks() -> Vec<(&'static str, TestMask)> {
    vec![
        (
            "rounded path",
            TestMask::Path(Outline::rounded_square(12.0, 48.0, 9.0)),
        ),
        (
            "fade with four distinct bands",
            TestMask::fade(logical(8.0, 6.0, 44.0, 46.0), bands(5.0, 9.0, 3.25, 12.0)),
        ),
        (
            "fade with bands wider than half the bounds",
            TestMask::fade(
                logical(10.0, 12.0, 40.0, 30.0),
                bands(20.0, 30.0, 0.0, 30.0),
            ),
        ),
        (
            "fade cut by the clip, with unfaded edges open to it",
            TestMask::EdgeFade {
                bounds: logical(0.0, 10.0, 60.0, 36.0),
                bands: bands(8.0, 0.0, 0.0, 20.0),
                clip: logical(12.0, 4.0, 30.0, 50.0),
            },
        ),
    ]
}

/// Outer masks of the nesting checks.
fn outer_masks() -> Vec<(&'static str, TestMask)> {
    vec![
        (
            "path",
            TestMask::Path(Outline::rounded_square(8.0, 52.0, 12.0)),
        ),
        (
            "fade",
            TestMask::fade(logical(4.0, 4.0, 52.0, 52.0), bands(10.0, 10.0, 10.0, 10.0)),
        ),
    ]
}

/// Inner masks of the nesting checks.
fn inner_masks() -> Vec<(&'static str, TestMask)> {
    vec![
        (
            "path",
            TestMask::Path(
                Outline::new((10.0, 10.0))
                    .line_to((55.0, 10.0))
                    .line_to((10.0, 55.0))
                    .line_to((10.0, 10.0)),
            ),
        ),
        (
            "fade",
            TestMask::fade(logical(10.0, 8.0, 45.0, 44.0), bands(18.0, 25.0, 0.0, 25.0)),
        ),
    ]
}

/// The range a pixel lies in when it depends linearly on each value in
/// `masks`: the pixel at the known values, spanning 0 through 1 for each
/// antialiased one.
fn span(masks: &[Option<f32>], pixel: impl Fn(&[f32]) -> Rgba) -> (Rgba, Rgba) {
    let unknown: Vec<usize> = (0..masks.len()).filter(|i| masks[*i].is_none()).collect();
    let mut lo = [f32::INFINITY; 4];
    let mut hi = [f32::NEG_INFINITY; 4];
    for corner in 0..1usize << unknown.len() {
        let values: Vec<f32> = masks
            .iter()
            .enumerate()
            .map(|(index, mask)| {
                mask.unwrap_or_else(|| {
                    let bit = unknown.iter().position(|unknown| *unknown == index);
                    bit.map_or(0.0, |bit| ((corner >> bit) & 1) as f32)
                })
            })
            .collect();
        let pixel = pixel(&values);
        for channel in 0..4 {
            lo[channel] = lo[channel].min(pixel[channel]);
            hi[channel] = hi[channel].max(pixel[channel]);
        }
    }
    (lo, hi)
}

/// Fails unless every pixel `(x, y)` for which `range` returns `(lo, hi)` has
/// each channel between `lo` and `hi`, within [`TOLERANCE`].
fn assert_pixels(name: &str, drawn: &Image, range: impl Fn(i32, i32) -> Option<(Rgba, Rgba)>) {
    let height = drawn.pixels.len() as i32 / drawn.width;
    let mut checked = 0;
    let mut worst: Option<(f32, String)> = None;
    for y in 0..height {
        for x in 0..drawn.width {
            let Some((lo, hi)) = range(x, y) else {
                continue;
            };
            checked += 1;
            let pixel = drawn.at(x, y);
            let error = (0..4)
                .map(|c| (lo[c].min(hi[c]) - pixel[c]).max(pixel[c] - lo[c].max(hi[c])))
                .fold(0.0, f32::max)
                * 255.0;
            if error > worst.as_ref().map_or(TOLERANCE, |(worst, _)| *worst) {
                let detail = format!("({x}, {y}) drew {pixel:?}, expected {lo:?}..={hi:?}");
                worst = Some((error, detail));
            }
        }
    }
    assert!(checked >= 100, "{name}: only {checked} pixels checked");
    if let Some((error, detail)) = worst {
        panic!("{name}: off by {error:.1} steps at {detail}");
    }
}

/// Catches a mask that leaks its subtree outside `LayerMask::bounds`, drops
/// it inside, composites straight instead of premultiplied color, uses `r`
/// instead of `r * r`, combines the edges by product instead of minimum,
/// fades an edge with a zero band, reads the bands in another edge order,
/// measures from `bounds` instead of `fade_bounds` or from the pixel corner
/// instead of its center, ignores the scale factor, or fills the curved
/// corners of a clip path as triangles.
#[test]
fn a_layer_mask_composites_its_premultiplied_layer_by_the_mask_value() {
    let masks = single_masks();
    assert_every_kind(&masks);
    let background = rgba(blue());
    for scale in SCALES {
        let mut target = Target::new(scale);
        for (name, mask) in &masks {
            let drawn = target.draw(|scene, scale| {
                scene.insert_primitive(quad(scale, window(), blue()));
                scene.push_layer_mask(mask.layer_mask(scale));
                draw_content(scene, scale);
                scene.pop_layer_mask();
            });
            assert_pixels(&format!("{name} at {scale}x"), &drawn, |x, y| {
                let layer = content_at(scale, x);
                Some(span(&[mask.value(scale, x, y)], |m| {
                    composite(background, layer, m[0])
                }))
            });
        }
    }
}

/// Catches an inner mask that replaces its parent's layer instead of
/// compositing onto it, composites onto the frame instead of the parent,
/// shares the parent's layer texture, or applies either mask to the other's
/// bounds, for every pairing of mask kinds.
#[test]
fn nested_layer_masks_composite_each_layer_onto_its_parent() {
    let outers = outer_masks();
    let inners = inner_masks();
    assert_every_kind(&outers);
    assert_every_kind(&inners);
    let background = rgba(blue());
    for scale in SCALES {
        let mut target = Target::new(scale);
        for (outer_name, outer) in &outers {
            for (inner_name, inner) in &inners {
                let drawn = target.draw(|scene, scale| {
                    scene.insert_primitive(quad(scale, window(), blue()));
                    scene.push_layer_mask(outer.layer_mask(scale));
                    draw_parent_content(scene, scale);
                    scene.push_layer_mask(inner.layer_mask(scale));
                    draw_content(scene, scale);
                    scene.pop_layer_mask();
                    scene.pop_layer_mask();
                });
                let name = format!("{inner_name} inside {outer_name} at {scale}x");
                assert_pixels(&name, &drawn, |x, y| {
                    let (parent, layer) = (parent_content_at(scale, x), content_at(scale, x));
                    let masks = [outer.value(scale, x, y), inner.value(scale, x, y)];
                    Some(span(&masks, |m| {
                        composite(background, composite(parent, layer, m[1]), m[0])
                    }))
                });
            }
        }
    }
}

/// Catches a backdrop blur inside nested masks that samples an open mask
/// layer, where the red quad drawn before it would show, instead of the
/// frame beneath every open layer.
#[test]
fn backdrop_blur_inside_layer_masks_samples_the_frame_beneath_them() {
    let stripe = |scale: f32, x: i32| {
        let index = ((x as f32 + 0.5) / scale / 2.0).floor() as i32;
        if index % 2 == 0 { white() } else { black() }
    };
    let outer = TestMask::Path(
        Outline::new((10.0, 10.0))
            .line_to((50.0, 10.0))
            .line_to((50.0, 50.0))
            .line_to((10.0, 50.0))
            .line_to((10.0, 10.0)),
    );
    let inner = TestMask::fade(window(), bands(0.0, 0.0, 0.0, 30.0));
    let red = hsla(0.0, 1.0, 0.5, 1.0);
    let (red_bounds, blur_bounds) = (
        logical(15.0, 15.0, 30.0, 30.0),
        logical(20.0, 20.0, 20.0, 20.0),
    );
    let inside = |scale: f32, bounds: &Bounds<Pixels>, x: i32, y: i32| {
        let bounds = bounds.scale(scale);
        let p = (x as f32 + 0.5, y as f32 + 0.5);
        p.0 >= bounds.left().0
            && p.0 < bounds.right().0
            && p.1 >= bounds.top().0
            && p.1 < bounds.bottom().0
    };
    for scale in SCALES {
        let mut target = Target::new(scale);
        let blur = BackdropBlur {
            order: 0,
            pad: 0,
            bounds: blur_bounds.scale(scale),
            content_mask: ContentMask {
                bounds: window().scale(scale),
                corner_radii: Corners::default(),
            },
            corner_radii: Corners::default(),
            blur_radius: ScaledPixels(0.0),
            saturation: 1.0,
            tint: hsla(0.0, 0.0, 0.0, 0.0),
            transformation: Default::default(),
        };
        let drawn = target.draw(|scene, scale| {
            for index in 0..(LOGICAL_SIZE / 2.0) as i32 {
                let x = index as f32 * 2.0;
                let color = if index % 2 == 0 { white() } else { black() };
                scene.insert_primitive(quad(scale, logical(x, 0.0, 2.0, LOGICAL_SIZE), color));
            }
            scene.push_layer_mask(outer.layer_mask(scale));
            scene.push_layer_mask(inner.layer_mask(scale));
            scene.insert_primitive(quad(scale, red_bounds, red));
            scene.insert_primitive(blur);
            scene.pop_layer_mask();
            scene.pop_layer_mask();
        });
        assert_pixels(&format!("blur inside masks at {scale}x"), &drawn, |x, y| {
            let frame = rgba(stripe(scale, x));
            let layer = if inside(scale, &blur_bounds, x, y) {
                frame
            } else if inside(scale, &red_bounds, x, y) {
                rgba(red)
            } else {
                [0.0; 4]
            };
            let masks = [outer.value(scale, x, y), inner.value(scale, x, y)];
            Some(span(&masks, |m| {
                composite(frame, composite([0.0; 4], layer, m[1]), m[0])
            }))
        });
    }
}

/// Catches a frame without masks that allocates a mask layer, a mask that
/// allocates a layer per mask instead of per depth, layers recreated every
/// frame, layers released before [`LAYER_IDLE_RELEASE_FRAMES`] idle frames
/// or never, and a masked frame that does not restart the idle count.
#[test]
fn mask_layers_are_allocated_per_depth_reused_and_released_when_idle() {
    let no_mask = |scene: &mut Scene, scale: f32| {
        scene.insert_primitive(quad(scale, window(), blue()));
    };
    let fade = TestMask::fade(window(), bands(10.0, 0.0, 0.0, 0.0));
    let path = TestMask::Path(Outline::rounded_square(8.0, 52.0, 12.0));
    let one_mask = |scene: &mut Scene, scale: f32| {
        scene.push_layer_mask(fade.layer_mask(scale));
        draw_content(scene, scale);
        scene.pop_layer_mask();
    };
    let two_masks = |scene: &mut Scene, scale: f32| {
        scene.push_layer_mask(path.layer_mask(scale));
        scene.push_layer_mask(fade.layer_mask(scale));
        draw_content(scene, scale);
        scene.pop_layer_mask();
        scene.pop_layer_mask();
    };
    let mut target = Target::new(1.0);

    target.draw_unread(no_mask);
    assert_eq!(target.mask_layers().len(), 0, "a frame without masks");

    target.draw_unread(two_masks);
    let layers = target.mask_layers();
    assert_eq!(layers.len(), 2, "two nested masks");
    target.draw_unread(one_mask);
    assert_eq!(target.mask_layers(), layers, "a later masked frame");

    for idle in 1..LAYER_IDLE_RELEASE_FRAMES {
        target.draw_unread(no_mask);
        assert_eq!(target.mask_layers(), layers, "after {idle} idle frames");
    }
    target.draw_unread(one_mask);
    for idle in 1..LAYER_IDLE_RELEASE_FRAMES {
        target.draw_unread(no_mask);
        assert_eq!(
            target.mask_layers(),
            layers,
            "after a masked frame and {idle} idle frames"
        );
    }
    target.draw_unread(no_mask);
    assert_eq!(
        target.mask_layers().len(),
        0,
        "after {LAYER_IDLE_RELEASE_FRAMES} idle frames"
    );

    target.draw_unread(one_mask);
    assert_eq!(
        target.mask_layers().len(),
        1,
        "a masked frame after release"
    );
}
