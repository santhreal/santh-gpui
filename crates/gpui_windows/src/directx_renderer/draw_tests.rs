//! Draws backdrop blurs, layer masks, and paths in rounded content masks
//! with the DirectX renderer into the render target of a hidden window, and
//! compares every pixel with the math documented on [`gpui::BackdropBlur`]
//! and [`gpui::StartLayerMask`], computed on the CPU from a frame drawn
//! without the primitive under test, or with the rounded rectangle of the
//! content mask.

mod reference;

use super::DirectXRenderer;
use crate::DirectXDevices;
use gpui::{
    BackdropBlur, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
    LAYER_IDLE_RELEASE_FRAMES, LayerMask, Pixels, Quad, ScaledPixels, Scene,
    WindowBackgroundAppearance, black, hsla, point, px, size, white,
};
use gpui_util::ResultExt;
use reference::{Fade, Image, Outline, Rgba, composite, expected_blur, over, quad_sdf, rgba};
use windows::{
    Win32::{
        Foundation::HWND,
        UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW,
        },
    },
    core::w,
};

/// Width and height of the render target, in device pixels.
const SIZE: i32 = 100;
/// Largest difference, in 8-bit steps, between a drawn channel and the
/// reference.
const TOLERANCE: f32 = 3.0;
/// Distance from a clip outline within which a pixel is antialiased.
const EDGE: f32 = 1.5;

struct HiddenWindow(HWND);

impl Drop for HiddenWindow {
    fn drop(&mut self) {
        unsafe { DestroyWindow(self.0) }.log_err();
    }
}

/// A renderer drawing into a `SIZE` square swap chain of a hidden window.
struct Target {
    renderer: DirectXRenderer,
    _window: HiddenWindow,
}

impl Target {
    fn new() -> Self {
        let window = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                None,
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                SIZE,
                SIZE,
                None,
                None,
                None,
                None,
            )
        }
        .map(HiddenWindow)
        .expect("creating a hidden window");
        let devices = DirectXDevices::new().expect("creating a Direct3D device");
        let mut renderer =
            DirectXRenderer::new(window.0, &devices, true).expect("creating the renderer");
        renderer
            .resize(size(DevicePixels(SIZE), DevicePixels(SIZE)))
            .expect("sizing the render target");
        Self {
            renderer,
            _window: window,
        }
    }

    fn draw(&mut self, build: impl FnOnce(&mut Scene)) -> Image {
        let mut scene = Scene::default();
        build(&mut scene);
        scene.finish();
        let image = self
            .renderer
            .render_to_image(&scene, WindowBackgroundAppearance::Opaque)
            .expect("drawing the scene");
        Image(
            image
                .pixels()
                .map(|pixel| pixel.0.map(|channel| f32::from(channel) / 255.0))
                .collect(),
        )
    }
}

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(width), ScaledPixels(height)),
    }
}

fn window_mask() -> ContentMask<ScaledPixels> {
    rect(0.0, 0.0, SIZE as f32, SIZE as f32).into()
}

fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
    Quad {
        bounds,
        content_mask: window_mask(),
        background: gpui::solid_background(color),
        ..Quad::default()
    }
}

/// White and black vertical stripes 2 pixels wide.
fn stripes(scene: &mut Scene) {
    for index in 0..SIZE / 2 {
        let color = if index % 2 == 0 { white() } else { black() };
        scene.insert_primitive(quad(rect(index as f32 * 2.0, 0.0, 2.0, SIZE as f32), color));
    }
}

/// Fails unless every pixel `(x, y)` for which `range` returns `(lo, hi)` has
/// each channel between `lo` and `hi`, within [`TOLERANCE`].
fn assert_pixels(name: &str, drawn: &Image, range: impl Fn(i32, i32) -> Option<(Rgba, Rgba)>) {
    let mut checked = 0;
    let mut worst: Option<(f32, String)> = None;
    for y in 0..SIZE {
        for x in 0..SIZE {
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

/// Catches a blur that samples the wrong texels, weighs or places a ring
/// wrongly, ignores the body corner radii, or misses the clamp at the frame
/// edge the pane touches. Does not catch an error below [`TOLERANCE`].
#[test]
fn backdrop_blur_draws_the_documented_blur_of_the_frame() {
    let mut target = Target::new();
    let frame = target.draw(stripes);
    let blur = BackdropBlur {
        order: 0,
        pad: 0,
        bounds: rect(30.0, 20.0, 70.0, 60.0),
        content_mask: window_mask(),
        corner_radii: Corners::all(ScaledPixels(10.0)),
        blur_radius: ScaledPixels(6.0),
        saturation: 1.0,
        tint: hsla(0.0, 0.0, 0.0, 0.0),
        transformation: Default::default(),
    };
    let drawn = target.draw(|scene| {
        stripes(scene);
        scene.insert_primitive(blur);
    });
    assert_pixels("blur", &drawn, |x, y| {
        let expected = expected_blur(&frame, &blur, x, y);
        Some((expected, expected))
    });
}

/// Catches saturation, tint, and content mask coverage that differ from the
/// documented math, including a rounded content mask treated as a
/// rectangle.
#[test]
fn backdrop_blur_saturates_tints_and_clips_to_rounded_masks() {
    let background = |scene: &mut Scene| {
        for (index, h) in [0.0, 1.0 / 3.0, 2.0 / 3.0].into_iter().enumerate() {
            let x = index as f32 * 34.0;
            scene.insert_primitive(quad(
                rect(x, 0.0, 34.0, SIZE as f32),
                hsla(h, 1.0, 0.5, 1.0),
            ));
        }
        scene.insert_primitive(quad(rect(0.0, 45.0, SIZE as f32, 10.0), white()));
    };
    let mut target = Target::new();
    let frame = target.draw(background);
    let blur = BackdropBlur {
        order: 0,
        pad: 0,
        bounds: rect(20.0, 20.0, 70.0, 60.0),
        content_mask: ContentMask {
            bounds: rect(30.0, 10.0, 60.0, 70.0),
            corner_radii: Corners::all(ScaledPixels(16.0)),
        },
        corner_radii: Corners::all(ScaledPixels(8.0)),
        blur_radius: ScaledPixels(5.0),
        saturation: 0.4,
        tint: hsla(0.0, 0.0, 0.2, 0.35),
        transformation: Default::default(),
    };
    let drawn = target.draw(|scene| {
        background(scene);
        scene.insert_primitive(blur);
    });
    assert_pixels("saturated blur", &drawn, |x, y| {
        let expected = expected_blur(&frame, &blur, x, y);
        Some((expected, expected))
    });
}

/// Catches a clip that leaks its subtree outside the path, drops it inside,
/// composites straight instead of premultiplied color, or fills the curved
/// corners of the path as triangles.
#[test]
fn path_clip_composites_its_premultiplied_layer_by_path_coverage() {
    let blue = rgba(hsla(2.0 / 3.0, 1.0, 0.5, 1.0));
    let green = hsla(1.0 / 3.0, 1.0, 0.5, 1.0);
    let red = hsla(0.0, 1.0, 0.5, 0.5);
    let outline = Outline::rounded_square(20.0, 80.0, 15.0);
    let mut target = Target::new();
    let drawn = target.draw(|scene| {
        scene.insert_primitive(quad(
            rect(0.0, 0.0, 100.0, 100.0),
            hsla(2.0 / 3.0, 1.0, 0.5, 1.0),
        ));
        scene.push_layer_mask(LayerMask::Path(outline.clip_path()));
        scene.insert_primitive(quad(rect(0.0, 0.0, 100.0, 50.0), green));
        scene.insert_primitive(quad(rect(50.0, 0.0, 50.0, 100.0), red));
        scene.pop_layer_mask();
    });
    assert_pixels("path clip", &drawn, |x, y| {
        let mut layer = [0.0; 4];
        if y < 50 {
            layer = over(layer, rgba(green));
        }
        if x >= 50 {
            layer = over(layer, rgba(red));
        }
        let (lo, hi) = (composite(blue, layer, 0.0), composite(blue, layer, 1.0));
        Some(match outline.coverage(x, y) {
            Some(m) if m > 0.0 => (hi, hi),
            Some(_) => (lo, lo),
            None => (lo, hi),
        })
    });
}

/// Catches a nested clip that replaces its parent's layer or composites
/// onto the frame instead of onto the parent, so the result is not the
/// intersection of both paths.
#[test]
fn nested_path_clips_intersect() {
    let blue = rgba(hsla(2.0 / 3.0, 1.0, 0.5, 1.0));
    let green = rgba(hsla(1.0 / 3.0, 1.0, 0.5, 1.0));
    let outer = Outline::rounded_square(20.0, 80.0, 15.0);
    let inner = Outline::new((10.0, 10.0))
        .line_to((90.0, 10.0))
        .line_to((10.0, 90.0))
        .line_to((10.0, 10.0));
    let mut target = Target::new();
    let drawn = target.draw(|scene| {
        scene.insert_primitive(quad(
            rect(0.0, 0.0, 100.0, 100.0),
            hsla(2.0 / 3.0, 1.0, 0.5, 1.0),
        ));
        scene.push_layer_mask(LayerMask::Path(outer.clip_path()));
        scene.push_layer_mask(LayerMask::Path(inner.clip_path()));
        scene.insert_primitive(quad(
            rect(0.0, 0.0, 100.0, 100.0),
            hsla(1.0 / 3.0, 1.0, 0.5, 1.0),
        ));
        scene.pop_layer_mask();
        scene.pop_layer_mask();
    });
    assert_pixels("nested path clips", &drawn, |x, y| {
        Some(match (outer.coverage(x, y), inner.coverage(x, y)) {
            (Some(outer), Some(inner)) if outer * inner > 0.0 => (green, green),
            (Some(0.0), _) | (_, Some(0.0)) => (blue, blue),
            _ => (blue, green),
        })
    });
}

/// Catches a blur inside a clip that samples the clip layer, where the red
/// quad drawn before it would tint the result, instead of the frame beneath
/// every open clip layer.
#[test]
fn backdrop_blur_inside_a_path_clip_samples_the_frame_beneath_it() {
    let outline = Outline::new((20.0, 20.0))
        .line_to((80.0, 20.0))
        .line_to((80.0, 80.0))
        .line_to((20.0, 80.0))
        .line_to((20.0, 20.0));
    let blur = BackdropBlur {
        order: 0,
        pad: 0,
        bounds: rect(0.0, 0.0, 100.0, 100.0),
        content_mask: window_mask(),
        corner_radii: Corners::default(),
        blur_radius: ScaledPixels(6.0),
        saturation: 1.0,
        tint: hsla(0.0, 0.0, 0.0, 0.0),
        transformation: Default::default(),
    };
    let mut target = Target::new();
    let frame = target.draw(stripes);
    let drawn = target.draw(|scene| {
        stripes(scene);
        scene.push_layer_mask(LayerMask::Path(outline.clip_path()));
        scene.insert_primitive(quad(rect(30.0, 30.0, 40.0, 40.0), hsla(0.0, 1.0, 0.5, 1.0)));
        scene.insert_primitive(blur);
        scene.pop_layer_mask();
    });
    assert_pixels("blur inside a path clip", &drawn, |x, y| {
        let pixel = match outline.coverage(x, y)? {
            m if m > 0.0 => expected_blur(&frame, &blur, x, y),
            _ => frame.at(x, y),
        };
        Some((pixel, pixel))
    });
}

/// Scale factors, in device pixels per logical pixel, the edge fade tests
/// draw at.
const SCALES: [f32; 2] = [1.0, 2.0];

fn logical(x: f32, y: f32, width: f32, height: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(px(x), px(y)),
        size: size(px(width), px(height)),
    }
}

fn edges(top: f32, right: f32, bottom: f32, left: f32) -> Edges<Pixels> {
    Edges {
        top: px(top),
        right: px(right),
        bottom: px(bottom),
        left: px(left),
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

/// Fills the target with opaque blue.
fn backdrop(scene: &mut Scene) {
    scene.insert_primitive(quad(rect(0.0, 0.0, SIZE as f32, SIZE as f32), blue()));
}

/// Draws opaque green left of logical x 25 and half transparent red right
/// of logical x 20, over the full height of the target, at `scale`.
fn layer_content(scene: &mut Scene, scale: f32) {
    scene.insert_primitive(quad(rect(0.0, 0.0, 25.0 * scale, SIZE as f32), green()));
    scene.insert_primitive(quad(
        rect(20.0 * scale, 0.0, SIZE as f32, SIZE as f32),
        translucent_red(),
    ));
}

/// The premultiplied layer [`layer_content`] draws in device pixel column
/// `x`.
fn layer_at(x: i32, scale: f32) -> Rgba {
    let x = (x as f32 + 0.5) / scale;
    let mut layer = [0.0; 4];
    if x < 25.0 {
        layer = over(layer, rgba(green()));
    }
    if x >= 20.0 {
        layer = over(layer, rgba(translucent_red()));
    }
    layer
}

/// The region the edge fade tests composite into, in logical pixels. It
/// fits the 50 logical pixel window of the 2x target.
fn fade_clip() -> Bounds<Pixels> {
    logical(2.0, 3.0, 45.0, 43.0)
}

/// The faded region of the edge fade tests, in logical pixels, inside
/// [`fade_clip`] on every side. No side is a whole device pixel at either
/// scale.
fn fade_bounds() -> Bounds<Pixels> {
    logical(5.25, 6.75, 36.5, 33.5)
}

/// The fade of the nested mask tests: every edge of [`fade_bounds`] faded
/// across 8.5 logical pixels.
fn nested_fade(scale: f32) -> Fade {
    Fade {
        bounds: fade_bounds(),
        bands: edges(8.5, 8.5, 8.5, 8.5),
        clip: fade_clip(),
        scale,
    }
}

/// The clip path of the nested mask tests, crossing every ramp of
/// [`nested_fade`].
fn nested_outline(scale: f32) -> Outline {
    Outline::rounded_square(10.0 * scale, 44.0 * scale, 8.0 * scale)
}

/// Draws [`layer_content`] through `inner` nested in `outer`, over
/// [`backdrop`].
fn draw_nested(target: &mut Target, outer: LayerMask, inner: LayerMask, scale: f32) -> Image {
    target.draw(|scene| {
        backdrop(scene);
        scene.push_layer_mask(outer);
        scene.push_layer_mask(inner);
        layer_content(scene, scale);
        scene.pop_layer_mask();
        scene.pop_layer_mask();
    })
}

/// The documented pixel `(x, y)` of [`layer_content`] composited through
/// `fade` nested in a path clip of `outline`, as the range the outline's
/// antialiasing spans.
fn fade_inside_path(fade: &Fade, outline: &Outline, x: i32, y: i32) -> (Rgba, Rgba) {
    let faded = composite([0.0; 4], layer_at(x, fade.scale), fade.m(x, y));
    let at = |coverage| composite(rgba(blue()), faded, coverage);
    match outline.coverage(x, y) {
        Some(coverage) => (at(coverage), at(coverage)),
        None => (at(0.0), at(1.0)),
    }
}

/// The documented pixel `(x, y)` of [`layer_content`] composited through a
/// path clip of `outline` nested in `fade`, as the range the outline's
/// antialiasing spans.
fn path_inside_fade(fade: &Fade, outline: &Outline, x: i32, y: i32) -> (Rgba, Rgba) {
    let at = |coverage| {
        let clipped = composite([0.0; 4], layer_at(x, fade.scale), coverage);
        composite(rgba(blue()), clipped, fade.m(x, y))
    };
    match outline.coverage(x, y) {
        Some(coverage) => (at(coverage), at(coverage)),
        None => (at(0.0), at(1.0)),
    }
}

/// Draws [`layer_content`] through `fade` over [`backdrop`] and compares
/// every pixel with the documented composite.
fn assert_fade(name: &str, target: &mut Target, fade: &Fade) {
    let drawn = target.draw(|scene| {
        backdrop(scene);
        scene.push_layer_mask(fade.mask());
        layer_content(scene, fade.scale);
        scene.pop_layer_mask();
    });
    let dst = rgba(blue());
    assert_pixels(name, &drawn, |x, y| {
        let expected = composite(dst, layer_at(x, fade.scale), fade.m(x, y));
        Some((expected, expected))
    });
}

/// Catches a ramp measured from the wrong side or in the wrong direction, a
/// band read from another edge, a dropped edge term, `m = r` in place of
/// `m = r * r`, an edge with a zero band that still limits the fade, a
/// straight instead of premultiplied composite, and a composite drawn
/// outside the clip.
#[test]
fn edge_fade_ramps_each_faded_edge_and_leaves_the_others_open() {
    let mut target = Target::new();
    for scale in SCALES {
        for (edge, bands) in [
            ("top", edges(9.5, 0.0, 0.0, 0.0)),
            ("right", edges(0.0, 9.5, 0.0, 0.0)),
            ("bottom", edges(0.0, 0.0, 9.5, 0.0)),
            ("left", edges(0.0, 0.0, 0.0, 9.5)),
        ] {
            let fade = Fade {
                bounds: fade_bounds(),
                bands,
                clip: fade_clip(),
                scale,
            };
            assert_fade(&format!("{edge} edge fade at {scale}x"), &mut target, &fade);
        }
    }
}

/// Catches `max` in place of `min` across edges, bands applied to the wrong
/// edges, and a composite drawn over the whole faded region where the clip
/// cuts through a ramp.
#[test]
fn edge_fade_takes_the_smallest_ratio_of_all_faded_edges_inside_the_clip() {
    let mut target = Target::new();
    for scale in SCALES {
        let fade = Fade {
            bounds: fade_bounds(),
            bands: edges(6.0, 9.0, 12.0, 4.5),
            // The right side of the clip cuts through the right ramp.
            clip: logical(2.0, 3.0, 34.0, 43.0),
            scale,
        };
        assert_fade(&format!("all edges fade at {scale}x"), &mut target, &fade);
    }
}

/// Catches a ratio that reaches 1 between opposite ramps that overlap, as a
/// fade that limits each band to half the faded region draws.
#[test]
fn edge_fade_bands_wider_than_half_the_bounds_overlap() {
    let mut target = Target::new();
    for scale in SCALES {
        let fade = Fade {
            bounds: logical(8.0, 6.0, 30.0, 36.0),
            bands: edges(20.0, 18.0, 0.0, 18.0),
            clip: fade_clip(),
            scale,
        };
        assert_fade(&format!("wide band fade at {scale}x"), &mut target, &fade);
    }
}

/// Catches an edge fade inside a path clip that composites onto the frame
/// instead of the clip layer, or draws into the clip layer itself, so the
/// result is not the fade scaled by the path coverage.
#[test]
fn edge_fade_inside_a_path_clip_composites_through_both() {
    let mut target = Target::new();
    for scale in SCALES {
        let fade = nested_fade(scale);
        let outline = nested_outline(scale);
        let drawn = draw_nested(
            &mut target,
            LayerMask::Path(outline.clip_path()),
            fade.mask(),
            scale,
        );
        assert_pixels(
            &format!("edge fade inside a path clip at {scale}x"),
            &drawn,
            |x, y| Some(fade_inside_path(&fade, &outline, x, y)),
        );
    }
}

/// Catches a path clip inside an edge fade that composites onto the frame
/// instead of the fade layer, or draws into the fade layer itself, so the
/// result is not the path coverage scaled by the fade.
#[test]
fn path_clip_inside_an_edge_fade_composites_through_both() {
    let mut target = Target::new();
    for scale in SCALES {
        let fade = nested_fade(scale);
        let outline = nested_outline(scale);
        let drawn = draw_nested(
            &mut target,
            fade.mask(),
            LayerMask::Path(outline.clip_path()),
            scale,
        );
        assert_pixels(
            &format!("path clip inside an edge fade at {scale}x"),
            &drawn,
            |x, y| Some(path_inside_fade(&fade, &outline, x, y)),
        );
    }
}

/// Catches a blur inside an edge fade that samples the fade layer, where
/// the red quad drawn before it would tint the result, instead of the frame
/// beneath every open layer.
#[test]
fn backdrop_blur_inside_an_edge_fade_samples_the_frame_beneath_it() {
    let blur = BackdropBlur {
        order: 0,
        pad: 0,
        bounds: rect(0.0, 0.0, 100.0, 100.0),
        content_mask: window_mask(),
        corner_radii: Corners::default(),
        blur_radius: ScaledPixels(6.0),
        saturation: 1.0,
        tint: hsla(0.0, 0.0, 0.0, 0.0),
        transformation: Default::default(),
    };
    let fade = nested_fade(2.0);
    let mut target = Target::new();
    let frame = target.draw(stripes);
    let drawn = target.draw(|scene| {
        stripes(scene);
        scene.push_layer_mask(fade.mask());
        scene.insert_primitive(quad(rect(30.0, 30.0, 40.0, 40.0), hsla(0.0, 1.0, 0.5, 1.0)));
        scene.insert_primitive(blur);
        scene.pop_layer_mask();
    });
    assert_pixels("blur inside an edge fade", &drawn, |x, y| {
        let layer = expected_blur(&frame, &blur, x, y);
        let expected = composite(frame.at(x, y), layer, fade.m(x, y));
        Some((expected, expected))
    });
}

/// Catches mask layers that stay allocated after
/// [`LAYER_IDLE_RELEASE_FRAMES`] frames without a mask, are released
/// sooner, keep counting idle frames across a masked frame, are allocated
/// by a frame without masks, or are not recreated for the next masked
/// frame.
#[test]
fn mask_layers_are_released_after_idle_frames_and_recreated_on_use() {
    let fade = nested_fade(1.0);
    let outline = nested_outline(1.0);
    let masked = |target: &mut Target| {
        draw_nested(
            target,
            LayerMask::Path(outline.clip_path()),
            fade.mask(),
            1.0,
        )
    };
    let layers = |target: &Target| target.renderer.layers.mask_layer_count();
    let mut target = Target::new();
    target.draw(backdrop);
    assert_eq!(layers(&target), 0, "a frame without masks allocated layers");
    // The masked frame of the second round restarts the idle count.
    for _ in 0..2 {
        masked(&mut target);
        assert_eq!(layers(&target), 2, "two nested masks use two layers");
        for idle in 1..LAYER_IDLE_RELEASE_FRAMES {
            target.draw(backdrop);
            assert_eq!(
                layers(&target),
                2,
                "layers released after {idle} idle frames"
            );
        }
    }
    target.draw(backdrop);
    assert_eq!(
        layers(&target),
        0,
        "layers kept after {LAYER_IDLE_RELEASE_FRAMES} idle frames"
    );
    let drawn = masked(&mut target);
    assert_eq!(layers(&target), 2, "layers not recreated after release");
    assert_pixels("nested masks after release", &drawn, |x, y| {
        Some(fade_inside_path(&fade, &outline, x, y))
    });
}

/// Catches a path clipped by the rectangle of its content mask without its
/// rounded corners, which paints the corners the mask cuts away, and a
/// path clipped by the corner radii of the mask applied to the bounds of
/// the path, which rounds the corners of a path smaller than its mask.
/// Does not check the antialiased band along the edge of the mask.
#[test]
fn paths_are_clipped_by_the_rounded_rectangle_of_their_content_mask() {
    let mask = ContentMask {
        bounds: rect(10.0, 10.0, 80.0, 80.0),
        corner_radii: Corners::all(ScaledPixels(20.0)),
    };
    let square = |lo: (f32, f32), hi: (f32, f32)| {
        Outline::new(lo)
            .line_to((hi.0, lo.1))
            .line_to(hi)
            .line_to((lo.0, hi.1))
    };
    let red = hsla(0.0, 1.0, 0.5, 1.0);
    let mut target = Target::new();
    for (name, outline) in [
        ("path inside", square((40.0, 40.0), (60.0, 60.0))),
        ("path across a corner", square((0.0, 0.0), (40.0, 40.0))),
        ("path over the frame", square((0.0, 0.0), (100.0, 100.0))),
    ] {
        let drawn = target.draw(|scene| {
            scene.insert_primitive(quad(rect(0.0, 0.0, 100.0, 100.0), black()));
            scene.insert_primitive(outline.path(red, mask));
        });
        assert_pixels(name, &drawn, |x, y| {
            let center = (x as f32 + 0.5, y as f32 + 0.5);
            let distance = quad_sdf(center, &mask.bounds, &mask.corner_radii);
            if distance.abs() < EDGE {
                return None;
            }
            let painted = distance < 0.0 && outline.coverage(x, y)? > 0.0;
            let color = rgba(if painted { red } else { black() });
            Some((color, color))
        });
    }
}
