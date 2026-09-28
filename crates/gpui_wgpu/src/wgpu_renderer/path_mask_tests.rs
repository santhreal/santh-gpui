//! Paths clipped by a rounded content mask on the wgpu renderer, checked
//! pixel by pixel at 1x and 2x scale.
//!
//! A path is clipped by the rounded rectangle of its content mask, whatever
//! the bounds of the path. Catches the corner radii of the mask applied to
//! the bounds of the path, or to those bounds clipped to the mask, which
//! rounds the corners of every path smaller than its mask, such as an icon
//! in a window with rounded corners; and corner radii dropped, which paints
//! a path in the corners the mask cuts away. Catches a gradient on a path
//! that spans the content mask instead of the bounds of the path. Does not
//! check the antialiased band along the edge of the mask, where the GPU and
//! this model may round differently.

use super::WgpuRenderer;
use crate::WgpuContext;
use gpui::{
    Background, Bounds, ContentMask, Corners, DevicePixels, Path, Point, ScaledPixels, Scene,
    linear_color_stop, linear_gradient, px, rgb, size, solid_background,
};

/// The logical width and height of every test frame.
const EXTENT: f32 = 100.0;

/// The rounded content mask of the cases, in logical pixels.
const MASK: [f32; 4] = [10.0, 10.0, 80.0, 80.0];
const RADIUS: f32 = 20.0;

/// The opaque paint of solid paths, as premultiplied RGBA.
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// Pixels whose center lies closer than this to the edge of the mask, in
/// device pixels, are not checked.
const EDGE_BAND: f32 = 1.0;

fn bounds([x, y, width, height]: [f32; 4], scale: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        Point::new(ScaledPixels(x * scale), ScaledPixels(y * scale)),
        size(ScaledPixels(width * scale), ScaledPixels(height * scale)),
    )
}

fn rounded_mask(scale: f32) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds(MASK, scale),
        corner_radii: Corners::all(ScaledPixels(RADIUS * scale)),
    }
}

fn frame_mask(scale: f32) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: bounds([0.0, 0.0, EXTENT, EXTENT], scale),
        corner_radii: Corners::default(),
    }
}

/// A rectangle path over `area`, in logical pixels, filled with `color`
/// and clipped by `mask`.
fn rect_path(
    [x, y, width, height]: [f32; 4],
    color: Background,
    mask: ContentMask<ScaledPixels>,
    scale: f32,
) -> Path<ScaledPixels> {
    let corner = |x: f32, y: f32| Point::new(px(x), px(y));
    let mut path = Path::new(corner(x, y));
    path.line_to(corner(x + width, y));
    path.line_to(corner(x + width, y + height));
    path.line_to(corner(x, y + height));
    path.line_to(corner(x, y));
    path.color = color;
    let mut path = path.scale(scale);
    path.content_mask = mask;
    path
}

fn renderer(scale: f32) -> WgpuRenderer {
    let extent = (EXTENT * scale) as i32;
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    WgpuRenderer::new_offscreen(&context, size(DevicePixels(extent), DevicePixels(extent)))
        .expect("offscreen renderer")
}

/// Draws `path` alone and returns the RGBA bytes of the frame.
fn draw(renderer: &mut WgpuRenderer, path: Path<ScaledPixels>) -> Vec<u8> {
    let mut scene = Scene::default();
    scene.insert_primitive(path);
    scene.finish();
    assert!(renderer.draw(&scene), "draw failed");
    renderer.read_pixels().expect("read pixels")
}

/// The signed distance from the device pixel center `center` to the edge
/// of the rounded mask, negative inside.
fn mask_distance(center: (f32, f32), scale: f32) -> f32 {
    let [x, y, width, height] = MASK.map(|value| value * scale);
    let radius = RADIUS * scale;
    let corner = (
        (center.0 - (x + width / 2.0)).abs() - width / 2.0 + radius,
        (center.1 - (y + height / 2.0)).abs() - height / 2.0 + radius,
    );
    corner.0.max(0.0).hypot(corner.1.max(0.0)) + corner.0.max(corner.1).min(0.0) - radius
}

#[test]
fn paths_are_clipped_by_the_rounded_rectangle_of_their_content_mask() {
    // An area well inside the mask, one across its top left corner, and
    // one over the whole frame.
    let cases = [
        ("inside", [40.0, 40.0, 20.0, 20.0]),
        ("across a corner", [0.0, 0.0, 40.0, 40.0]),
        ("over the frame", [0.0, 0.0, EXTENT, EXTENT]),
    ];
    for scale in [1.0, 2.0] {
        let mut renderer = renderer(scale);
        let extent = (EXTENT * scale) as usize;
        for (name, area) in cases {
            let paint = solid_background(rgb(0xff0000));
            let bytes = draw(
                &mut renderer,
                rect_path(area, paint, rounded_mask(scale), scale),
            );
            let area = bounds(area, scale);
            let mut wrong = Vec::new();
            for y in 0..extent {
                for x in 0..extent {
                    let center = (x as f32 + 0.5, y as f32 + 0.5);
                    let distance = mask_distance(center, scale);
                    if distance.abs() < EDGE_BAND {
                        continue;
                    }
                    let painted = distance < 0.0
                        && center.0 > area.left().0
                        && center.0 < area.right().0
                        && center.1 > area.top().0
                        && center.1 < area.bottom().0;
                    let expected = if painted { RED } else { [0.0; 4] };
                    let offset = (y * extent + x) * 4;
                    let actual = &bytes[offset..offset + 4];
                    if actual
                        .iter()
                        .zip(expected)
                        .any(|(actual, expected)| (*actual as f32 - expected * 255.0).abs() > 1.5)
                    {
                        wrong.push((x, y, actual.to_vec(), painted));
                    }
                }
            }
            assert!(
                wrong.is_empty(),
                "path {name} at scale {scale}: {} pixels differ, first (x, y, actual, painted): {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(8)],
            );
        }
    }
}

#[test]
fn a_gradient_on_a_path_spans_the_path_whatever_its_content_mask() {
    let area = [40.0, 40.0, 20.0, 20.0];
    let gradient = linear_gradient(
        90.0,
        linear_color_stop(rgb(0xff0000), 0.0),
        linear_color_stop(rgb(0x0000ff), 1.0),
    );
    for scale in [1.0, 2.0] {
        let mut renderer = renderer(scale);
        let in_frame = draw(
            &mut renderer,
            rect_path(area, gradient, frame_mask(scale), scale),
        );
        let in_mask = draw(
            &mut renderer,
            rect_path(area, gradient, rounded_mask(scale), scale),
        );
        let extent = (EXTENT * scale) as usize;
        let row = (50.0 * scale) as usize * extent * 4;
        let first = row + (40.0 * scale) as usize * 4;
        let last = row + ((60.0 * scale) as usize - 1) * 4;
        assert!(
            in_frame[first] > 200 && in_frame[last + 2] > 200,
            "at scale {scale} the gradient runs from red to blue across the path: first {:?}, last {:?}",
            &in_frame[first..first + 4],
            &in_frame[last..last + 4],
        );
        assert!(
            in_frame == in_mask,
            "at scale {scale} a gradient path draws the same inside a rounded mask as inside the frame",
        );
    }
}
