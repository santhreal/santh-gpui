//! Image sprites drawn smaller, larger, and at the size of their tile, read
//! back from the wgpu renderer.
//!
//! WHY: the class closed here is "a minified image aliases". The atlas has
//! no mip levels and the sprite shader took one linear sample per pixel,
//! which reads the 2x2 texels nearest the pixel's center and none of the
//! others under it: an image drawn under half its size skipped texels
//! outright, and one between half and full size weighted them unevenly, so
//! text and hairlines in a capture shown to fit broke up. The case draws a
//! one-texel checkerboard, the finest detail an image holds, magnified, at
//! its own size, and minified with whole and fractional texel steps, from a
//! sub-pixel origin, and reads that every pixel inside the sprite is within
//! TOLERANCE of the mean of the texels under it, each weighed by its covered
//! area, over a footprint at least one texel wide: a box-filtered thumbnail
//! of the image when minified, linear interpolation when magnified. A second
//! case reads that an image drawn at its tile's size on whole pixels shows
//! its texels unblurred.
//!
//! Not covered: the Metal and Direct3D shaders, which take the same filter
//! and have no offscreen renderer here; rotated sprites; the pixels along a
//! sprite's edge, whose footprint the tile's edge cuts, which
//! `sprite_edge_tests` covers; minifications past the 31-texel footprint
//! cap.

use std::borrow::Cow;

use super::WgpuRenderer;
use crate::WgpuContext;
use gpui::{
    AtlasKey, AtlasTile, Bounds, ContentMask, Corners, DevicePixels, ImageId, PlatformAtlas,
    PolychromeSprite, RenderImageParams, ScaledPixels, Scene, TransformationMatrix, point, size,
};

/// The frame's width and height, in device pixels.
const EXTENT: i32 = 100;
/// The checkerboard's side, in texels.
const SIDE: i32 = 60;
/// How far a pixel may stray from the mean of the texels under it, in 8-bit
/// levels: the renderer's rounding of linear sample weights. One linear
/// sample per pixel strays 44 or more in every minified case below.
const TOLERANCE: f32 = 3.0;

/// Texel `j` of a row of the checkerboard: +1 white, -1 black.
fn sign(j: i32) -> f32 {
    if j.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

/// The mean of `sign` over texels `[from, to)` of a row, partial texels
/// weighted by their covered fraction.
fn mean_sign(from: f32, to: f32) -> f32 {
    let mut sum = 0.0;
    for j in from.floor() as i32..to.ceil() as i32 {
        let covered = (to.min(j as f32 + 1.0) - from.max(j as f32)).max(0.0);
        sum += sign(j) * covered;
    }
    sum / (to - from)
}

/// A `SIDE` x `SIDE` one-texel checkerboard of white and black, its top-left
/// texel white, inserted into the renderer's atlas.
fn checkerboard(renderer: &WgpuRenderer) -> AtlasTile {
    let key = AtlasKey::Image(RenderImageParams {
        image_id: ImageId(1),
        frame_index: 0,
    });
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| sign(x + y)))
        .flat_map(|s| {
            let level = if s > 0.0 { 255 } else { 0 };
            [level, level, level, 255]
        })
        .collect();
    renderer
        .sprite_atlas()
        .get_or_insert_with(&key, &mut || {
            Ok(Some((
                size(DevicePixels(SIDE), DevicePixels(SIDE)),
                Cow::Owned(texels.clone()),
            )))
        })
        .expect("the atlas allocates the tile")
        .expect("the tile has texels")
}

/// The pixels of `tile` drawn `side` device pixels square at `origin`.
fn draw(renderer: &mut WgpuRenderer, tile: AtlasTile, origin: (f32, f32), side: f32) -> Vec<u8> {
    let frame = Bounds {
        origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size: size(ScaledPixels(EXTENT as f32), ScaledPixels(EXTENT as f32)),
    };
    let mut scene = Scene::default();
    scene.insert_primitive(PolychromeSprite {
        order: 0,
        pad: 0,
        grayscale: false.into(),
        opacity: 1.0,
        bounds: Bounds {
            origin: point(ScaledPixels(origin.0), ScaledPixels(origin.1)),
            size: size(ScaledPixels(side), ScaledPixels(side)),
        },
        content_mask: ContentMask {
            bounds: frame,
            corner_radii: Corners::default(),
        },
        corner_radii: Corners::default(),
        tile,
        transformation: TransformationMatrix::unit(),
    });
    scene.finish();
    assert!(
        renderer.draw(&scene),
        "draw failed for a {side}-pixel sprite"
    );
    renderer.read_pixels().expect("read pixels")
}

fn renderer() -> WgpuRenderer {
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    WgpuRenderer::new_offscreen(&context, size(DevicePixels(EXTENT), DevicePixels(EXTENT)))
        .expect("offscreen renderer")
}

#[test]
fn a_scaled_image_shows_the_mean_of_the_texels_under_each_pixel() {
    let mut renderer = renderer();
    let tile = checkerboard(&renderer);
    let (x, y) = (4.25, 4.37);
    for texels_per_pixel in [0.65_f32, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.65, 5.5] {
        let side = SIDE as f32 / texels_per_pixel;
        let pixels = draw(&mut renderer, tile, (x, y), side);
        // The footprint of pixel `p` along an axis whose sprite starts at
        // `start`, in texels: at least one texel wide, so a magnified image
        // interpolates linearly between texels.
        let width = texels_per_pixel.max(1.0);
        let footprint = |p: usize, start: f32| {
            let center = (p as f32 + 0.5 - start) * texels_per_pixel;
            (center - width / 2.0, center + width / 2.0)
        };
        let inside = |start: f32| start.ceil() as usize + 1..(start + side).floor() as usize - 1;
        let wrong: Vec<_> = inside(y)
            .flat_map(|py| inside(x).map(move |px| (px, py)))
            .filter_map(|(px, py)| {
                let (x0, x1) = footprint(px, x);
                let (y0, y1) = footprint(py, y);
                let mean = 127.5 + 127.5 * mean_sign(x0, x1) * mean_sign(y0, y1);
                let at = (py * EXTENT as usize + px) * 4;
                let texel: [u8; 4] = pixels[at..at + 4].try_into().unwrap();
                let off = texel[..3]
                    .iter()
                    .any(|&c| (c as f32 - mean).abs() > TOLERANCE)
                    || texel[3] != 255;
                off.then_some((px, py, mean.round(), texel))
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "a one-texel checkerboard drawn at 1/{texels_per_pixel} its size shows {} pixels \
             more than {TOLERANCE} off the mean of the texels under them, first \
             (x, y, mean, rgba): {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(6)],
        );
    }
}

#[test]
fn an_image_at_its_tiles_size_shows_its_texels_unblurred() {
    let mut renderer = renderer();
    let tile = checkerboard(&renderer);
    let pixels = draw(&mut renderer, tile, (4.0, 4.0), SIDE as f32);
    let wrong: Vec<_> = (4..4 + SIDE as usize)
        .flat_map(|py| (4..4 + SIDE as usize).map(move |px| (px, py)))
        .filter_map(|(px, py)| {
            let level = if sign(px as i32 + py as i32) > 0.0 {
                255
            } else {
                0
            };
            let at = (py * EXTENT as usize + px) * 4;
            let texel: [u8; 4] = pixels[at..at + 4].try_into().unwrap();
            let off = texel
                .iter()
                .zip([level, level, level, 255])
                .any(|(&a, b)| a.abs_diff(b) > 1);
            off.then_some((px, py, texel))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "a checkerboard drawn at its own size on whole pixels shows {} blurred pixels, first \
         (x, y, rgba): {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(6)],
    );
}
