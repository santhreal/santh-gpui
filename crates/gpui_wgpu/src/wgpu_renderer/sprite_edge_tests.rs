//! Image sprites drawn at a size other than their tile's, read back from
//! the wgpu renderer.
//!
//! WHY: the class closed here is "a scaled image takes in texels from
//! beside its tile". The atlas packs image tiles with no gutter, and the
//! sprite shader sampled with a linear filter up to the tile's edge, so an
//! image drawn larger than its tile blended its last row and column with
//! the atlas texels past them: a neighboring tile's, or the transparent
//! texels of unused atlas space. The case puts a light tile and a dark one
//! into the atlas, draws each magnified at a scale with fractional texel
//! steps and at a whole one, and reads that every pixel wholly inside a
//! drawn sprite has its own tile's color.
//!
//! Not covered: the Metal and Direct3D shaders, which take the same clamp
//! and have no offscreen renderer here; minified sprites, whose pixels
//! wholly inside sample no nearer an edge than half a texel; monochrome and
//! subpixel sprites, which draw at their tile's size.

use std::borrow::Cow;

use super::WgpuRenderer;
use crate::WgpuContext;
use gpui::{
    AtlasKey, AtlasTile, Bounds, ContentMask, Corners, DevicePixels, ImageId, PlatformAtlas,
    PolychromeSprite, RenderImageParams, ScaledPixels, Scene, TransformationMatrix, point, size,
};

/// The frame's width and height, in device pixels.
const EXTENT: i32 = 100;
/// The tiles' side, in texels.
const SIDE: i32 = 10;
/// The tiles' gray levels, far apart.
const LIGHT: u8 = 200;
const DARK: u8 = 20;

/// An opaque `SIDE` x `SIDE` image tile of gray `level`, inserted into the
/// renderer's atlas under `id`.
fn tile(renderer: &WgpuRenderer, id: usize, level: u8) -> AtlasTile {
    let key = AtlasKey::Image(RenderImageParams {
        image_id: ImageId(id),
        frame_index: 0,
    });
    let texels = [level, level, level, 255].repeat((SIDE * SIDE) as usize);
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

fn sprite(tile: AtlasTile, bounds: Bounds<ScaledPixels>) -> PolychromeSprite {
    let frame = Bounds {
        origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size: size(ScaledPixels(EXTENT as f32), ScaledPixels(EXTENT as f32)),
    };
    PolychromeSprite {
        order: 0,
        pad: 0,
        grayscale: false.into(),
        opacity: 1.0,
        bounds,
        content_mask: ContentMask {
            bounds: frame,
            corner_radii: Corners::default(),
        },
        corner_radii: Corners::default(),
        tile,
        transformation: TransformationMatrix::unit(),
    }
}

#[test]
fn a_magnified_image_shows_only_its_own_tile() {
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    let mut renderer =
        WgpuRenderer::new_offscreen(&context, size(DevicePixels(EXTENT), DevicePixels(EXTENT)))
            .expect("offscreen renderer");
    let tiles = [
        (tile(&renderer, 1, LIGHT), LIGHT),
        (tile(&renderer, 2, DARK), DARK),
    ];
    for scale in [1.45, 3.0] {
        let side = SIDE as f32 * scale;
        let drawn: Vec<_> = tiles
            .iter()
            .enumerate()
            .map(|(i, &(tile, level))| {
                let origin = (4.0 + i as f32 * (side + 8.0), 4.0);
                (origin, level, tile)
            })
            .collect();
        let mut scene = Scene::default();
        for &((x, y), _, tile) in &drawn {
            scene.insert_primitive(sprite(
                tile,
                Bounds {
                    origin: point(ScaledPixels(x), ScaledPixels(y)),
                    size: size(ScaledPixels(side), ScaledPixels(side)),
                },
            ));
        }
        scene.finish();
        assert!(renderer.draw(&scene), "draw failed at scale {scale}");
        let pixels = renderer.read_pixels().expect("read pixels");
        for ((x, y), level, _) in drawn {
            let (left, top) = (x.ceil() as usize, y.ceil() as usize);
            let (right, bottom) = ((x + side).floor() as usize, (y + side).floor() as usize);
            let wrong: Vec<_> = (top..bottom)
                .flat_map(|py| (left..right).map(move |px| (px, py)))
                .filter_map(|(px, py)| {
                    let at = (py * EXTENT as usize + px) * 4;
                    let texel: [u8; 4] = pixels[at..at + 4].try_into().unwrap();
                    let gray = [level, level, level, 255];
                    let off = texel.iter().zip(gray).any(|(&a, b)| a.abs_diff(b) > 1);
                    off.then_some((px, py, texel))
                })
                .collect();
            assert!(
                wrong.is_empty(),
                "the gray {level} tile drawn {scale} times its size at ({x}, {y}) shows {} \
                 pixels of another color, first (x, y, rgba): {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(6)],
            );
        }
    }
}
