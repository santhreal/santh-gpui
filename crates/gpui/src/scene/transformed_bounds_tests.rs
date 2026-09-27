//! A primitive is culled and ordered by its bounds in window space, after its
//! transformation.
//!
//! Catches `Scene::insert_primitive` reading a primitive's local `bounds` as
//! window space: a primitive transformed into the content mask dropped as
//! empty, one transformed out of it kept, and one transformed over an earlier
//! primitive batched under it. Covers every primitive kind with a
//! transformation; a kind is stored only through `Stored`, whose
//! `clipped_bounds` each kind implements, so a new kind does not compile until
//! it states its window-space bounds. Does not catch a renderer that draws a
//! primitive at the wrong window position.

use super::*;
use crate::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, BorderStyle, TileId, TransformationMatrix, point,
    px, size,
};

fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(w), ScaledPixels(h)),
    }
}

fn mask() -> ContentMask<ScaledPixels> {
    rect(0.0, 0.0, 1000.0, 1000.0).into()
}

fn translation(x: f32, y: f32) -> TransformationMatrix {
    TransformationMatrix {
        rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
        translation: [x, y],
    }
}

fn tile(kind: AtlasTextureKind) -> AtlasTile {
    AtlasTile {
        texture_id: AtlasTextureId { index: 0, kind },
        tile_id: TileId(0),
        padding: 0,
        bounds: Bounds::default(),
    }
}

/// One primitive of every kind with a transformation, with local bounds
/// `bounds`.
fn every_transformable_kind(
    bounds: Bounds<ScaledPixels>,
    transformation: TransformationMatrix,
) -> Vec<Primitive> {
    let mut path = Path::new(point(px(bounds.left().0), px(bounds.top().0)));
    path.line_to(point(px(bounds.right().0), px(bounds.top().0)));
    path.line_to(point(px(bounds.right().0), px(bounds.bottom().0)));
    path.line_to(point(px(bounds.left().0), px(bounds.bottom().0)));
    let mut path = path.scale(1.0);
    path.content_mask = mask();
    path.transformation = transformation;
    vec![
        Primitive::Quad(Quad {
            border_style: BorderStyle::default(),
            bounds,
            content_mask: mask(),
            transformation,
            ..Quad::default()
        }),
        Primitive::Shadow(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.0),
            bounds,
            corner_radii: Corners::default(),
            content_mask: mask(),
            color: Hsla::default(),
            element_bounds: bounds,
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
            transformation,
        }),
        Primitive::Path(path),
        Primitive::Underline(Underline {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            thickness: ScaledPixels(1.0),
            wavy: false.into(),
            transformation,
        }),
        Primitive::MonochromeSprite(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            tile: tile(AtlasTextureKind::Monochrome),
            transformation,
        }),
        Primitive::SubpixelSprite(SubpixelSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            tile: tile(AtlasTextureKind::Subpixel),
            transformation,
        }),
        Primitive::PolychromeSprite(PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.0,
            bounds,
            content_mask: mask(),
            corner_radii: Corners::default(),
            tile: tile(AtlasTextureKind::Polychrome),
            transformation,
        }),
        Primitive::BackdropBlur(BackdropBlur {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            corner_radii: Corners::default(),
            blur_radius: ScaledPixels(4.0),
            saturation: 1.0,
            tint: Hsla::default(),
            transformation,
        }),
    ]
}

fn batch_count(primitive: Primitive) -> usize {
    let mut scene = Scene::default();
    scene.insert_primitive(primitive);
    scene.finish();
    scene.batches().count()
}

fn kind(primitive: &Primitive) -> PrimitiveKind {
    match primitive {
        Primitive::Shadow(_) => PrimitiveKind::Shadow,
        Primitive::Quad(_) => PrimitiveKind::Quad,
        Primitive::Path(_) => PrimitiveKind::Path,
        Primitive::Underline(_) => PrimitiveKind::Underline,
        Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
        Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
        Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
        Primitive::Surface(_) => PrimitiveKind::Surface,
        Primitive::BackdropBlur(_) => PrimitiveKind::BackdropBlur,
    }
}

#[test]
fn a_primitive_transformed_into_the_content_mask_is_kept() {
    for primitive in
        every_transformable_kind(rect(-300.0, -300.0, 50.0, 50.0), translation(400.0, 400.0))
    {
        let kind = kind(&primitive);
        assert_eq!(batch_count(primitive), 1, "dropped {kind:?}");
    }
}

#[test]
fn a_primitive_transformed_out_of_the_content_mask_is_culled() {
    for primitive in
        every_transformable_kind(rect(100.0, 100.0, 50.0, 50.0), translation(2000.0, 0.0))
    {
        let kind = kind(&primitive);
        assert_eq!(batch_count(primitive), 0, "kept {kind:?}");
    }
}

/// Quads batch before surfaces at equal orders, so a quad that overlaps an
/// earlier surface only after its transformation must still get the higher
/// order and draw above it.
#[test]
fn a_primitive_transformed_over_an_earlier_one_draws_above_it() {
    for primitive in
        every_transformable_kind(rect(10.0, 10.0, 50.0, 50.0), translation(500.0, 500.0))
    {
        let kind = kind(&primitive);
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds: rect(500.0, 500.0, 100.0, 100.0),
            content_mask: mask(),
            #[cfg(target_os = "macos")]
            image_buffer: test_image_buffer(),
        });
        scene.insert_primitive(primitive);
        scene.finish();
        let first = scene.batches().next();
        assert!(
            matches!(first, Some(PrimitiveBatch::Surfaces(_))),
            "{kind:?} drew under the surface it covers: first batch {first:?}"
        );
    }
}
