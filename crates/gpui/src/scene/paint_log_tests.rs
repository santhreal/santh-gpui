//! The paint log holds a reference to each primitive rather than a copy, and
//! `Scene::replay` resolves those references against the previous frame's
//! scene, whose vectors `Scene::finish` has already sorted into draw order.
//!
//! Catches a reference resolving to the wrong primitive: an index left
//! pointing at the position a primitive was inserted at after `finish` moved
//! it, a partial range replaying more or less than asked, a primitive clipped
//! away to nothing reaching the log, a replayed frame or layer mask that
//! differs from the one it replays, a `finish` that breaks ties in a
//! different order than a stable sort by draw order (and by atlas tile for
//! sprites), and a log entry that holds a primitive instead of a reference.
//! Covers every primitive kind: `next_kind` and `sample` are exhaustive
//! matches over `PrimitiveKind`, so a new kind does not compile until it has
//! a sample. Does not catch a renderer that draws a correct scene wrongly.

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
    rect(0.0, 0.0, 10_000.0, 10_000.0).into()
}

fn tile(kind: AtlasTextureKind, id: u32) -> AtlasTile {
    AtlasTile {
        texture_id: AtlasTextureId { index: 0, kind },
        tile_id: TileId(id),
        padding: 0,
        bounds: Bounds::default(),
    }
}

/// The kind after `kind` in declaration order.
fn next_kind(kind: PrimitiveKind) -> Option<PrimitiveKind> {
    match kind {
        PrimitiveKind::StartLayerMask => Some(PrimitiveKind::Shadow),
        PrimitiveKind::Shadow => Some(PrimitiveKind::Quad),
        PrimitiveKind::Quad => Some(PrimitiveKind::Path),
        PrimitiveKind::Path => Some(PrimitiveKind::Underline),
        PrimitiveKind::Underline => Some(PrimitiveKind::MonochromeSprite),
        PrimitiveKind::MonochromeSprite => Some(PrimitiveKind::SubpixelSprite),
        PrimitiveKind::SubpixelSprite => Some(PrimitiveKind::PolychromeSprite),
        PrimitiveKind::PolychromeSprite => Some(PrimitiveKind::Surface),
        PrimitiveKind::Surface => Some(PrimitiveKind::BackdropBlur),
        PrimitiveKind::BackdropBlur => Some(PrimitiveKind::EndLayerMask),
        PrimitiveKind::EndLayerMask => None,
    }
}

/// Every kind but the layer mask markers.
fn drawn_kinds() -> Vec<PrimitiveKind> {
    std::iter::successors(Some(PrimitiveKind::StartLayerMask), |kind| next_kind(*kind))
        .filter(|kind| sample(*kind, rect(0.0, 0.0, 1.0, 1.0), 0).is_some())
        .collect()
}

/// A primitive of `kind` over `bounds`, on atlas tile `tile_id` when it is a
/// sprite. `None` for the layer mask markers.
fn sample(kind: PrimitiveKind, bounds: Bounds<ScaledPixels>, tile_id: u32) -> Option<Primitive> {
    let transformation = TransformationMatrix::unit();
    Some(match kind {
        PrimitiveKind::StartLayerMask | PrimitiveKind::EndLayerMask => return None,
        PrimitiveKind::Shadow => Primitive::Shadow(Shadow {
            order: 0,
            blur_radius: ScaledPixels(2.0),
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
        PrimitiveKind::Quad => Primitive::Quad(Quad {
            border_style: BorderStyle::default(),
            bounds,
            content_mask: mask(),
            ..Quad::default()
        }),
        PrimitiveKind::Path => {
            let mut path = Path::new(point(px(bounds.left().0), px(bounds.top().0)));
            path.line_to(point(px(bounds.right().0), px(bounds.top().0)));
            path.line_to(point(px(bounds.right().0), px(bounds.bottom().0)));
            path.line_to(point(px(bounds.left().0), px(bounds.bottom().0)));
            let mut path = path.scale(1.0);
            path.content_mask = mask();
            Primitive::Path(path)
        }
        PrimitiveKind::Underline => Primitive::Underline(Underline {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            thickness: ScaledPixels(1.0),
            wavy: false.into(),
            transformation,
        }),
        PrimitiveKind::MonochromeSprite => Primitive::MonochromeSprite(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            tile: tile(AtlasTextureKind::Monochrome, tile_id),
            transformation,
        }),
        PrimitiveKind::SubpixelSprite => Primitive::SubpixelSprite(SubpixelSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: mask(),
            color: Hsla::default(),
            tile: tile(AtlasTextureKind::Subpixel, tile_id),
            transformation,
        }),
        PrimitiveKind::PolychromeSprite => Primitive::PolychromeSprite(PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.0,
            bounds,
            content_mask: mask(),
            corner_radii: Corners::default(),
            tile: tile(AtlasTextureKind::Polychrome, tile_id),
            transformation,
        }),
        PrimitiveKind::Surface => Primitive::Surface(PaintSurface {
            order: 0,
            bounds,
            content_mask: mask(),
            #[cfg(target_os = "macos")]
            image_buffer: test_image_buffer(),
        }),
        PrimitiveKind::BackdropBlur => Primitive::BackdropBlur(BackdropBlur {
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
    })
}

/// The left edge of every primitive of `kind`, in the order its vector holds
/// them. Each test places its primitives at distinct left edges.
fn vector_markers(scene: &Scene, kind: PrimitiveKind) -> Vec<f32> {
    fn left<'a>(bounds: impl Iterator<Item = &'a Bounds<ScaledPixels>>) -> Vec<f32> {
        bounds.map(|bounds| bounds.origin.x.0).collect()
    }
    match kind {
        PrimitiveKind::Shadow => left(scene.shadows.iter().map(|p| &p.bounds)),
        PrimitiveKind::Quad => left(scene.quads.iter().map(|p| &p.bounds)),
        PrimitiveKind::Path => left(scene.paths.iter().map(|p| &p.bounds)),
        PrimitiveKind::Underline => left(scene.underlines.iter().map(|p| &p.bounds)),
        PrimitiveKind::MonochromeSprite => left(scene.monochrome_sprites.iter().map(|p| &p.bounds)),
        PrimitiveKind::SubpixelSprite => left(scene.subpixel_sprites.iter().map(|p| &p.bounds)),
        PrimitiveKind::PolychromeSprite => left(scene.polychrome_sprites.iter().map(|p| &p.bounds)),
        PrimitiveKind::Surface => left(scene.surfaces.iter().map(|p| &p.bounds)),
        PrimitiveKind::BackdropBlur => left(scene.backdrop_blurs.iter().map(|p| &p.bounds)),
        PrimitiveKind::StartLayerMask | PrimitiveKind::EndLayerMask => Vec::new(),
    }
}

/// The kind and left edge of the primitive each logged reference resolves
/// to, in log order, leaving out layer mask markers.
fn trace(scene: &Scene) -> Vec<(PrimitiveKind, f32)> {
    scene
        .paint_operations
        .iter()
        .filter_map(|operation| match operation {
            PaintOperation::Primitive(PrimitiveRef {
                kind: PrimitiveKind::StartLayerMask | PrimitiveKind::EndLayerMask,
                ..
            }) => None,
            PaintOperation::Primitive(reference) => Some((
                reference.kind,
                vector_markers(scene, reference.kind)[reference.index as usize],
            )),
            _ => None,
        })
        .collect()
}

/// Every vector of `scene`, printed.
fn snapshot(scene: &Scene) -> String {
    format!(
        "{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}",
        scene.shadows,
        scene.quads,
        scene.paths,
        scene.underlines,
        scene.monochrome_sprites,
        scene.subpixel_sprites,
        scene.polychrome_sprites,
        scene.surfaces,
        scene.backdrop_blurs,
        scene.start_layer_masks,
        scene.end_layer_masks,
    )
}

/// Paints two primitives of every logged kind, one row per kind starting at
/// `y`: the first at z-index 1, then the second at z-index 0. `finish` puts
/// the second ahead of the first in every vector. Returns what `trace`
/// reports for them.
fn paint_pairs(scene: &mut Scene, x: f32, y: f32) -> Vec<(PrimitiveKind, f32)> {
    let mut painted = Vec::new();
    for (row, kind) in drawn_kinds().into_iter().enumerate() {
        let top = y + row as f32 * 20.0;
        let first = sample(kind, rect(x, top, 8.0, 8.0), 1).unwrap();
        let second = sample(kind, rect(x + 1.0, top, 8.0, 8.0), 1).unwrap();
        scene.push_z_index(1);
        scene.insert_primitive(first);
        scene.pop_z_index();
        scene.insert_primitive(second);
        painted.push((kind, x));
        painted.push((kind, x + 1.0));
    }
    painted
}

/// Pairs at the root, then pairs inside a layer inside an edge fade.
fn frame() -> (Scene, Vec<(PrimitiveKind, f32)>) {
    let mut scene = Scene::default();
    let mut painted = paint_pairs(&mut scene, 10.0, 0.0);
    let region = rect(0.0, 0.0, 1000.0, 1000.0);
    scene.push_layer_mask(LayerMask::EdgeFade(EdgeFadeMask::new(
        region,
        Edges::all(ScaledPixels(16.0)),
        region,
    )));
    scene.push_layer(region);
    painted.extend(paint_pairs(&mut scene, 100.0, 300.0));
    scene.pop_layer();
    scene.pop_layer_mask();
    (scene, painted)
}

#[test]
fn every_reference_resolves_to_its_primitive_before_and_after_finish() {
    let (mut scene, painted) = frame();
    assert_eq!(trace(&scene), painted);

    scene.finish();

    // The root's z-index 1 lifts its primitive above the layer painted after
    // it; inside the layer, z-index 1 orders only among the layer's children.
    for kind in drawn_kinds() {
        assert_eq!(
            vector_markers(&scene, kind),
            vec![11.0, 101.0, 100.0, 10.0],
            "finish did not reorder the {kind:?} vector, so this test proves nothing for it",
        );
    }
    assert_eq!(trace(&scene), painted);
}

#[test]
fn replaying_a_finished_frame_reproduces_it() {
    let (mut source, painted) = frame();
    source.finish();

    let mut replayed = Scene::default();
    replayed.replay(0..source.len(), &source);
    assert_eq!(trace(&replayed), painted);
    replayed.finish();

    assert_eq!(replayed.len(), source.len());
    assert_eq!(snapshot(&replayed), snapshot(&source));
    assert_eq!(trace(&replayed), painted);
}

/// Two sibling edge fades, the first painted at z-index 1, so `finish` swaps
/// them. Replaying the finished frame has to push each mask where the log
/// recorded it, with the bands it was painted with.
#[test]
fn replaying_a_finished_frame_reproduces_its_layer_masks() {
    let mut source = Scene::default();
    let region = rect(0.0, 0.0, 100.0, 100.0);
    for (z_index, band) in [(1, 4.0), (0, 8.0)] {
        source.push_z_index(z_index);
        source.push_layer_mask(LayerMask::EdgeFade(EdgeFadeMask::new(
            region,
            Edges::all(ScaledPixels(band)),
            region,
        )));
        source.insert_primitive(sample(PrimitiveKind::Quad, rect(band, 0.0, 8.0, 8.0), 0).unwrap());
        source.pop_layer_mask();
        source.pop_z_index();
    }
    source.finish();
    let bands = |scene: &Scene| {
        scene
            .start_layer_masks
            .iter()
            .map(|start| match &start.mask {
                LayerMask::EdgeFade(fade) => fade.bands.top.0,
                LayerMask::Path(_) => f32::NAN,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bands(&source),
        vec![8.0, 4.0],
        "finish did not reorder the masks"
    );

    let mut replayed = Scene::default();
    replayed.replay(0..source.len(), &source);
    replayed.finish();

    assert_eq!(bands(&replayed), vec![8.0, 4.0]);
    assert_eq!(snapshot(&replayed), snapshot(&source));
}

#[test]
fn replaying_part_of_a_finished_frame_takes_only_that_part() {
    let (mut source, _) = frame();
    source.finish();

    // Each kind's pair is four operations at the start of the log: push the
    // z-index, the first primitive, pop the z-index, the second primitive.
    for (row, kind) in drawn_kinds().into_iter().enumerate() {
        let mut replayed = Scene::default();
        replayed.replay(row * 4..row * 4 + 4, &source);
        assert_eq!(trace(&replayed), vec![(kind, 10.0), (kind, 11.0)]);
        replayed.finish();
        for other in drawn_kinds() {
            let expected = if other == kind {
                vec![11.0, 10.0]
            } else {
                vec![]
            };
            assert_eq!(vector_markers(&replayed, other), expected, "{kind:?}");
        }
    }
}

#[test]
fn a_primitive_clipped_to_nothing_never_reaches_the_log() {
    for kind in drawn_kinds() {
        let mut scene = Scene::default();
        scene.insert_primitive(sample(kind, rect(20_000.0, 0.0, 8.0, 8.0), 0).unwrap());
        assert_eq!(scene.len(), 0, "{kind:?}");
        assert_eq!(vector_markers(&scene, kind), Vec::<f32>::new(), "{kind:?}");
    }
}

/// `finish` orders primitives of one kind by draw order, sprites then by
/// atlas tile, and otherwise keeps paint order. The primitives at z-index 1
/// alternate with those at z-index 0 in paint order, so the sort has to move
/// primitives past others with an equal draw order.
#[test]
fn finish_breaks_ties_by_tile_then_by_paint_order() {
    const PAIRS: u32 = 64;
    for kind in drawn_kinds() {
        let sprite = matches!(
            kind,
            PrimitiveKind::MonochromeSprite
                | PrimitiveKind::SubpixelSprite
                | PrimitiveKind::PolychromeSprite
        );
        let mut scene = Scene::default();
        for pair in 0..PAIRS {
            let x = pair as f32 * 20.0;
            // Descending tiles, so a sprite sort that ignores the tile keeps
            // paint order and fails below.
            let tile_id = PAIRS - pair;
            scene.push_z_index(1);
            scene.insert_primitive(sample(kind, rect(x, 0.0, 8.0, 8.0), tile_id).unwrap());
            scene.pop_z_index();
            scene.insert_primitive(sample(kind, rect(x + 10.0, 0.0, 8.0, 8.0), tile_id).unwrap());
        }
        scene.finish();

        let lower = (0..PAIRS).map(|pair| pair as f32 * 20.0 + 10.0);
        let upper = (0..PAIRS).map(|pair| pair as f32 * 20.0);
        let expected: Vec<f32> = if sprite {
            lower.rev().chain(upper.rev()).collect()
        } else {
            lower.chain(upper).collect()
        };
        assert_eq!(vector_markers(&scene, kind), expected, "{kind:?}");
    }
}

/// A log entry is a reference or a layer bound, never a primitive: the
/// smallest primitive is several times this size.
#[test]
fn a_log_entry_holds_no_primitive() {
    assert!(
        size_of::<PaintOperation>() <= 20,
        "a paint operation is {} bytes",
        size_of::<PaintOperation>()
    );
}
