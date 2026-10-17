//! A layer mask brackets its subtree in draw order and replays unchanged,
//! for every mask kind, and an edge fade covers the region its fade ratio
//! can be nonzero in.
//!
//! Catches a mask marker keyed so that a child sorts outside its mask (a
//! child at a higher z-index, or a child of an outer mask after an inner mask
//! closed), which draws the child unmasked; a scope beside a mask keyed so
//! that its primitives sort between the mask's markers (a layer, a layer in a
//! layer, or a second mask, painted before or after the mask and not
//! overlapping it), which draws them into the mask's layer and composites
//! them away outside the mask; a replay that records a mask's scope twice,
//! which nests one more layer per replayed frame; a replay that drops or
//! alters the mask payload; and edge fade bounds that cut off content beyond
//! an unfaded edge or keep content beyond a faded edge. Does not catch a
//! renderer that composites a correctly ordered mask wrongly.

use super::*;
use crate::{point, px, size};

fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(w), ScaledPixels(h)),
    }
}

fn edges(top: f32, right: f32, bottom: f32, left: f32) -> Edges<ScaledPixels> {
    Edges {
        top: ScaledPixels(top),
        right: ScaledPixels(right),
        bottom: ScaledPixels(bottom),
        left: ScaledPixels(left),
    }
}

fn quad(bounds: Bounds<ScaledPixels>) -> Quad {
    Quad {
        bounds,
        content_mask: rect(0.0, 0.0, 1000.0, 1000.0).into(),
        ..Quad::default()
    }
}

/// A path mask covering the square from `lo` to `hi`.
fn path_square(lo: f32, hi: f32) -> LayerMask {
    let mut path = Path::new(point(px(lo), px(lo)));
    path.line_to(point(px(hi), px(lo)));
    path.line_to(point(px(hi), px(hi)));
    path.line_to(point(px(lo), px(hi)));
    let mut path = path.scale(1.0);
    path.content_mask = rect(0.0, 0.0, 1000.0, 1000.0).into();
    LayerMask::Path(path)
}

/// An edge fade mask covering the square from `lo` to `hi`, faded at every
/// edge.
fn fade_square(lo: f32, hi: f32) -> LayerMask {
    LayerMask::EdgeFade(EdgeFadeMask::new(
        rect(lo, lo, hi - lo, hi - lo),
        edges(4.0, 4.0, 4.0, 4.0),
        rect(0.0, 0.0, 1000.0, 1000.0),
    ))
}

/// Every mask kind, as a constructor of the square from `lo` to `hi`. The
/// exhaustive match fails to compile when a kind is added without a case.
fn mask_kinds() -> Vec<(&'static str, fn(f32, f32) -> LayerMask)> {
    let kinds: Vec<(&'static str, fn(f32, f32) -> LayerMask)> =
        vec![("path", path_square), ("edge fade", fade_square)];
    for (_, square) in &kinds {
        match square(0.0, 1.0) {
            LayerMask::Path(_) | LayerMask::EdgeFade(_) => {}
        }
    }
    kinds
}

/// `S` for a mask start, `E` for a mask end, `Q` for each quad, in draw
/// order.
fn sequence(scene: &Scene) -> String {
    scene
        .batches()
        .map(|batch| match batch {
            PrimitiveBatch::StartLayerMask(_) => "S".to_string(),
            PrimitiveBatch::EndLayerMask => "E".to_string(),
            PrimitiveBatch::Quads(range) => "Q".repeat(range.len()),
            other => panic!("unexpected batch {other:?}"),
        })
        .collect()
}

#[test]
fn a_child_at_any_z_index_draws_inside_its_mask() {
    for (kind, square) in mask_kinds() {
        for z_index in [i32::MIN, -1, 1, i32::MAX] {
            let mut scene = Scene::default();
            scene.push_layer_mask(square(0.0, 100.0));
            scene.push_z_index(z_index);
            scene.insert_primitive(quad(rect(10.0, 10.0, 20.0, 20.0)));
            scene.pop_z_index();
            scene.pop_layer_mask();
            scene.finish();
            assert_eq!(sequence(&scene), "SQE", "{kind} at z-index {z_index}");
        }
    }
}

/// Quads painted in the outer mask after the inner mask closed, stacked so
/// each overlaps the one before and none overlaps the inner mask. A quad that
/// does not overlap the inner mask may draw before or after it; every quad
/// must fall between the outer mask's start and end. Covers every pairing of
/// mask kinds.
#[test]
fn children_of_an_outer_mask_painted_after_an_inner_mask_draw_inside_the_outer_mask() {
    for (outer_kind, outer) in mask_kinds() {
        for (inner_kind, inner) in mask_kinds() {
            let mut scene = Scene::default();
            scene.push_layer_mask(outer(0.0, 200.0));
            scene.push_layer_mask(inner(0.0, 50.0));
            scene.insert_primitive(quad(rect(10.0, 10.0, 20.0, 20.0)));
            scene.pop_layer_mask();
            for offset in 0..4 {
                let offset = offset as f32 * 5.0;
                scene.insert_primitive(quad(rect(100.0 + offset, 100.0, 50.0, 50.0)));
            }
            scene.pop_layer_mask();
            scene.finish();
            let drawn = sequence(&scene);
            assert_eq!(drawn.matches('Q').count(), 5, "{drawn}");
            assert!(
                drawn.starts_with('S') && drawn.ends_with('E') && drawn.contains("SQE"),
                "a quad drew outside its mask ({outer_kind} around {inner_kind}): {drawn}"
            );
        }
    }
}

/// A mask around a quad, covering the square from `lo` to `lo + 100`.
fn masked_quad(scene: &mut Scene, square: fn(f32, f32) -> LayerMask, lo: f32) {
    scene.push_layer_mask(square(lo, lo + 100.0));
    scene.insert_primitive(quad(rect(lo + 10.0, lo + 10.0, 20.0, 20.0)));
    scene.pop_layer_mask();
}

/// A quad inside `depth` nested layers, covering the square from `lo` to
/// `lo + 100`.
fn layered_quad(scene: &mut Scene, depth: usize, lo: f32) {
    let bounds = rect(lo, lo, 100.0, 100.0);
    for _ in 0..depth {
        scene.push_layer(bounds);
    }
    scene.insert_primitive(quad(rect(lo + 10.0, lo + 10.0, 20.0, 20.0)));
    for _ in 0..depth {
        scene.pop_layer();
    }
}

/// Paints `masked` inside a layer covering the square from `lo` to
/// `lo + 100`.
fn in_layer(scene: &mut Scene, lo: f32, masked: impl FnOnce(&mut Scene)) {
    scene.push_layer(rect(lo, lo, 100.0, 100.0));
    masked(scene);
    scene.pop_layer();
}

type Arrangement = fn(&mut Scene, fn(f32, f32) -> LayerMask);

/// Every way to paint a scope beside a mask without overlapping it, so that
/// the scope and the mask share their step in the enclosing scope: each
/// operation that opens a scope, at the mask's depth and one level deeper,
/// painted before and after the mask. The exhaustive match fails to compile
/// when a paint operation is added without a case.
fn arrangements_beside_a_mask() -> Vec<(&'static str, Arrangement)> {
    let arrangements: Vec<(&'static str, Arrangement)> = vec![
        ("a quad after the mask", |scene, square| {
            masked_quad(scene, square, 0.0);
            layered_quad(scene, 0, 300.0);
        }),
        ("a layer after the mask", |scene, square| {
            masked_quad(scene, square, 0.0);
            layered_quad(scene, 1, 300.0);
        }),
        ("a layer before the mask", |scene, square| {
            layered_quad(scene, 1, 300.0);
            masked_quad(scene, square, 0.0);
        }),
        ("a layer in a layer after the mask", |scene, square| {
            masked_quad(scene, square, 0.0);
            layered_quad(scene, 2, 300.0);
        }),
        ("a layer in a layer before the mask", |scene, square| {
            layered_quad(scene, 2, 300.0);
            masked_quad(scene, square, 0.0);
        }),
        (
            "a layer in a layer after a mask in a layer",
            |scene, square| {
                in_layer(scene, 0.0, |scene| masked_quad(scene, square, 0.0));
                layered_quad(scene, 2, 300.0);
            },
        ),
        ("a mask after the mask", |scene, square| {
            masked_quad(scene, square, 0.0);
            masked_quad(scene, square, 300.0);
        }),
        (
            "a mask in a layer after a mask in a layer",
            |scene, square| {
                in_layer(scene, 0.0, |scene| masked_quad(scene, square, 0.0));
                in_layer(scene, 300.0, |scene| masked_quad(scene, square, 300.0));
            },
        ),
        (
            "a layer at the lowest z-index in a layer after two masks",
            |scene, square| {
                masked_quad(scene, square, 0.0);
                masked_quad(scene, square, 150.0);
                scene.push_layer(rect(300.0, 300.0, 100.0, 100.0));
                scene.push_z_index(i32::MIN);
                layered_quad(scene, 1, 300.0);
                scene.pop_z_index();
                scene.pop_layer();
            },
        ),
    ];
    let operation = PaintOperation::EndLayer;
    match operation {
        // A quad and a mask: the primitive arrangements above.
        PaintOperation::Primitive(_) => {}
        // A layer: the layer arrangements above.
        PaintOperation::StartLayer(_) | PaintOperation::EndLayer => {}
        // A z-index: the lowest z-index arrangement above, whose layer's step
        // is the smallest a scope's step reaches.
        PaintOperation::PushZIndex(_) | PaintOperation::PopZIndex => {}
    }
    arrangements
}

/// Asserts that the masks open around each quad when it draws are the masks
/// whose bounds contain it, and that every mask closes.
fn assert_each_quad_draws_in_the_masks_around_it(scene: &Scene, arrangement: &str) {
    let masks: Vec<Bounds<ScaledPixels>> = scene
        .batches()
        .filter_map(|batch| match batch {
            PrimitiveBatch::StartLayerMask(mask) => Some(mask.bounds()),
            _ => None,
        })
        .collect();
    let around =
        |mask: &Bounds<ScaledPixels>, quad: &Quad| mask.intersect(&quad.bounds) == quad.bounds;
    let mut open: Vec<Bounds<ScaledPixels>> = Vec::new();
    for batch in scene.batches() {
        match batch {
            PrimitiveBatch::StartLayerMask(mask) => open.push(mask.bounds()),
            PrimitiveBatch::EndLayerMask => {
                assert!(open.pop().is_some(), "{arrangement}: a mask closed twice");
            }
            PrimitiveBatch::Quads(range) => {
                for quad in &scene.quads[range] {
                    let containing = masks.iter().filter(|mask| around(mask, quad)).count();
                    assert!(
                        open.len() == containing && open.iter().all(|mask| around(mask, quad)),
                        "{arrangement}: the quad at {:?} drew inside the masks {open:?}",
                        quad.bounds
                    );
                }
            }
            other => panic!("{arrangement}: unexpected batch {other:?}"),
        }
    }
    assert!(open.is_empty(), "{arrangement}: a mask never closed");
}

/// A scope painted beside a mask draws outside the mask, and the mask's own
/// quad draws inside it, for every mask kind, as painted and as replayed.
#[test]
fn a_scope_beside_a_mask_draws_outside_the_mask() {
    for (kind, square) in mask_kinds() {
        for (arrangement, paint) in arrangements_beside_a_mask() {
            let context = format!("{arrangement} ({kind})");
            let mut scene = Scene::default();
            paint(&mut scene, square);
            scene.finish();
            assert_each_quad_draws_in_the_masks_around_it(&scene, &context);

            let mut replayed = Scene::default();
            replayed.replay(0..scene.len(), &scene);
            replayed.finish();
            assert_each_quad_draws_in_the_masks_around_it(
                &replayed,
                &format!("{context}, replayed"),
            );
        }
    }
}

/// Replaying a masked subtree, and replaying that replay, records the same
/// operations and draws the same batches with the same mask as the original
/// paint.
#[test]
fn replaying_a_mask_records_the_same_operations_and_mask() {
    for (kind, square) in mask_kinds() {
        let mut first = Scene::default();
        first.push_layer_mask(square(0.0, 100.0));
        first.insert_primitive(quad(rect(10.0, 10.0, 20.0, 20.0)));
        first.pop_layer_mask();
        first.finish();

        let operations = |scene: &Scene| {
            scene
                .paint_operations
                .iter()
                .map(std::mem::discriminant)
                .collect::<Vec<_>>()
        };
        let mask = |scene: &Scene| {
            let masks = scene
                .batches()
                .filter_map(|batch| match batch {
                    PrimitiveBatch::StartLayerMask(mask) => Some(format!("{mask:?}")),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(masks.len(), 1, "{kind}");
            masks.into_iter().next()
        };
        let original_mask = mask(&first);
        let mut previous = first;
        for _ in 0..3 {
            let mut next = Scene::default();
            next.replay(0..previous.len(), &previous);
            next.finish();
            assert_eq!(operations(&next), operations(&previous), "{kind}");
            assert_eq!(next.scopes.len(), 1, "replay left a {kind} scope open");
            assert_eq!(sequence(&next), "SQE", "{kind}");
            assert_eq!(mask(&next), original_mask, "{kind}");
            previous = next;
        }
    }
}

/// The composited region of an edge fade: the clip, limited at each faded
/// edge by the fade bounds and left at the clip on each unfaded edge.
#[test]
fn edge_fade_bounds_limit_the_clip_only_at_faded_edges() {
    let fade = rect(100.0, 100.0, 200.0, 200.0);
    let clip = rect(50.0, 50.0, 400.0, 400.0);
    let cases = [
        (
            "every edge",
            edges(8.0, 8.0, 8.0, 8.0),
            rect(100.0, 100.0, 200.0, 200.0),
        ),
        (
            "top",
            edges(8.0, 0.0, 0.0, 0.0),
            rect(50.0, 100.0, 400.0, 350.0),
        ),
        (
            "right",
            edges(0.0, 8.0, 0.0, 0.0),
            rect(50.0, 50.0, 250.0, 400.0),
        ),
        (
            "bottom",
            edges(0.0, 0.0, 8.0, 0.0),
            rect(50.0, 50.0, 400.0, 250.0),
        ),
        (
            "left",
            edges(0.0, 0.0, 0.0, 8.0),
            rect(100.0, 50.0, 350.0, 400.0),
        ),
        (
            "top and bottom",
            edges(8.0, 0.0, 8.0, 0.0),
            rect(50.0, 100.0, 400.0, 200.0),
        ),
        ("no edge", edges(0.0, 0.0, 0.0, 0.0), clip),
        ("negative bands", edges(-8.0, -1.0, -8.0, -1.0), clip),
        (
            "NaN bands",
            edges(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            clip,
        ),
    ];
    for (name, bands, expected) in cases {
        let mask = EdgeFadeMask::new(fade, bands, clip);
        assert_eq!(mask.bounds, expected, "{name}");
        assert_eq!(mask.fade_bounds, fade, "{name}");
        for band in [
            mask.bands.top,
            mask.bands.right,
            mask.bands.bottom,
            mask.bands.left,
        ] {
            assert!(band.0 >= 0.0, "{name}: band {band:?} is not a width");
        }
    }
}

/// A clip that lies entirely beyond a faded edge leaves an empty region, never
/// an inverted one, whichever side it lies on.
#[test]
fn edge_fade_bounds_are_empty_when_the_clip_lies_beyond_a_faded_edge() {
    let fade = rect(100.0, 100.0, 100.0, 100.0);
    let bands = edges(8.0, 8.0, 8.0, 8.0);
    for clip in [
        rect(0.0, 0.0, 50.0, 1000.0),
        rect(300.0, 0.0, 50.0, 1000.0),
        rect(0.0, 0.0, 1000.0, 50.0),
        rect(0.0, 300.0, 1000.0, 50.0),
    ] {
        let bounds = EdgeFadeMask::new(fade, bands, clip).bounds;
        assert!(bounds.is_empty(), "{clip:?} gave {bounds:?}");
        assert!(
            bounds.size.width.0 >= 0.0 && bounds.size.height.0 >= 0.0,
            "{clip:?} gave inverted bounds {bounds:?}"
        );
    }
}
