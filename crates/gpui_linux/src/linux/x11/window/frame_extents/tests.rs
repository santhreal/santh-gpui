//! WHY: a client-decorated window on X11 wrote `_GTK_FRAME_EXTENTS` only
//! when the app set its client inset, from the tiling of that moment, so a
//! window that maximized afterwards kept its shadow room listed and one that
//! set the inset after the map got no shadow from Mutter, which reads the
//! property at the map. These tests pin the property a window carries for
//! each frame, each tiling and each inset: the room on every untiled side in
//! left, right, top, bottom order, none on a tiled side, and no property for
//! a window the window manager frames or one with no inset. Not covered:
//! the write itself and when it runs, which need an X server.

use super::{frame_extents, tiling};
use gpui::{Decorations, Tiling};

fn client(tiling: Tiling) -> Decorations {
    Decorations::Client { tiling }
}

/// Tiled on one side only.
fn only(side: &str) -> Tiling {
    let tiling = Tiling {
        top: side == "top",
        left: side == "left",
        right: side == "right",
        bottom: side == "bottom",
    };
    assert!(tiling.is_tiled(), "no side named {side}");
    tiling
}

#[test]
fn an_untiled_client_frame_lists_its_inset_on_every_side() {
    assert_eq!(
        frame_extents(client(Tiling::default()), Some(40)),
        Some([40, 40, 40, 40])
    );
}

#[test]
fn a_tiled_side_lists_no_room_in_left_right_top_bottom_order() {
    let cases = [
        ("left", [0, 40, 40, 40]),
        ("right", [40, 0, 40, 40]),
        ("top", [40, 40, 0, 40]),
        ("bottom", [40, 40, 40, 0]),
    ];
    for (side, extents) in cases {
        assert_eq!(
            frame_extents(client(only(side)), Some(40)),
            Some(extents),
            "tiled {side}"
        );
    }
    assert_eq!(
        frame_extents(client(Tiling::tiled()), Some(40)),
        Some([0, 0, 0, 0])
    );
}

#[test]
fn a_window_framed_by_the_window_manager_or_with_no_inset_has_no_property() {
    assert_eq!(frame_extents(Decorations::Server, Some(40)), None);
    assert_eq!(frame_extents(Decorations::Server, None), None);
    assert_eq!(frame_extents(client(Tiling::default()), None), None);
    assert_eq!(frame_extents(client(Tiling::tiled()), None), None);
}

#[test]
fn fullscreen_tiles_every_side_over_edge_constraints_and_maximize() {
    for constraints in [None, Some(Tiling::default()), Some(only("left"))] {
        for (horizontal, vertical) in [(false, false), (true, false), (false, true)] {
            assert_eq!(
                tiling(true, constraints, horizontal, vertical),
                Tiling::tiled(),
                "{constraints:?} maximized horizontally {horizontal}, vertically {vertical}"
            );
        }
    }
}

#[test]
fn edge_constraints_set_the_tiling_over_maximize() {
    assert_eq!(
        tiling(false, Some(only("right")), true, true),
        only("right")
    );
    assert_eq!(
        tiling(false, Some(Tiling::default()), true, true),
        Tiling::default()
    );
}

#[test]
fn a_maximize_tiles_the_sides_it_spans() {
    assert_eq!(tiling(false, None, false, false), Tiling::default());
    assert_eq!(
        tiling(false, None, true, false),
        Tiling {
            left: true,
            right: true,
            ..Tiling::default()
        }
    );
    assert_eq!(
        tiling(false, None, false, true),
        Tiling {
            top: true,
            bottom: true,
            ..Tiling::default()
        }
    );
    assert_eq!(tiling(false, None, true, true), Tiling::tiled());
}
