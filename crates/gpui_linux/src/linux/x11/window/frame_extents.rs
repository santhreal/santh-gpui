//! The `_GTK_FRAME_EXTENTS` of a window that draws its own frame.
//!
//! A client-decorated window keeps room around its frame for a shadow and
//! resize handles, and `_GTK_FRAME_EXTENTS` lists that room per side. A
//! window manager that reads it places, snaps and tiles the frame instead of
//! the whole window and draws no shadow of its own. Mutter reads the property
//! when the window maps, so a window opened with a client inset has it
//! before the map. A side tiled against a screen edge or a neighbor keeps no
//! room, and a window the window manager frames has no property.

use gpui::{Decorations, Tiling};

/// The sides of a client-decorated window tiled against a screen edge or a
/// neighbor: every side of a fullscreen window, the sides
/// `_GTK_EDGE_CONSTRAINTS` lists when the window manager sets it, and the
/// sides a maximize spans otherwise.
pub(super) fn tiling(
    fullscreen: bool,
    edge_constraints: Option<Tiling>,
    maximized_horizontal: bool,
    maximized_vertical: bool,
) -> Tiling {
    if fullscreen {
        return Tiling::tiled();
    }
    // https://source.chromium.org/chromium/chromium/src/+/main:ui/ozone/platform/x11/x11_window.cc;l=2519;drc=1f14cc876cc5bf899d13284a12c451498219bb2d
    edge_constraints.unwrap_or(Tiling {
        top: maximized_vertical,
        bottom: maximized_vertical,
        left: maximized_horizontal,
        right: maximized_horizontal,
    })
}

/// The `_GTK_FRAME_EXTENTS` for a window with `decorations` and a client
/// inset of `inset` device pixels, in the property's order: left, right,
/// top, bottom. `None` is a window with no property: one the window manager
/// frames, or one with no client inset.
pub(super) fn frame_extents(decorations: Decorations, inset: Option<u32>) -> Option<[u32; 4]> {
    let (Decorations::Client { tiling }, Some(inset)) = (decorations, inset) else {
        return None;
    };
    let side = |tiled: bool| if tiled { 0 } else { inset };
    Some([
        side(tiling.left),
        side(tiling.right),
        side(tiling.top),
        side(tiling.bottom),
    ])
}

#[cfg(test)]
mod tests;
