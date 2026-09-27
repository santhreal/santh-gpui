//! Detection of a compositing manager on an X screen.

use gpui_util::ResultExt as _;
use log::Level;
use x11rb::{
    protocol::xproto::{self, ConnectionExt as _},
    xcb_ffi::XCBConnection,
};

use super::get_reply;

/// The root window property some older compositing managers set in place of
/// owning the EWMH selection.
const OWNER_PROPERTY: &str = "_NET_WM_CM_OWNER";

/// Whether a compositing manager runs on screen `screen`, whose root window
/// is `root`: a client owns the screen's `_NET_WM_CM_S<screen>` selection
/// (EWMH), or `root` has a `_NET_WM_CM_OWNER` property. A window manager
/// alone (`_NET_SUPPORTING_WM_CHECK`) composites nothing and does not count:
/// the alpha in a window's pixels blends only under a compositing manager.
pub(crate) fn compositor_present(xcb: &XCBConnection, screen: usize, root: xproto::Window) -> bool {
    let selection = format!("_NET_WM_CM_S{screen}");
    let owned = existing_atom(xcb, &selection).is_some_and(|atom| {
        get_reply(
            || format!("Failed to get the {selection} owner"),
            xcb.get_selection_owner(atom),
        )
        .map(|reply| reply.owner != x11rb::NONE)
        .log_with_level(Level::Debug)
        .unwrap_or(false)
    });
    let announced = existing_atom(xcb, OWNER_PROPERTY).is_some_and(|atom| {
        get_reply(
            || format!("Failed to get {OWNER_PROPERTY}"),
            xcb.get_property(false, root, atom, xproto::AtomEnum::WINDOW, 0, 1),
        )
        .map(|reply| reply.value_len > 0)
        .log_with_level(Level::Debug)
        .unwrap_or(false)
    });
    log::debug!(
        "x11: compositor detection: {selection} owned: {owned}, {OWNER_PROPERTY} set: {announced}"
    );
    owned || announced
}

/// The atom named `name`, or `None` when no client interned it. An atom no
/// client interned has no selection owner and names no property, so the
/// lookup creates none.
fn existing_atom(xcb: &XCBConnection, name: &str) -> Option<xproto::Atom> {
    get_reply(
        || format!("Failed to look up the atom {name}"),
        xcb.intern_atom(true, name.as_bytes()),
    )
    .log_with_level(Level::Debug)
    .map(|reply| reply.atom)
    .filter(|&atom| atom != x11rb::NONE)
}

#[cfg(test)]
mod tests;
