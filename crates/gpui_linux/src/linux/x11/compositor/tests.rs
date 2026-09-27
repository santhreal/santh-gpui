//! WHY: compositor detection read `_NET_WM_CM_S<root window id>`, a selection
//! no compositing manager owns, and counted any EWMH window manager as a
//! compositing manager. On a screen with a window manager and no compositing
//! manager, a transparent window then opened with an ARGB background pixel
//! the server draws as black, and client-side decorations were offered with
//! nothing to blend their transparent margins. This test walks one screen
//! through every signal and checks that only a compositing manager's own
//! signals count, and that each stops counting once withdrawn.
//!
//! Not covered: a compositing manager that sets neither signal, and screens
//! other than the connection's default screen.

use super::compositor_present;
use std::ffi::CString;
use x11rb::{
    CURRENT_TIME, NONE,
    connection::Connection as _,
    protocol::xproto::{self, AtomEnum, ConnectionExt as _, PropMode, WindowClass},
    wrapper::ConnectionExt as _,
    xcb_ffi::XCBConnection,
};

fn connect() -> anyhow::Result<(XCBConnection, usize)> {
    let display = std::env::var("GPUI_X11_TEST_DISPLAY")?;
    anyhow::ensure!(
        display != ":0" && display != ":1",
        "a private display is required"
    );
    Ok(XCBConnection::connect(Some(&CString::new(display)?))?)
}

fn intern(xcb: &XCBConnection, name: &str) -> anyhow::Result<xproto::Atom> {
    Ok(xcb.intern_atom(false, name.as_bytes())?.reply()?.atom)
}

#[test]
#[ignore = "requires a dedicated X server and GPUI_X11_TEST_DISPLAY"]
fn only_a_compositing_manager_signal_counts() -> anyhow::Result<()> {
    let (xcb, screen) = connect()?;
    let root = xcb.setup().roots[screen].root;
    let owner_property = intern(&xcb, "_NET_WM_CM_OWNER")?;
    let wm_check = intern(&xcb, "_NET_SUPPORTING_WM_CHECK")?;
    let root_named = intern(&xcb, &format!("_NET_WM_CM_S{root}"))?;
    let selection = intern(&xcb, &format!("_NET_WM_CM_S{screen}"))?;
    let present = || compositor_present(&xcb, screen, root);

    xcb.delete_property(root, owner_property)?.check()?;
    xcb.delete_property(root, wm_check)?.check()?;
    assert!(!present(), "a server with no manager has no compositor");

    let window = xcb.generate_id()?;
    xcb.create_window(
        0,
        window,
        root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        0,
        &Default::default(),
    )?
    .check()?;

    for holder in [root, window] {
        xcb.change_property32(
            PropMode::REPLACE,
            holder,
            wm_check,
            AtomEnum::WINDOW,
            &[window],
        )?
        .check()?;
    }
    assert!(!present(), "a window manager alone composites nothing");

    xcb.set_selection_owner(window, root_named, CURRENT_TIME)?
        .check()?;
    assert!(
        !present(),
        "the selection is named after the screen number, not the root window id"
    );

    xcb.set_selection_owner(window, selection, CURRENT_TIME)?
        .check()?;
    assert!(present(), "an owned _NET_WM_CM_S<screen> is a compositor");
    xcb.set_selection_owner(NONE, selection, CURRENT_TIME)?
        .check()?;
    assert!(!present(), "a released selection is no compositor");

    xcb.change_property32(
        PropMode::REPLACE,
        root,
        owner_property,
        AtomEnum::WINDOW,
        &[window],
    )?
    .check()?;
    assert!(present(), "_NET_WM_CM_OWNER on the root is a compositor");
    xcb.delete_property(root, owner_property)?.check()?;
    assert!(!present(), "a deleted _NET_WM_CM_OWNER is no compositor");

    xcb.delete_property(root, wm_check)?.check()?;
    xcb.destroy_window(window)?.check()?;
    xcb.flush()?;
    Ok(())
}
