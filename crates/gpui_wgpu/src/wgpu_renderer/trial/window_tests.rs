//! WHY: an X11 client's first window adopts a context whose adapter was
//! selected without a surface and does not wait for its surface's first
//! configure. These tests drive window renderers on an X server through
//! that configure's result and close the defect classes of reading it:
//!
//! - a failed configure that leaves the untested context in place, so the
//!   window draws failed frames on an adapter that cannot present to it;
//! - a recovery that keeps the rejected context, or that fails to record
//!   that its replacement configured a surface, so the replacement is
//!   rejected in turn;
//! - a successful configure that is not recorded, so a later window's
//!   failed configure throws away a context that works;
//! - a failure on a context that configured a surface that replaces the
//!   context instead of failing a frame;
//! - a window left on the rejected context after another window replaced
//!   it, or one whose recovery replaces the replacement again.
//!
//! The configure's failure is injected: the error is recorded where the
//! device's error handler records it, after a configure that succeeded.
//! Not covered: a configure that fails in the driver, which the test
//! server's adapter does not do, and the platform calling
//! `needs_recovery` before it draws.
//!
//! The X server must run with -noreset. The other tests of this crate load
//! GPU drivers that connect to `DISPLAY` and disconnect, and a server that
//! resets when its last client disconnects fails a test's connection that
//! is still in setup.

use crate::wgpu_renderer::{WgpuRenderer, WgpuSurfaceConfig};
use crate::{DisplayInstances, GpuContext, WgpuContext};
use gpui::{DevicePixels, Size};
use raw_window_handle as rwh;
use std::ffi::{CString, c_void};
use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::sync::Arc;
use x11rb::connection::Connection as _;
use x11rb::protocol::xproto::{ConnectionExt as _, CreateWindowAux, WindowClass};
use x11rb::xcb_ffi::XCBConnection;

const SIDE: u16 = 64;

/// A window of the test's X connection.
#[derive(Debug, Clone, Copy)]
struct XcbWindow {
    connection: NonNull<c_void>,
    screen: i32,
    window: NonZeroU32,
    visual: u32,
}

// Safety: an xcb connection is safe to use from any thread, and each test
// holds its connection open until its renderers and contexts are dropped.
unsafe impl Send for XcbWindow {}
unsafe impl Sync for XcbWindow {}

impl rwh::HasWindowHandle for XcbWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        let mut handle = rwh::XcbWindowHandle::new(self.window);
        handle.visual_id = NonZeroU32::new(self.visual);
        // Safety: the window lives as long as the test's connection.
        Ok(unsafe { rwh::WindowHandle::borrow_raw(handle.into()) })
    }
}

impl rwh::HasDisplayHandle for XcbWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        let handle = rwh::XcbDisplayHandle::new(Some(self.connection), self.screen);
        // Safety: the connection outlives every renderer of the test.
        Ok(unsafe { rwh::DisplayHandle::borrow_raw(handle.into()) })
    }
}

/// The connection to the X server `GPUI_X11_TEST_DISPLAY` names, and its
/// default screen.
fn connect() -> (XCBConnection, usize) {
    let display = std::env::var("GPUI_X11_TEST_DISPLAY")
        .expect("GPUI_X11_TEST_DISPLAY names the X server the test opens windows on");
    XCBConnection::connect(Some(&CString::new(display).expect("a display name")))
        .expect("connect to the test X server")
}

/// Opens and maps a window on `screen`.
fn open_window(xcb: &XCBConnection, screen: usize) -> XcbWindow {
    let root = &xcb.setup().roots[screen];
    let window = xcb.generate_id().expect("a window id");
    xcb.create_window(
        root.root_depth,
        window,
        root.root,
        0,
        0,
        SIDE,
        SIDE,
        0,
        WindowClass::INPUT_OUTPUT,
        root.root_visual,
        &CreateWindowAux::new(),
    )
    .expect("send CreateWindow")
    .check()
    .expect("create the window");
    xcb.map_window(window)
        .expect("send MapWindow")
        .check()
        .expect("map the window");
    XcbWindow {
        connection: NonNull::new(xcb.get_raw_xcb_connection().cast())
            .expect("the connection's xcb handle"),
        screen: i32::try_from(screen).expect("a screen number"),
        window: NonZeroU32::new(window).expect("a nonzero window id"),
        visual: root.root_visual,
    }
}

/// The display's GPU context as an X11 client's first window finds it: an
/// adapter selected without a surface.
fn untested_display(window: &XcbWindow) -> GpuContext {
    let instances = DisplayInstances::new(*window);
    let context = WgpuContext::for_display(&instances, None).expect("a context without a surface");
    assert!(
        !context.surface_tested(),
        "a context selected without a surface reads as having configured one"
    );
    let display = GpuContext::with_instances(instances);
    *display.borrow_mut() = Some(context);
    display
}

/// A renderer of `window` on the display's context whose first configure
/// has returned.
fn renderer(display: &GpuContext, window: &XcbWindow) -> WgpuRenderer {
    let config = WgpuSurfaceConfig {
        size: Size {
            width: DevicePixels(i32::from(SIDE)),
            height: DevicePixels(i32::from(SIDE)),
        },
        transparent: false,
        preferred_present_mode: None,
    };
    let mut renderer =
        WgpuRenderer::new(display.clone(), window, config, None).expect("a window renderer");
    renderer
        .finish_configure()
        .expect("the test server's adapter configures the surface");
    renderer
}

/// Records a configure error as the device's error handler does.
fn fail_configure(renderer: &WgpuRenderer) {
    *renderer.last_error.lock() = Some("the surface's configure failed".into());
}

fn display_device(display: &GpuContext) -> Arc<wgpu::Device> {
    Arc::clone(
        &display
            .borrow()
            .as_ref()
            .expect("the display has a context")
            .device,
    )
}

fn display_tested(display: &GpuContext) -> bool {
    display
        .borrow()
        .as_ref()
        .expect("the display has a context")
        .surface_tested()
}

fn renderer_device(renderer: &WgpuRenderer) -> Arc<wgpu::Device> {
    Arc::clone(&renderer.resources().device)
}

#[test]
#[ignore = "requires an X server and GPUI_X11_TEST_DISPLAY"]
fn a_failed_first_configure_replaces_the_untested_context() {
    let (xcb, screen) = connect();
    let window = open_window(&xcb, screen);
    let display = untested_display(&window);
    let untested = display_device(&display);

    let mut renderer = renderer(&display, &window);
    fail_configure(&renderer);
    assert!(
        renderer.needs_recovery(),
        "the context that failed its first configure was kept"
    );
    renderer
        .recover(&window)
        .expect("recover on a context selected by configuring the surface");

    let replacement = display_device(&display);
    assert!(
        !Arc::ptr_eq(&replacement, &untested),
        "the rejected context is still the display's"
    );
    assert!(Arc::ptr_eq(&renderer_device(&renderer), &replacement));
    assert!(
        display_tested(&display),
        "the replacement reads as having configured no surface"
    );
    renderer
        .finish_configure()
        .expect("the replacement configures the surface");
    fail_configure(&renderer);
    assert!(
        !renderer.needs_recovery(),
        "a failure on the replacement replaced it in turn"
    );
}

#[test]
#[ignore = "requires an X server and GPUI_X11_TEST_DISPLAY"]
fn a_first_configure_that_succeeds_keeps_the_context_and_records_it() {
    let (xcb, screen) = connect();
    let window = open_window(&xcb, screen);
    let display = untested_display(&window);
    let untested = display_device(&display);

    let mut renderer = renderer(&display, &window);
    assert!(!renderer.needs_recovery(), "a working context was rejected");
    assert!(Arc::ptr_eq(&display_device(&display), &untested));
    assert!(
        display_tested(&display),
        "the configure that succeeded was not recorded on the context"
    );
}

#[test]
#[ignore = "requires an X server and GPUI_X11_TEST_DISPLAY"]
fn a_failure_on_a_context_that_configured_a_surface_fails_a_frame_instead() {
    let (xcb, screen) = connect();
    let first_window = open_window(&xcb, screen);
    let second_window = open_window(&xcb, screen);
    let third_window = open_window(&xcb, screen);
    let display = untested_display(&first_window);
    let context = display_device(&display);

    // The second window opens before the first window's configure is
    // read, and fails after the first window's succeeded.
    let mut first = renderer(&display, &first_window);
    let mut second = renderer(&display, &second_window);
    assert!(!first.needs_recovery());
    fail_configure(&second);
    assert!(
        !second.needs_recovery(),
        "a context that configured another window's surface was rejected"
    );

    // The third window opens on a context that has configured a surface.
    let mut third = renderer(&display, &third_window);
    fail_configure(&third);
    assert!(
        !third.needs_recovery(),
        "a context that configured a surface was rejected"
    );

    assert!(Arc::ptr_eq(&display_device(&display), &context));
    for failed in [&second, &third] {
        assert!(
            failed.last_error.lock().is_some(),
            "the error no longer fails the renderer's next frame"
        );
    }
}

#[test]
#[ignore = "requires an X server and GPUI_X11_TEST_DISPLAY"]
fn a_window_on_a_rejected_context_moves_to_its_replacement() {
    let (xcb, screen) = connect();
    let first_window = open_window(&xcb, screen);
    let second_window = open_window(&xcb, screen);
    let display = untested_display(&first_window);
    let untested = display_device(&display);

    let mut first = renderer(&display, &first_window);
    let mut second = renderer(&display, &second_window);
    fail_configure(&first);
    fail_configure(&second);

    assert!(first.needs_recovery());
    first
        .recover(&first_window)
        .expect("recover the first window");
    let replacement = display_device(&display);
    assert!(!Arc::ptr_eq(&replacement, &untested));

    assert!(
        second.needs_recovery(),
        "the second window stays on the rejected context"
    );
    second
        .recover(&second_window)
        .expect("recover the second window");
    assert!(
        Arc::ptr_eq(&display_device(&display), &replacement),
        "the second window replaced the first window's replacement"
    );
    assert!(Arc::ptr_eq(&renderer_device(&second), &replacement));
}
