//! A window that closes partway through its own frame, or before a queued
//! platform event or a deferred action reaches it, has nothing left to update.
//! Those late updates end without logging: the frame stops and the event or
//! action is dropped. An update that fails for another reason, a window held
//! further up the stack, is still logged as an error.
//!
//! Not covered: a platform callback added to `Window::new` that bypasses
//! `update_from_platform`, and the callbacks the test platform does not store
//! (button layout, window tabs, accessibility).

use std::cell::RefCell;
use std::sync::Once;

use crate::{
    self as gpui, Context, EmptyView, IntoElement, Modifiers, MouseMoveEvent, PlatformInput,
    Render, RequestFrameOptions, TestAppContext, TestWindow, Window, WindowAppearance, div, point,
    px, size,
};

crate::actions!(closed_tests, [Poke]);

thread_local! {
    static RECORDS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Records every warning and error logged on the thread that logs it.
struct Recorder;

impl log::Log for Recorder {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let line = format!("{} {}: {}", record.level(), record.target(), record.args());
            RECORDS.with_borrow_mut(|records| records.push(line));
        }
    }

    fn flush(&self) {}
}

/// The warnings and errors this thread logs while `run` runs.
fn logged(run: impl FnOnce()) -> Vec<String> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&Recorder).expect("gpui's tests install no other logger");
        log::set_max_level(log::LevelFilter::Warn);
    });
    RECORDS.with_borrow_mut(Vec::clear);
    log::error!(target: "closed_tests", "probe");
    run();
    let mut records = RECORDS.take();
    assert_eq!(
        records.first().map(String::as_str),
        Some("ERROR closed_tests: probe"),
        "the recorder receives this thread's records"
    );
    records.remove(0);
    records
}

struct ClosesWhileRendering {
    close: bool,
}

impl Render for ClosesWhileRendering {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        if self.close {
            window.remove_window();
        }
        div()
    }
}

#[gpui::test]
fn a_window_its_own_render_closes_ends_the_frame_without_an_error(cx: &mut TestAppContext) {
    let window = cx.add_window(|_, _| ClosesWhileRendering { close: false });
    let test_window = cx.test_window(window.into());
    // A test app draws a notified window as its update ends, so the flag is
    // set without a notify and the frame request forces the draw: the render
    // then closes the window inside the frame, as an animation frame's does.
    window.update(cx, |view, _, _| view.close = true).unwrap();

    let records = logged(|| {
        test_window.simulate_frame_request(RequestFrameOptions {
            require_presentation: true,
            force_render: true,
        })
    });

    assert!(
        window.update(cx, |_, _, _| ()).is_err(),
        "the render closed the window"
    );
    assert_eq!(records, Vec::<String>::new());
}

#[gpui::test]
fn a_window_a_next_frame_callback_closes_ends_the_frame_without_an_error(cx: &mut TestAppContext) {
    // After its callbacks a clean window only presents; a dirtied one draws.
    for dirty in [false, true] {
        let window = cx.add_window(|_, _| EmptyView);
        let test_window = cx.test_window(window.into());
        test_window.simulate_frame_request(RequestFrameOptions::default());
        window
            .update(cx, |_, window, _| {
                window.on_next_frame(move |window, _| {
                    if dirty {
                        window.refresh();
                    }
                    window.remove_window();
                })
            })
            .unwrap();

        // A presentation request is never throttled, so the callback runs now.
        let records = logged(|| {
            test_window.simulate_frame_request(RequestFrameOptions {
                require_presentation: true,
                force_render: false,
            })
        });

        assert!(
            window.update(cx, |_, _, _| ()).is_err(),
            "the next-frame callback closed the window"
        );
        assert_eq!(records, Vec::<String>::new(), "dirty: {dirty}");
    }
}

type Deliver = fn(&mut TestWindow, &mut TestAppContext);

#[gpui::test]
fn platform_events_that_reach_a_closed_window_are_dropped_without_an_error(
    cx: &mut TestAppContext,
) {
    let events: [(&str, Deliver); 8] = [
        ("a frame request", |window, _| {
            window.simulate_frame_request(RequestFrameOptions::default())
        }),
        ("input", |window, _| {
            window.simulate_input(PlatformInput::MouseMove(MouseMoveEvent {
                position: point(px(10.), px(10.)),
                modifiers: Modifiers::default(),
                pressed_button: None,
            }));
        }),
        ("a resize", |window, _| {
            window.simulate_resize(size(px(300.), px(200.)))
        }),
        ("a move", |window, _| window.simulate_moved()),
        ("an activation change", |window, _| {
            window.simulate_active_status_change(true)
        }),
        ("a hover change", |window, _| {
            window.simulate_hover_status_change(true)
        }),
        ("a window control hit test", |window, _| {
            assert_eq!(window.simulate_hit_test_window_control(), None);
        }),
        ("an appearance change", |window, cx| {
            window.simulate_appearance_change(WindowAppearance::Dark);
            cx.run_until_parked();
        }),
    ];

    for (event, deliver) in events {
        let window = cx.add_window(|_, _| EmptyView);
        let mut test_window = cx.test_window(window.into());
        window
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();

        let records = logged(|| deliver(&mut test_window, cx));

        assert_eq!(
            records,
            Vec::<String>::new(),
            "{event} delivered after the window closed"
        );
    }
}

#[gpui::test]
fn an_action_dispatched_as_its_window_closes_is_dropped_without_an_error(cx: &mut TestAppContext) {
    let window = cx.add_window(|_, _| EmptyView);

    let records = logged(|| {
        window
            .update(cx, |_, window, cx| {
                window.dispatch_action(Box::new(Poke), cx);
                window.remove_window();
            })
            .unwrap();
    });

    assert_eq!(records, Vec::<String>::new());
}

#[gpui::test]
fn a_platform_update_that_finds_its_window_held_up_the_stack_is_still_logged(
    cx: &mut TestAppContext,
) {
    let window = cx.add_window(|_, _| EmptyView);
    let mut test_window = cx.test_window(window.into());

    let records = logged(|| {
        window
            .update(cx, |_, _, _| {
                test_window.simulate_resize(size(px(300.), px(200.)))
            })
            .unwrap();
    });

    assert_eq!(records.len(), 1, "{records:?}");
    assert!(
        records[0].starts_with("ERROR gpui::window: "),
        "{records:?}"
    );
}
