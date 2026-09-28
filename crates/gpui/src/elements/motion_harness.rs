//! Frame stepping and paint readback for the tests of the motion elements.

use std::time::Duration;

use crate::{Bounds, Pixels, Render, TestAppContext, WindowHandle, motion::SpringConfig, point, px};

/// Clock time between simulated frames.
pub(crate) const FRAME: Duration = Duration::from_millis(16);
/// Frames after which a motion that still requests frames fails a test.
pub(crate) const MAX_FRAMES: usize = 1000;
/// Renders at rest after which a motion must still request no frame.
pub(crate) const IDLE_RENDERS: usize = 10;

/// Advances the clock by [`FRAME`], delivers the frames the last draw
/// requested, draws, and returns how many frames were delivered.
pub(crate) fn next_frame<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext) -> usize {
    cx.executor().advance_clock(FRAME);
    let delivered = window
        .update(cx, |_, window, cx| window.simulate_next_frame(cx))
        .unwrap();
    cx.run_until_parked();
    delivered
}

/// Updates the view with `f`, notifies it, and draws, without advancing the
/// clock.
pub(crate) fn update_view<V: Render>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
    f: impl FnOnce(&mut V),
) {
    window
        .update(cx, |view, _, cx| {
            f(view);
            cx.notify();
        })
        .unwrap();
    cx.run_until_parked();
}

/// Renders the view [`IDLE_RENDERS`] times one frame apart and returns how
/// many frames those renders requested.
pub(crate) fn idle_requests<V: Render>(window: &WindowHandle<V>, cx: &mut TestAppContext) -> usize {
    (0..IDLE_RENDERS)
        .map(|_| {
            update_view(window, cx, |_| {});
            next_frame(window, cx)
        })
        .sum()
}

/// The bounds the element with `selector` as its debug selector was painted
/// at in the last drawn frame.
pub(crate) fn painted_bounds<V: 'static>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
    selector: &str,
) -> Option<Bounds<Pixels>> {
    window
        .update(cx, |_, window, _| {
            window.rendered_frame.debug_bounds.get(selector).copied()
        })
        .unwrap()
}

/// The damage of the last drawn frame: `None` for the whole viewport.
pub(crate) fn last_damage<V: 'static>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
) -> Option<Bounds<Pixels>> {
    window
        .update(cx, |_, window, _| window.last_frame_damage())
        .unwrap()
}

/// Half a device pixel of the window, in logical pixels: how far a painted
/// position snapped to device pixels lies from the unsnapped one.
pub(crate) fn snap_tolerance<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext) -> f32 {
    let scale_factor = window
        .update(cx, |_, window, _| window.scale_factor())
        .unwrap();
    0.5 / scale_factor + 1e-3
}

/// The value of `spring` released from `start` at `velocity` toward
/// `target`, `frames` frames of [`FRAME`] later.
pub(crate) fn spring_at(
    spring: SpringConfig,
    start: f32,
    velocity: f32,
    target: f32,
    frames: u32,
) -> (f32, f32) {
    let state = spring.evaluate(start, velocity, target, (FRAME * frames).as_secs_f32());
    (state.position, state.velocity)
}

/// Whether `outer` contains `inner`.
pub(crate) fn contains(outer: Bounds<Pixels>, inner: Bounds<Pixels>) -> bool {
    outer.left() <= inner.left()
        && outer.top() <= inner.top()
        && outer.right() >= inner.right()
        && outer.bottom() >= inner.bottom()
}

/// `bounds` grown by one pixel on every side.
pub(crate) fn dilated(bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    Bounds::from_corners(
        bounds.origin - point(px(1.0), px(1.0)),
        bounds.bottom_right() + point(px(1.0), px(1.0)),
    )
}
