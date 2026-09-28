//! Smooth wheel scrolling sampled frame by frame on the test clock.
//!
//! WHY: closes the classes "a line wheel tick jumps instead of easing", "a
//! tick during a motion restarts it from rest or drops the earlier ticks", "a
//! touchpad event is delayed or offset by a pending motion", "a motion runs
//! past the end of the content", "a scroll state that did not opt in changes
//! behavior", "reduced motion still eases", "a resting scroll state keeps
//! requesting frames", and "a tick up leaves a list following its tail, or a
//! motion to the end leaves it detached". Offsets are compared with the
//! closed-form spring at each frame instant for a scrollable div, a uniform
//! list and a list. Not caught: a scrollbar painted outside the scroll
//! container from the offset lies outside the declared damage.

use super::*;
use crate::{
    Context, FollowMode, InputEvent as _, InteractiveElement as _, IntoElement, ListAlignment,
    ListState, ParentElement as _, Render, ScrollHandle, ScrollWheelEvent,
    StatefulInteractiveElement as _, Styled as _, TestAppContext, UniformListScrollHandle,
    WindowHandle, div,
    elements::motion_harness::{
        MAX_FRAMES, contains, dilated, idle_requests, last_damage, next_frame, painted_bounds,
        spring_at, update_view,
    },
    list,
    motion::MotionPolicy,
    size, uniform_list,
};

const VIEWPORT: f32 = 200.0;
const ROW: f32 = 50.0;
const ROWS: usize = 20;
/// The largest scroll distance: the rows below the viewport.
const MAX: f32 = ROW * ROWS as f32 - VIEWPORT;
/// A sample may lie this far from the closed-form spring: the rest distance
/// a landing snaps across, plus rounding.
const TOLERANCE: f32 = PIXEL_REST_DISTANCE + 1e-3;
const SCROLLER: &str = "scroller";

struct Scroller(ScrollHandle);

impl Render for Scroller {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("scroller")
            .debug_selector(|| SCROLLER.into())
            .flex()
            .flex_col()
            .w(px(VIEWPORT))
            .h(px(VIEWPORT))
            .overflow_y_scroll()
            .track_scroll(&self.0)
            .children((0..ROWS).map(|_| div().flex_shrink_0().h(px(ROW))))
    }
}

struct Uniform(UniformListScrollHandle);

impl Render for Uniform {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        uniform_list("rows", ROWS, |range, _, _| {
            range.map(|_| div().h(px(ROW))).collect()
        })
        .track_scroll(&self.0)
        .w(px(VIEWPORT))
        .h(px(VIEWPORT))
    }
}

struct Rows(ListState);

impl Render for Rows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        list(self.0.clone(), |_, _, _| div().h(px(ROW)).w_full().into_any_element())
            .w(px(VIEWPORT))
            .h(px(VIEWPORT))
    }
}

fn open<V: Render>(cx: &mut TestAppContext, view: V) -> WindowHandle<V> {
    let window = cx.open_window(size(px(VIEWPORT), px(400.0)), move |_, _| view);
    cx.run_until_parked();
    window
}

fn scroller(cx: &mut TestAppContext, smooth: bool) -> (ScrollHandle, WindowHandle<Scroller>) {
    let handle = ScrollHandle::new();
    handle.set_smooth_wheel(smooth);
    let window = open(cx, Scroller(handle.clone()));
    (handle, window)
}

fn rows(cx: &mut TestAppContext, follow: bool) -> (ListState, WindowHandle<Rows>) {
    let state = ListState::new(ROWS, ListAlignment::Top, px(0.0)).measure_all();
    state.set_smooth_wheel(true);
    if follow {
        state.set_follow_mode(FollowMode::Tail);
    }
    let window = open(cx, Rows(state.clone()));
    (state, window)
}

/// The scroll top of a list of rows, in pixels from the top of the content.
fn list_top(state: &ListState) -> f32 {
    let top = state.logical_scroll_top();
    top.item_ix as f32 * ROW + top.offset_in_item.0
}

fn lines(y: f32) -> ScrollDelta {
    ScrollDelta::Lines(point(0.0, y))
}

fn line_height<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext) -> f32 {
    window
        .update(cx, |_, window, _| window.line_height().0)
        .unwrap()
}

/// Dispatches a wheel event over the scroll state and draws.
fn wheel<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext, delta: ScrollDelta) {
    let event = ScrollWheelEvent {
        position: point(px(50.0), px(50.0)),
        delta,
        ..Default::default()
    };
    cx.test_window(**window)
        .simulate_input(event.to_platform_input());
    cx.run_until_parked();
}

/// Delivers frames until none is requested, asserting one frame per display
/// frame, and returns what `read` returns and the damage after each frame.
fn run_to_rest<V: 'static>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
    mut read: impl FnMut() -> f32,
) -> Vec<(f32, Option<Bounds<Pixels>>)> {
    let mut frames = Vec::new();
    loop {
        let delivered = next_frame(window, cx);
        if delivered == 0 {
            return frames;
        }
        assert_eq!(delivered, 1, "one frame per display frame while moving");
        frames.push((read(), last_damage(window, cx)));
        assert!(frames.len() < MAX_FRAMES, "the motion comes to rest");
    }
}

/// Asserts that `frames`, one per display frame after a tick from rest at
/// `from`, follow [`WHEEL_SPRING`] to `to` and end on `to`.
fn assert_eases(frames: &[(f32, Option<Bounds<Pixels>>)], from: f32, to: f32) {
    assert!(frames.len() > 5, "the motion took {} frames", frames.len());
    for (ix, (value, _)) in frames.iter().enumerate() {
        let (remaining, _) = spring_at(WHEEL_SPRING, to - from, 0.0, 0.0, ix as u32 + 1);
        let expected = to - remaining;
        assert!(
            (value - expected).abs() <= TOLERANCE,
            "frame {}: at {value}, the spring at {expected}",
            ix + 1
        );
    }
    let (last, _) = frames.last().unwrap();
    assert!((last - to).abs() <= 1e-3, "the motion ends on {to}, not {last}");
}

#[gpui::test]
fn a_line_tick_eases_a_scroll_handle_along_the_spring_to_its_target(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, true);
    let target = -3.0 * line_height(&window, cx);
    wheel(&window, cx, lines(-3.0));
    assert_eq!(handle.offset().y, px(0.0), "the tick moves nothing at once");

    let viewport = painted_bounds(&window, cx, SCROLLER).unwrap();
    let frames = run_to_rest(&window, cx, || handle.offset().y.0);
    assert_eases(&frames, 0.0, target);
    for (ix, (_, damage)) in frames.iter().enumerate() {
        let damage = damage.expect("a frame of the motion repaints a region");
        assert!(
            contains(damage, viewport) && contains(dilated(viewport), damage),
            "frame {} repainted {damage:?}, the viewport is {viewport:?}",
            ix + 1
        );
    }
    assert_eq!(idle_requests(&window, cx), 0, "at rest, no frame is requested");
}

#[gpui::test]
fn consecutive_ticks_accumulate_the_target_and_keep_the_velocity(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, true);
    let line = line_height(&window, cx);
    let mut offsets = vec![handle.offset().y.0];

    wheel(&window, cx, lines(-1.0));
    for _ in 0..2 {
        next_frame(&window, cx);
        offsets.push(handle.offset().y.0);
    }
    let (remaining, velocity) = spring_at(WHEEL_SPRING, -line, 0.0, 0.0, 2);
    wheel(&window, cx, lines(-1.0));
    next_frame(&window, cx);
    offsets.push(handle.offset().y.0);
    let (continued, _) = spring_at(WHEEL_SPRING, remaining - line, velocity, 0.0, 1);
    let expected = -2.0 * line - continued;
    assert!(
        (offsets[3] - expected).abs() <= TOLERANCE,
        "the second tick continues at the spring's velocity: at {}, expected {expected}",
        offsets[3]
    );

    // The test draws a tick at once, between display frames, so its frame
    // request queues beside the one the last frame made; both notify the view
    // for the same display frame, which draws once.
    wheel(&window, cx, lines(-1.0));
    assert!(next_frame(&window, cx) <= 2);
    offsets.push(handle.offset().y.0);
    offsets.extend(
        run_to_rest(&window, cx, || handle.offset().y.0)
            .into_iter()
            .map(|(y, _)| y),
    );
    for pair in offsets.windows(2) {
        assert!(pair[1] < pair[0], "the motion is monotonic: {offsets:?}");
    }
    let last = *offsets.last().unwrap();
    assert!(
        (last + 3.0 * line).abs() <= 1e-3,
        "three ticks land three lines down, not at {last}"
    );
}

#[gpui::test]
fn a_pixel_event_mid_motion_applies_exactly_and_ends_the_motion(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, true);
    let target = -3.0 * line_height(&window, cx);
    wheel(&window, cx, lines(-3.0));
    for _ in 0..3 {
        next_frame(&window, cx);
    }
    let painted = handle.offset().y;
    assert!(painted.0 < 0.0 && painted.0 > target, "mid-motion at {painted:?}");

    // A touchpad touch arrives as a pixel event with no distance.
    wheel(&window, cx, ScrollDelta::Pixels(point(px(0.0), px(0.0))));
    next_frame(&window, cx);
    assert_eq!(handle.offset().y, painted, "a touch ends the motion where it is");

    wheel(&window, cx, ScrollDelta::Pixels(point(px(0.0), px(-7.0))));
    assert_eq!(handle.offset().y, painted - px(7.0), "the pixel event applies at once");
    next_frame(&window, cx);
    assert_eq!(handle.offset().y, painted - px(7.0), "no motion resumes");
    assert_eq!(idle_requests(&window, cx), 0, "at rest, no frame is requested");
}

#[gpui::test]
fn a_tick_past_the_end_clamps_the_target_to_the_content(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, true);
    handle.set_offset(point(px(0.0), px(30.0 - MAX)));
    update_view(&window, cx, |_| {});
    wheel(&window, cx, lines(-10.0));

    let frames = run_to_rest(&window, cx, || handle.offset().y.0);
    assert_eases(&frames, 30.0 - MAX, -MAX);
    assert!(
        frames.iter().all(|(y, _)| *y >= -MAX),
        "the motion stays within the content: {frames:?}"
    );
}

#[gpui::test]
fn without_smooth_wheel_a_line_tick_applies_at_once(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, false);
    let target = -3.0 * line_height(&window, cx);
    wheel(&window, cx, lines(-3.0));
    assert_eq!(handle.offset().y.0, target);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn under_reduced_motion_a_line_tick_applies_at_once(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let (handle, window) = scroller(cx, true);
    let target = -3.0 * line_height(&window, cx);
    wheel(&window, cx, lines(-3.0));
    assert_eq!(handle.offset().y.0, target);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn the_target_offset_holds_where_a_motion_lands_for_every_frame_of_it(cx: &mut TestAppContext) {
    let (handle, window) = scroller(cx, true);
    let line = line_height(&window, cx);
    assert_eq!(handle.target_offset(), handle.offset(), "at rest");

    wheel(&window, cx, lines(-3.0));
    assert_eq!(handle.offset().y, px(0.0));
    assert_eq!(handle.target_offset().y.0, -3.0 * line, "a tick moves the target at once");
    next_frame(&window, cx);
    wheel(&window, cx, lines(-2.0));
    let target = -5.0 * line;
    assert!(
        (handle.target_offset().y.0 - target).abs() <= 1e-3,
        "a tick mid-motion moves the target on: {:?}, expected {target}",
        handle.target_offset()
    );
    // The tick's frame request queues beside the last frame's, and both
    // draw in one display frame.
    assert!(next_frame(&window, cx) <= 2);
    for (offset, _) in run_to_rest(&window, cx, || handle.offset().y.0) {
        let lead = handle.target_offset().y.0;
        assert!(
            offset >= target - TOLERANCE && (lead - target).abs() <= 1e-3,
            "at {offset} the target is {lead}, the motion lands on {target}"
        );
    }
    assert_eq!(handle.target_offset(), handle.offset(), "at rest again");

    // A tick past the end moves the target to the end; a write mid-motion
    // ends the motion, and the target is the written offset.
    wheel(&window, cx, lines(-100.0));
    assert_eq!(handle.target_offset().y.0, -MAX);
    next_frame(&window, cx);
    handle.set_offset(point(px(0.0), px(-10.0)));
    assert_eq!(handle.target_offset().y, px(-10.0));
    next_frame(&window, cx);
    assert_eq!(handle.offset().y, px(-10.0));
    assert_eq!(handle.target_offset().y, px(-10.0));

    let (handle, window) = scroller(cx, false);
    wheel(&window, cx, lines(-3.0));
    assert_eq!(handle.target_offset(), handle.offset(), "without smooth wheel");
}

#[gpui::test]
fn a_line_tick_eases_a_uniform_list(cx: &mut TestAppContext) {
    let handle = UniformListScrollHandle::new();
    handle.set_smooth_wheel(true);
    let window = open(cx, Uniform(handle.clone()));
    let target = -3.0 * line_height(&window, cx);
    let offset = || handle.0.borrow().base_handle.offset().y.0;
    wheel(&window, cx, lines(-3.0));
    assert_eq!(offset(), 0.0, "the tick moves nothing at once");

    let frames = run_to_rest(&window, cx, offset);
    assert_eases(&frames, 0.0, target);
    assert_eq!(idle_requests(&window, cx), 0, "at rest, no frame is requested");
}

#[gpui::test]
fn a_line_tick_eases_a_list(cx: &mut TestAppContext) {
    let (state, window) = rows(cx, false);
    wheel(&window, cx, lines(-3.0));
    assert_eq!(list_top(&state), 0.0, "the tick moves nothing at once");

    let frames = run_to_rest(&window, cx, || list_top(&state));
    assert_eases(&frames, 0.0, 60.0);
    assert_eq!(idle_requests(&window, cx), 0, "at rest, no frame is requested");
}

#[gpui::test]
fn a_tick_up_detaches_a_following_list_and_a_motion_to_the_end_reattaches_it(
    cx: &mut TestAppContext,
) {
    let (state, window) = rows(cx, true);
    assert!(state.is_following_tail());
    assert_eq!(list_top(&state), MAX);

    wheel(&window, cx, lines(2.0));
    assert!(!state.is_following_tail(), "a tick up detaches at once");
    let frames = run_to_rest(&window, cx, || list_top(&state));
    assert_eases(&frames, MAX, MAX - 40.0);
    assert!(!state.is_following_tail(), "the list stays detached above the end");

    wheel(&window, cx, lines(-5.0));
    let frames = run_to_rest(&window, cx, || list_top(&state));
    assert_eases(&frames, MAX - 40.0, MAX);
    assert!(state.is_following_tail(), "landing on the end reattaches");
}
