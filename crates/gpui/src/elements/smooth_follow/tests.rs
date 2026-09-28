//! Smooth tail following sampled frame by frame on the test clock.
//!
//! WHY: closes the classes "content growing at the end of a following list
//! jumps instead of easing", "growth during a motion restarts it from rest or
//! drops the earlier growth", "a splice mid-motion jumps the list", "a wheel
//! event cannot interrupt the motion, or the list jumps when it does",
//! "entering follow mode or resetting the list eases", "reduced motion still
//! eases", "a list that did not opt in changes behavior", "a growth taller
//! than the viewport draws past the laid-out items", "pausing mid-motion
//! keeps requesting frames", "a reduced-motion layout reports a scroll top
//! above the one it shows", "a resting list keeps requesting frames", "a
//! spring carried past the end draws the list past its end", and "rows
//! arriving in an empty list ease from where nothing was shown".
//! The lag is read from where the last item was painted and compared with
//! the closed-form spring at each frame instant. Not caught: growth of an
//! item above the viewport of a following list, which the list does not
//! measure until it is shown.

use std::{cell::RefCell, rc::Rc};

use super::*;
use crate::{
    Context, FollowMode, InputEvent as _, InteractiveElement as _, IntoElement, ListAlignment,
    ListState, ParentElement as _, Render, ScrollDelta, ScrollWheelEvent, Styled as _,
    TestAppContext, WindowHandle, div,
    elements::motion_harness::{
        MAX_FRAMES, contains, dilated, idle_requests, last_damage, next_frame, painted_bounds,
        snap_tolerance, spring_at, update_view,
    },
    list,
    motion::MotionPolicy,
    point, size,
};

const VIEWPORT: f32 = 200.0;
const ROW: f32 = 50.0;
const ROWS: usize = 10;
const LAST: &str = "last";
const LIST: &str = "list";

struct Feed {
    state: ListState,
    heights: Rc<RefCell<Vec<f32>>>,
    width: f32,
}

impl Render for Feed {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let heights = self.heights.clone();
        div()
            .debug_selector(|| LIST.into())
            .w(px(self.width))
            .h(px(VIEWPORT))
            .child(
                list(self.state.clone(), move |ix, _, _| {
                    let heights = heights.borrow();
                    let row = div().w_full().h(px(heights[ix]));
                    if ix + 1 == heights.len() {
                        row.debug_selector(|| LAST.into()).into_any_element()
                    } else {
                        row.into_any_element()
                    }
                })
                .size_full(),
            )
    }
}

/// A list of [`ROWS`] rows following its tail, with smooth following when
/// `smooth` is set, drawn at its end.
fn feed(cx: &mut TestAppContext, smooth: bool) -> WindowHandle<Feed> {
    feed_aligned(cx, smooth, ListAlignment::Bottom)
}

/// [`feed`] with its items aligned by `alignment`.
fn feed_aligned(
    cx: &mut TestAppContext,
    smooth: bool,
    alignment: ListAlignment,
) -> WindowHandle<Feed> {
    feed_rows(cx, smooth, alignment, ROWS)
}

/// [`feed_aligned`] with `rows` rows.
fn feed_rows(
    cx: &mut TestAppContext,
    smooth: bool,
    alignment: ListAlignment,
    rows: usize,
) -> WindowHandle<Feed> {
    let state = ListState::new(rows, alignment, px(0.0));
    state.set_follow_mode(FollowMode::Tail);
    state.set_smooth_follow(smooth);
    let heights = Rc::new(RefCell::new(vec![ROW; rows]));
    let window = cx.open_window(size(px(VIEWPORT), px(400.0)), move |_, _| Feed {
        state,
        heights,
        width: VIEWPORT,
    });
    cx.run_until_parked();
    window
}

fn state(window: &WindowHandle<Feed>, cx: &mut TestAppContext) -> ListState {
    window.update(cx, |feed, _, _| feed.state.clone()).unwrap()
}

/// How far the end of the content was painted below the end of the viewport.
fn lag(window: &WindowHandle<Feed>, cx: &mut TestAppContext) -> f32 {
    painted_bounds(window, cx, LAST)
        .expect("the last row is painted")
        .bottom()
        .0
        - VIEWPORT
}

/// Grows the last row by `by` and draws.
fn grow(window: &WindowHandle<Feed>, cx: &mut TestAppContext, by: f32) {
    update_view(window, cx, |feed| {
        let last = {
            let mut heights = feed.heights.borrow_mut();
            let last = heights.len() - 1;
            heights[last] += by;
            last
        };
        feed.state.remeasure_items(last..last + 1);
    });
}

/// Replaces the rows in `range` with rows of `heights` and draws.
fn splice(
    window: &WindowHandle<Feed>,
    cx: &mut TestAppContext,
    range: std::ops::Range<usize>,
    heights: &[f32],
) {
    update_view(window, cx, |feed| {
        feed.heights
            .borrow_mut()
            .splice(range.clone(), heights.iter().copied());
        feed.state.splice(range, heights.len());
    });
}

fn wheel(window: &WindowHandle<Feed>, cx: &mut TestAppContext, y: f32) {
    let event = ScrollWheelEvent {
        position: point(px(50.0), px(50.0)),
        delta: ScrollDelta::Pixels(point(px(0.0), px(y))),
        ..Default::default()
    };
    cx.test_window(**window)
        .simulate_input(event.to_platform_input());
    cx.run_until_parked();
}

fn tolerance(window: &WindowHandle<Feed>, cx: &mut TestAppContext) -> f32 {
    PIXEL_REST_DISTANCE + snap_tolerance(window, cx)
}

/// Delivers frames until none is requested, asserting one frame per display
/// frame, and returns the painted lag and the damage after each frame.
fn run_to_rest(
    window: &WindowHandle<Feed>,
    cx: &mut TestAppContext,
) -> Vec<(f32, Option<Bounds<Pixels>>)> {
    let mut frames = Vec::new();
    loop {
        let delivered = next_frame(window, cx);
        if delivered == 0 {
            return frames;
        }
        assert_eq!(delivered, 1, "one frame per display frame while moving");
        frames.push((lag(window, cx), last_damage(window, cx)));
        assert!(frames.len() < MAX_FRAMES, "the motion comes to rest");
    }
}

/// Asserts that `frames`, one per display frame after the end moved `from`
/// below the viewport at rest, follow [`WHEEL_SPRING`] to the end.
fn assert_eases(
    window: &WindowHandle<Feed>,
    cx: &mut TestAppContext,
    frames: &[(f32, Option<Bounds<Pixels>>)],
    from: f32,
) {
    let tolerance = tolerance(window, cx);
    assert!(frames.len() > 5, "the motion took {} frames", frames.len());
    for (ix, (lag, _)) in frames.iter().enumerate() {
        let (expected, _) = spring_at(WHEEL_SPRING, from, 0.0, 0.0, ix as u32 + 1);
        assert!(
            (lag - expected).abs() <= tolerance,
            "frame {}: {lag} behind, the spring {expected}",
            ix + 1
        );
    }
    let (last, _) = frames.last().unwrap();
    assert!(
        last.abs() <= tolerance,
        "the motion ends at the end, {last} behind"
    );
}

#[gpui::test]
fn growth_at_the_end_eases_into_view_along_the_spring(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    assert!(
        lag(&window, cx).abs() <= tolerance,
        "the list opens at its end"
    );
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "a list at rest requests no frame"
    );

    grow(&window, cx, 40.0);
    assert!(
        (lag(&window, cx) - 40.0).abs() <= tolerance,
        "the frame that draws the growth moves nothing yet"
    );
    let viewport = painted_bounds(&window, cx, LIST).unwrap();
    let frames = run_to_rest(&window, cx);
    assert_eases(&window, cx, &frames, 40.0);
    for (ix, (_, damage)) in frames.iter().enumerate() {
        let damage = damage.expect("a frame of the motion repaints a region");
        assert!(
            contains(damage, viewport) && contains(dilated(viewport), damage),
            "frame {} repainted {damage:?}, the viewport is {viewport:?}",
            ix + 1
        );
    }
    assert!(state(&window, cx).is_following_tail());
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn an_arriving_row_eases_into_view(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    splice(&window, cx, ROWS..ROWS, &[60.0]);
    let frames = run_to_rest(&window, cx);
    assert_eases(&window, cx, &frames, 60.0);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn growth_during_the_motion_extends_it_and_keeps_its_velocity(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 40.0);
    next_frame(&window, cx);
    next_frame(&window, cx);
    let (behind, velocity) = spring_at(WHEEL_SPRING, 40.0, 0.0, 0.0, 2);

    grow(&window, cx, 30.0);
    assert!(
        (lag(&window, cx) - (behind + 30.0)).abs() <= tolerance,
        "the growth adds to the lag at once"
    );
    next_frame(&window, cx);
    let (expected, _) = spring_at(WHEEL_SPRING, behind + 30.0, velocity, 0.0, 1);
    let lag = lag(&window, cx);
    assert!(
        (lag - expected).abs() <= tolerance,
        "the motion continues at its velocity: {lag} behind, expected {expected}"
    );
    let frames = run_to_rest(&window, cx);
    assert!(frames.last().unwrap().0.abs() <= tolerance);
}

#[gpui::test]
fn splices_that_keep_the_shown_row_continue_the_motion(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    // The last row fills the viewport, so it is the row shown at the top.
    grow(&window, cx, VIEWPORT + ROW);
    run_to_rest(&window, cx);
    grow(&window, cx, 40.0);
    next_frame(&window, cx);
    next_frame(&window, cx);
    let (behind, velocity) = spring_at(WHEEL_SPRING, 40.0, 0.0, 0.0, 2);

    // Rows loaded above the view, and the shown row replaced in place by a
    // row of its height, as a streamed reply by its committed entry.
    splice(&window, cx, 0..0, &[ROW, ROW]);
    let last = ROWS + 1;
    splice(&window, cx, last..last + 1, &[VIEWPORT + 2.0 * ROW + 40.0]);
    assert!(
        (lag(&window, cx) - behind).abs() <= tolerance,
        "the splices move nothing at once"
    );
    next_frame(&window, cx);
    let (expected, _) = spring_at(WHEEL_SPRING, behind, velocity, 0.0, 1);
    let lag = lag(&window, cx);
    assert!(
        (lag - expected).abs() <= tolerance,
        "the motion continues: {lag} behind, expected {expected}"
    );
}

#[gpui::test]
fn removing_the_shown_row_ends_the_motion_at_the_end(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    // The lag outruns a row, so a shown index that kept naming the row after
    // the removed one would measure the end still behind and keep easing.
    grow(&window, cx, 3.0 * ROW);
    next_frame(&window, cx);
    assert!(
        lag(&window, cx) > ROW,
        "mid-motion, {} behind",
        lag(&window, cx)
    );
    let top = state(&window, cx).logical_scroll_top().item_ix;
    assert!(
        top < ROWS,
        "the scroll top names a row mid-motion, not {top}"
    );
    splice(&window, cx, top..top + 1, &[]);
    assert!(
        lag(&window, cx).abs() <= tolerance,
        "the list shows its end"
    );
    next_frame(&window, cx);
    assert!(lag(&window, cx).abs() <= tolerance, "and stays there");
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn a_wheel_event_ends_the_motion_where_it_is(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 40.0);
    next_frame(&window, cx);
    next_frame(&window, cx);
    let shown = lag(&window, cx);
    assert!(shown > 1.0, "mid-motion, {shown} behind");

    wheel(&window, cx, 5.0);
    next_frame(&window, cx);
    assert!(
        (lag(&window, cx) - (shown + 5.0)).abs() <= tolerance,
        "the wheel scrolls from where the list was shown"
    );
    assert!(!state(&window, cx).is_following_tail());
    assert_eq!(idle_requests(&window, cx), 0, "no motion resumes");
    assert!((lag(&window, cx) - (shown + 5.0)).abs() <= tolerance);
}

#[gpui::test]
fn entering_follow_mode_shows_the_end_at_once(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    update_view(&window, cx, |feed| {
        feed.state.scroll_to(ListOffset {
            item_ix: 0,
            offset_in_item: px(0.0),
        })
    });
    let top = state(&window, cx);
    assert_eq!(top.logical_scroll_top().item_ix, 0, "scrolled to the top");
    assert!(!top.is_following_tail());

    update_view(&window, cx, |feed| {
        feed.state.set_follow_mode(FollowMode::Tail)
    });
    assert!(lag(&window, cx).abs() <= tolerance, "the end shows at once");
    assert_eq!(idle_requests(&window, cx), 0);
}

/// Changes of a following list that keep it following, each of which ends a
/// follow motion.
const INTERRUPTS: [(&str, fn(&mut Feed)); 9] = [
    ("set_follow_mode", |feed| {
        feed.state.set_follow_mode(FollowMode::Tail)
    }),
    ("scroll_to_end", |feed| feed.state.scroll_to_end()),
    ("scroll_to past the end", |feed| {
        feed.state.scroll_to(ListOffset {
            item_ix: usize::MAX,
            offset_in_item: px(0.0),
        })
    }),
    ("scroll_by toward the end", |feed| {
        feed.state.scroll_by(px(10.0))
    }),
    ("scroll_to_reveal_item", |feed| {
        let last = feed.heights.borrow().len() - 1;
        feed.state.scroll_to_reveal_item(last);
    }),
    ("scrollbar dragged to the end", |feed| {
        let max = feed.state.max_offset_for_scrollbar().y;
        feed.state.set_offset_from_scrollbar(point(px(0.0), -max));
    }),
    ("remeasure", |feed| feed.state.remeasure()),
    ("reset", |feed| {
        let count = feed.heights.borrow().len();
        feed.state.reset(count);
    }),
    ("width", |feed| feed.width = VIEWPORT - ROW),
];

#[gpui::test]
fn every_other_change_of_the_scroll_position_ends_the_motion_at_the_end(cx: &mut TestAppContext) {
    for (name, interrupt) in INTERRUPTS {
        let window = feed(cx, true);
        let tolerance = tolerance(&window, cx);
        grow(&window, cx, 40.0);
        next_frame(&window, cx);
        assert!(lag(&window, cx) > 1.0, "{name}: mid-motion");

        update_view(&window, cx, interrupt);
        assert!(
            lag(&window, cx).abs() <= tolerance,
            "{name}: the end shows at once, not {} behind",
            lag(&window, cx)
        );
        assert!(
            state(&window, cx).is_following_tail(),
            "{name}: still following"
        );
        next_frame(&window, cx);
        assert!(
            lag(&window, cx).abs() <= tolerance,
            "{name}: the end stays shown"
        );
        assert_eq!(idle_requests(&window, cx), 0, "{name}: at rest");
    }
}

#[gpui::test]
fn without_smooth_follow_growth_shows_at_once(cx: &mut TestAppContext) {
    let window = feed(cx, false);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 40.0);
    assert!(lag(&window, cx).abs() <= tolerance);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn under_reduced_motion_growth_shows_at_once(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 40.0);
    assert!(lag(&window, cx).abs() <= tolerance);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn under_reduced_motion_the_scroll_top_is_the_row_shown_at_the_top(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = feed_aligned(cx, true, ListAlignment::Top);
    // The last row grows by more than two rows, so the rows shown are the
    // last two and a part of the one before.
    grow(&window, cx, 2.0 * ROW + 20.0);
    assert!(lag(&window, cx).abs() <= tolerance(&window, cx));
    let visible = ListOffset {
        item_ix: ROWS - 2,
        offset_in_item: px(20.0),
    };
    assert_eq!(state(&window, cx).logical_scroll_top(), visible);
}

#[gpui::test]
fn pausing_mid_motion_keeps_the_list_where_it_was_drawn(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 3.0 * ROW);
    next_frame(&window, cx);
    let shown = lag(&window, cx);
    assert!(shown > 1.0, "mid-motion, {shown} behind");

    update_view(&window, cx, |feed| feed.state.pause_following_tail());
    assert!(
        (lag(&window, cx) - shown).abs() <= tolerance,
        "the list stays put"
    );
    // The mid-motion draw asked for a frame before the pause; it is
    // delivered once, and nothing asks for another.
    next_frame(&window, cx);
    assert!(
        (lag(&window, cx) - shown).abs() <= tolerance,
        "that frame draws it in place"
    );
    assert_eq!(idle_requests(&window, cx), 0, "no motion continues");
    assert!(
        (lag(&window, cx) - shown).abs() <= tolerance,
        "and stays there"
    );
    assert!(!state(&window, cx).is_following_tail());
}

#[gpui::test]
fn growth_taller_than_the_viewport_eases_from_laid_out_rows(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    splice(&window, cx, ROWS..ROWS, &[3.0 * VIEWPORT]);
    let first = lag(&window, cx);
    let top = painted_bounds(&window, cx, LAST).unwrap().top().0;
    assert!(
        first >= VIEWPORT && top <= tolerance,
        "the view starts within the new row: {first} behind, its top at {top}"
    );
    let frames = run_to_rest(&window, cx);
    let mut lags = vec![first];
    lags.extend(frames.iter().map(|(lag, _)| *lag));
    for pair in lags.windows(2) {
        assert!(
            pair[1] <= pair[0] + tolerance,
            "the motion is monotonic: {lags:?}"
        );
    }
    assert!(lags.last().unwrap().abs() <= tolerance);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn content_that_shrinks_mid_motion_never_draws_the_list_past_its_end(cx: &mut TestAppContext) {
    let window = feed(cx, true);
    let tolerance = tolerance(&window, cx);
    grow(&window, cx, 4.0 * ROW);
    next_frame(&window, cx);
    next_frame(&window, cx);
    let behind = lag(&window, cx);
    assert!(behind > ROW, "mid-motion, {behind} behind");
    // The end moves up to just below the viewport while the list moves
    // toward it fast enough that the spring alone would carry it past.
    grow(&window, cx, 2.0 - behind);
    let mut lags = vec![lag(&window, cx)];
    // The draw the change caused asked for a frame beside the pending one.
    next_frame(&window, cx);
    lags.push(lag(&window, cx));
    lags.extend(run_to_rest(&window, cx).iter().map(|(lag, _)| *lag));
    assert!(
        lags.iter().all(|lag| *lag >= -tolerance),
        "the list is never drawn past its end: {lags:?}"
    );
    assert!(lags.last().unwrap().abs() <= tolerance);
    assert_eq!(idle_requests(&window, cx), 0);
}

#[gpui::test]
fn rows_arriving_in_an_empty_list_show_their_end_at_once(cx: &mut TestAppContext) {
    let window = feed_rows(cx, true, ListAlignment::Bottom, 0);
    let tolerance = tolerance(&window, cx);
    splice(&window, cx, 0..0, &[ROW; ROWS]);
    let behind = lag(&window, cx);
    assert!(
        behind.abs() <= tolerance,
        "the list shows its end, {behind} behind"
    );
    assert_eq!(idle_requests(&window, cx), 0);
}
