//! Height reveals sampled frame by frame on the test clock.
//!
//! WHY: closes the classes "a revealed element jumps to its height", "a
//! reversed or retargeted reveal restarts from rest or from the wrong
//! height", "content that grows while expanded pushes what follows at once",
//! "the child is squeezed or reflowed to the animated height", "a collapsed
//! reveal still takes space or paints its child", "a resting reveal keeps
//! requesting frames", and "reduced motion still animates". Heights are read
//! from the position of the element laid out after the reveal and compared
//! with the closed-form spring at each frame instant. Not caught: a reveal
//! inside a list item laid out in its own layout tree, and a child whose
//! height depends on the height of the reveal.

use super::*;
use crate::{
    Context, Hsla, InteractiveElement as _, IntoElement, ParentElement as _, Render, Styled as _,
    TestAppContext, Window, WindowHandle, div,
    elements::motion_harness::{
        MAX_FRAMES, contains, dilated, idle_requests, last_damage, next_frame, painted_bounds,
        snap_tolerance, spring_at, update_view,
    },
    motion::MotionPolicy,
    point, rgb, size,
};

const CONTENT: &str = "content";
const BELOW: &str = "below";

/// A column: the reveal of a box `content` pixels tall, then a box whose top
/// is the height of the reveal.
struct Section {
    expanded: bool,
    content: f32,
}

impl Render for Section {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let content = div()
            .debug_selector(|| CONTENT.into())
            .w(px(50.0))
            .h(px(self.content))
            .bg(Hsla::from(rgb(0x2080f0)));
        let below = div()
            .debug_selector(|| BELOW.into())
            .w(px(50.0))
            .h(px(10.0))
            .bg(Hsla::from(rgb(0xf08020)));
        div()
            .flex()
            .flex_col()
            .child(height_reveal("reveal", self.expanded, content))
            .child(below)
    }
}

fn open(cx: &mut TestAppContext, expanded: bool, content: f32) -> WindowHandle<Section> {
    let window = cx.open_window(size(px(200.0), px(400.0)), move |_, _| Section {
        expanded,
        content,
    });
    cx.run_until_parked();
    window
}

/// The height of the reveal in the last drawn frame.
fn height(window: &WindowHandle<Section>, cx: &mut TestAppContext) -> f32 {
    painted_bounds(window, cx, BELOW)
        .expect("the box below the reveal is painted")
        .origin
        .y
        .0
}

/// The union of every primitive of the last drawn frame, in logical pixels.
fn painted_region(window: &WindowHandle<Section>, cx: &mut TestAppContext) -> Bounds<Pixels> {
    window
        .update(cx, |_, window, _| {
            let scale = window.scale_factor();
            let bounds = window
                .rendered_frame
                .scene
                .painted_bounds_since(0)
                .expect("the frame paints");
            let far = bounds.bottom_right();
            Bounds::from_corners(
                point(px(bounds.origin.x.0 / scale), px(bounds.origin.y.0 / scale)),
                point(px(far.x.0 / scale), px(far.y.0 / scale)),
            )
        })
        .unwrap()
}

/// Delivers frames until none is requested and returns how many were.
fn settle(window: &WindowHandle<Section>, cx: &mut TestAppContext) -> usize {
    let mut frames = 0;
    while next_frame(window, cx) > 0 {
        frames += 1;
        assert!(frames < MAX_FRAMES, "the motion comes to rest");
    }
    frames
}

#[gpui::test]
fn an_expanded_reveal_grows_on_the_spring_with_its_child_at_natural_height(
    cx: &mut TestAppContext,
) {
    let window = open(cx, false, 100.0);
    let tolerance = snap_tolerance(&window, cx);
    assert_eq!(height(&window, cx), 0.0);

    update_view(&window, cx, |section| section.expanded = true);
    assert_eq!(
        height(&window, cx),
        0.0,
        "the frame of the change keeps the height"
    );

    let mut previous = painted_bounds(&window, cx, BELOW).unwrap();
    let mut frames = 0;
    loop {
        let delivered = next_frame(&window, cx);
        if delivered == 0 {
            break;
        }
        assert_eq!(delivered, 1, "one frame per frame while moving");
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the motion comes to rest");

        let (offset, _) = spring_at(LAYOUT_SPRING, -100.0, 0.0, 0.0, frames);
        let expected = (100.0 + offset).max(0.0);
        let painted = height(&window, cx);
        assert!(
            (painted - expected).abs() <= tolerance,
            "frame {frames}: height {painted}, spring at {expected}"
        );
        let content = painted_bounds(&window, cx, CONTENT).expect("the child is painted");
        assert_eq!(
            content.size.height,
            px(100.0),
            "frame {frames}: the child keeps its natural height"
        );
        let below = painted_bounds(&window, cx, BELOW).unwrap();
        let swept = previous.union(&below);
        assert!(
            last_damage(&window, cx).is_none_or(|damage| contains(damage, swept)),
            "frame {frames}: the repaint covers the box the height moved"
        );
        previous = below;
    }
    assert!(frames > 10, "the motion took {frames} frames");
    assert_eq!(
        height(&window, cx),
        100.0,
        "the motion ends on the natural height"
    );
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn a_collapsed_reveal_at_rest_takes_no_height_and_paints_nothing(cx: &mut TestAppContext) {
    let window = open(cx, true, 100.0);
    assert_eq!(
        height(&window, cx),
        100.0,
        "the first frame is at the natural height"
    );
    assert_eq!(
        next_frame(&window, cx),
        0,
        "the first frame does not animate"
    );

    update_view(&window, cx, |section| section.expanded = false);
    assert!(settle(&window, cx) > 10, "the collapse animates");
    assert_eq!(height(&window, cx), 0.0);
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );

    assert_eq!(height(&window, cx), 0.0);
    assert!(
        painted_bounds(&window, cx, CONTENT).is_none(),
        "the child is not painted"
    );
    let below = painted_bounds(&window, cx, BELOW).unwrap();
    let painted = painted_region(&window, cx);
    assert!(
        contains(dilated(below), painted),
        "only the box below paints: {painted:?}"
    );
}

#[gpui::test]
fn an_interrupted_collapse_reverses_with_its_position_and_velocity(cx: &mut TestAppContext) {
    const FIRST_LEG: u32 = 5;
    let window = open(cx, true, 100.0);
    let tolerance = snap_tolerance(&window, cx);
    update_view(&window, cx, |section| section.expanded = false);
    for _ in 0..FIRST_LEG {
        next_frame(&window, cx);
    }
    let (offset, velocity) = spring_at(LAYOUT_SPRING, 100.0, 0.0, 0.0, FIRST_LEG);
    assert!(
        velocity < -100.0,
        "the collapse is mid-flight at {velocity} px/s"
    );
    let before = height(&window, cx);
    assert!((before - offset).abs() <= tolerance);

    update_view(&window, cx, |section| section.expanded = true);
    assert!(
        (height(&window, cx) - before).abs() <= 2.0 * tolerance,
        "the frame of the interruption keeps the height"
    );

    // The target moved up by the natural height; the velocity carried over.
    let mut heights = Vec::new();
    for frame in 1..=12 {
        next_frame(&window, cx);
        let (expected, _) = spring_at(LAYOUT_SPRING, offset - 100.0, velocity, 0.0, frame);
        let painted = height(&window, cx);
        assert!(
            (painted - (100.0 + expected)).abs() <= tolerance,
            "frame {frame} after the interruption: height {painted}, spring at {}",
            100.0 + expected
        );
        heights.push(painted);
    }
    assert!(
        heights[0] < before,
        "the collapse's momentum carries into the first frame of the reversal"
    );
    assert!(heights[11] > before, "the reveal reverses");
}

#[gpui::test]
fn content_that_grows_while_expanded_moves_the_height_on_the_spring(cx: &mut TestAppContext) {
    let window = open(cx, true, 100.0);
    let tolerance = snap_tolerance(&window, cx);

    update_view(&window, cx, |section| section.content = 160.0);
    assert_eq!(
        height(&window, cx),
        100.0,
        "the frame of the growth keeps the height"
    );
    assert_eq!(
        painted_bounds(&window, cx, CONTENT).unwrap().size.height,
        px(160.0),
        "the child is laid out at its new natural height at once"
    );

    let mut frames = 0;
    loop {
        let below = painted_bounds(&window, cx, BELOW).unwrap();
        assert!(
            painted_region(&window, cx).bottom() <= below.bottom() + px(1.0),
            "frame {frames}: the child is clipped to the height of the reveal"
        );
        if next_frame(&window, cx) == 0 {
            break;
        }
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the motion comes to rest");
        let (offset, _) = spring_at(LAYOUT_SPRING, -60.0, 0.0, 0.0, frames);
        let painted = height(&window, cx);
        assert!(
            (painted - (160.0 + offset)).abs() <= tolerance,
            "frame {frames}: height {painted}, spring at {}",
            160.0 + offset
        );
    }
    assert!(frames > 10, "the growth took {frames} frames");
    assert_eq!(height(&window, cx), 160.0);
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn reduced_motion_places_the_height_in_the_frame_it_changes(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = open(cx, false, 100.0);

    update_view(&window, cx, |section| section.expanded = true);
    assert_eq!(height(&window, cx), 100.0);
    assert_eq!(next_frame(&window, cx), 0, "expanding requests no frame");

    update_view(&window, cx, |section| section.content = 160.0);
    assert_eq!(height(&window, cx), 160.0);
    assert_eq!(next_frame(&window, cx), 0, "growing requests no frame");

    update_view(&window, cx, |section| section.expanded = false);
    assert_eq!(height(&window, cx), 0.0);
    assert!(painted_bounds(&window, cx, CONTENT).is_none());
    assert_eq!(next_frame(&window, cx), 0, "collapsing requests no frame");
}
