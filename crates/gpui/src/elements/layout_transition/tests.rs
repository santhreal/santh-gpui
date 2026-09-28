//! Layout transitions sampled frame by frame on the test clock.
//!
//! WHY: closes the classes "a moved element jumps to its new position", "a
//! move interrupted mid-flight restarts from rest or from the wrong place",
//! "a moving element repaints more than the region it moves through", "a
//! resting element keeps requesting frames", "reduced motion still animates",
//! "an element nested in a moving element moves twice", and "scrolling or
//! resizing animates what only the viewport moved". Positions are compared
//! with the closed-form spring at each frame instant. Not caught: a nested
//! transition laid out in a separate layout tree (a list item), and elements
//! that paint outside their own primitives' bounds.

use super::*;
use crate::{
    Context, Hsla, InteractiveElement as _, IntoElement, MotionExt as _, Render, ScrollHandle,
    StatefulInteractiveElement as _, Styled as _, TestAppContext, Window, WindowHandle, div,
    elements::motion_harness::{
        MAX_FRAMES, contains, dilated, idle_requests, last_damage, next_frame, painted_bounds,
        snap_tolerance, spring_at, update_view,
    },
    motion::MotionPolicy,
    rgb, size,
};

const MOVING: &str = "moving";
const PARENT: &str = "parent";
/// Height of the header inside the nested parent above the nested child.
const HEADER: f32 = 10.0;

fn moving_box() -> crate::Stateful<crate::Div> {
    div()
        .id("box")
        .debug_selector(|| MOVING.into())
        .w(px(50.0))
        .h(px(20.0))
        .bg(Hsla::from(rgb(0x2080f0)))
}

/// A column: a spacer of height `above`, then the moving box. With `nested`,
/// the box sits under a header inside a parent that transitions too.
struct Column {
    above: f32,
    nested: bool,
}

impl Render for Column {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let spacer = div().w(px(50.0)).h(px(self.above));
        let column = div().flex().flex_col().child(spacer);
        if self.nested {
            column.child(
                div()
                    .id("parent-box")
                    .debug_selector(|| PARENT.into())
                    .flex()
                    .flex_col()
                    .child(div().h(px(HEADER)))
                    .child(moving_box().animate_layout("moving"))
                    .animate_layout("parent"),
            )
        } else {
            column.child(moving_box().animate_layout("moving"))
        }
    }
}

fn open(cx: &mut TestAppContext, above: f32, nested: bool) -> WindowHandle<Column> {
    let window = cx.open_window(size(px(200.0), px(400.0)), move |_, _| Column {
        above,
        nested,
    });
    cx.run_until_parked();
    window
}

fn top<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext, selector: &str) -> f32 {
    painted_bounds(window, cx, selector)
        .unwrap_or_else(|| panic!("{selector} was painted"))
        .origin
        .y
        .0
}

#[gpui::test]
fn a_moved_element_travels_on_the_spring_from_where_it_was_painted(cx: &mut TestAppContext) {
    let window = open(cx, 20.0, false);
    let tolerance = snap_tolerance(&window, cx);
    assert_eq!(top(&window, cx, MOVING), 20.0);

    update_view(&window, cx, |column| column.above = 100.0);
    assert!(
        (top(&window, cx, MOVING) - 20.0).abs() <= tolerance,
        "the frame of the change paints the element where it was"
    );

    let mut previous = painted_bounds(&window, cx, MOVING).unwrap();
    let mut frames = 0;
    loop {
        let delivered = next_frame(&window, cx);
        if delivered == 0 {
            break;
        }
        assert_eq!(delivered, 1, "one frame per frame while moving");
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the motion comes to rest");

        let bounds = painted_bounds(&window, cx, MOVING).unwrap();
        let (offset, _) = spring_at(LAYOUT_SPRING, -80.0, 0.0, 0.0, frames);
        assert!(
            (bounds.origin.y.0 - (100.0 + offset)).abs() <= tolerance,
            "frame {frames}: painted at {:?}, spring at {}",
            bounds.origin.y,
            100.0 + offset
        );
        let damage = last_damage(&window, cx).expect("a frame of the motion repaints a region");
        let swept = previous.union(&bounds);
        assert!(
            contains(damage, swept) && contains(dilated(swept), damage),
            "frame {frames} repainted {damage:?}, the element swept {swept:?}"
        );
        previous = bounds;
    }
    assert!(frames > 10, "the motion took {frames} frames");
    assert_eq!(previous.origin.y, px(100.0), "the motion ends on the layout");
    assert_eq!(idle_requests(&window, cx), 0, "at rest, no frame is requested");
}

#[gpui::test]
fn an_interrupted_move_continues_from_its_position_and_velocity(cx: &mut TestAppContext) {
    const FIRST_LEG: u32 = 5;
    let window = open(cx, 20.0, false);
    let tolerance = snap_tolerance(&window, cx);
    update_view(&window, cx, |column| column.above = 100.0);
    for _ in 0..FIRST_LEG {
        next_frame(&window, cx);
    }
    let (offset, velocity) = spring_at(LAYOUT_SPRING, -80.0, 0.0, 0.0, FIRST_LEG);
    assert!(velocity > 100.0, "the first leg is mid-flight at {velocity} px/s");
    let before = top(&window, cx, MOVING);

    update_view(&window, cx, |column| column.above = 40.0);
    assert!(
        (top(&window, cx, MOVING) - before).abs() <= 2.0 * tolerance,
        "the frame of the interruption paints the element where it was"
    );

    // The layout moved up by 60, so the displacement grew by 60 and the
    // velocity carried over.
    for frame in 1..=12 {
        next_frame(&window, cx);
        let (expected, _) = spring_at(LAYOUT_SPRING, offset + 60.0, velocity, 0.0, frame);
        let painted = top(&window, cx, MOVING);
        assert!(
            (painted - (40.0 + expected)).abs() <= tolerance,
            "frame {frame} after the interruption: painted at {painted}, spring at {}",
            40.0 + expected
        );
    }
}

#[gpui::test]
fn reduced_motion_places_a_moved_element_in_the_frame_it_moves(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = open(cx, 20.0, false);
    update_view(&window, cx, |column| column.above = 100.0);
    assert_eq!(top(&window, cx, MOVING), 100.0);
    assert_eq!(next_frame(&window, cx), 0, "no frame is requested");
}

#[gpui::test]
fn an_element_nested_in_a_moving_transition_moves_with_it_once(cx: &mut TestAppContext) {
    let window = open(cx, 20.0, true);
    update_view(&window, cx, |column| column.above = 100.0);
    let mut frames = 0;
    let mut moved = false;
    loop {
        let parent = top(&window, cx, PARENT);
        let child = top(&window, cx, MOVING);
        assert_eq!(
            child - parent,
            HEADER,
            "frame {frames}: the child keeps its place in the parent"
        );
        moved |= parent != 100.0;
        if next_frame(&window, cx) == 0 {
            break;
        }
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the motion comes to rest");
    }
    assert!(moved, "the parent moved");
    assert_eq!(top(&window, cx, PARENT), 100.0);
}

/// A scroll container holding a spacer and the moving box.
struct Scrolled {
    handle: ScrollHandle,
}

impl Render for Scrolled {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("scroller")
            .overflow_y_scroll()
            .track_scroll(&self.handle)
            .w(px(100.0))
            .h(px(100.0))
            .child(div().h(px(40.0)))
            .child(moving_box().animate_layout("moving"))
            .child(div().h(px(300.0)))
    }
}

#[gpui::test]
fn scrolling_an_ancestor_moves_the_element_at_once(cx: &mut TestAppContext) {
    let handle = ScrollHandle::new();
    let window = cx.open_window(size(px(200.0), px(200.0)), {
        let handle = handle.clone();
        move |_, _| Scrolled { handle }
    });
    cx.run_until_parked();
    assert_eq!(top(&window, cx, MOVING), 40.0);

    handle.set_offset(crate::point(px(0.0), px(-30.0)));
    update_view(&window, cx, |_| {});
    assert_eq!(top(&window, cx, MOVING), 10.0);
    assert_eq!(next_frame(&window, cx), 0, "scrolling requests no frame");
}

#[gpui::test]
fn resizing_the_window_moves_the_element_at_once(cx: &mut TestAppContext) {
    let window = open(cx, 20.0, false);
    window
        .update(cx, |column, _, _| column.above = 100.0)
        .unwrap();
    cx.simulate_window_resize(window.into(), size(px(300.0), px(400.0)));
    cx.run_until_parked();
    assert_eq!(top(&window, cx, MOVING), 100.0);
    assert_eq!(next_frame(&window, cx), 0, "a resize requests no frame");
}
