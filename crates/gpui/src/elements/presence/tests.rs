//! Presence sampled frame by frame on the test clock.
//!
//! WHY: closes the classes "an entering child pops in", "a removed child
//! vanishes before its exit runs", "an exited child is never dropped", "a
//! child re-added mid-exit restarts from rest or jumps", "siblings in layout
//! transitions jump when an exiting child drops", "a crossfade paints only
//! one child during the swap, or both side by side, or keeps the old one at
//! rest", "reduced motion still animates", and "a resting presence keeps
//! requesting frames". Opacity is read from the alpha of the quad each child
//! paints and compared with the closed-form spring at each frame instant.
//! Not caught: the scale effect and duration models (the tests run the
//! default rise and fade on the default spring), and hit testing of exiting
//! children.

use super::*;
use crate::{
    Context, Hsla, InteractiveElement as _, Render, TestAppContext, WindowHandle,
    elements::motion_harness::{
        MAX_FRAMES, contains, idle_requests, last_damage, next_frame, painted_bounds,
        snap_tolerance, spring_at, update_view,
    },
    layout_transition,
    motion::MotionPolicy,
    rgb, size,
};

/// Height of each child.
const ROW: f32 = 20.0;
/// The default rise of an entering child.
const RISE: f32 = 6.0;
/// How far a painted opacity may lie from the spring: the rest distance at
/// which the spring lands, and float rounding.
const OPACITY_TOLERANCE: f32 = 2e-3;

fn color(key: &str) -> Hsla {
    Hsla::from(rgb(match key {
        "a" => 0x2080f0,
        "b" => 0xf08020,
        _ => 0x20a060,
    }))
}

fn row(key: &'static str) -> Div {
    div()
        .debug_selector(move || key.into())
        .w(px(50.0))
        .h(px(ROW))
        .bg(color(key))
}

/// A column of rows under a presence, each wrapped in a layout transition
/// when `transitions` is set.
struct List {
    keys: Vec<&'static str>,
    transitions: bool,
}

impl Render for List {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let transitions = self.transitions;
        presence("list")
            .flex()
            .flex_col()
            .children(self.keys.iter().map(|&key| {
                (key, move |_: &mut Window, _: &mut App| {
                    if transitions {
                        layout_transition(key, row(key)).into_any_element()
                    } else {
                        row(key).into_any_element()
                    }
                })
            }))
    }
}

fn open(cx: &mut TestAppContext, keys: &[&'static str], transitions: bool) -> WindowHandle<List> {
    let keys = keys.to_vec();
    let window = cx.open_window(size(px(200.0), px(400.0)), move |_, _| List {
        keys: keys.clone(),
        transitions,
    });
    cx.run_until_parked();
    window
}

fn top<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext, key: &str) -> f32 {
    painted_bounds(window, cx, key)
        .unwrap_or_else(|| panic!("{key} was painted"))
        .origin
        .y
        .0
}

/// The alpha of the quad painted in the color of `key` in the last drawn
/// frame, or `None` when no such quad was painted.
fn opacity<V: 'static>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
    key: &str,
) -> Option<f32> {
    let color = color(key);
    window
        .update(cx, |_, window, _| {
            window.rendered_frame.scene.quads.iter().find_map(|quad| {
                let painted = quad.background.as_solid()?;
                (painted.h == color.h && painted.s == color.s && painted.l == color.l)
                    .then_some(painted.a)
            })
        })
        .unwrap()
}

fn assert_opacity<V: 'static>(
    window: &WindowHandle<V>,
    cx: &mut TestAppContext,
    key: &str,
    expected: f32,
    context: &str,
) {
    let painted = opacity(window, cx, key).unwrap_or(0.0);
    let expected = expected.clamp(0.0, 1.0);
    assert!(
        (painted - expected).abs() <= OPACITY_TOLERANCE,
        "{context}: {key} painted at opacity {painted}, expected {expected}"
    );
}

/// Draws frames until none is requested and returns how many were drawn.
fn settle<V: 'static>(window: &WindowHandle<V>, cx: &mut TestAppContext) -> usize {
    let mut frames = 0;
    while next_frame(window, cx) > 0 {
        frames += 1;
        assert!(frames < MAX_FRAMES, "the motion comes to rest");
    }
    frames
}

#[gpui::test]
fn an_entering_child_fades_in_and_rises_on_the_spring(cx: &mut TestAppContext) {
    let window = open(cx, &["a"], false);
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "a child present when the presence first paints appears at rest"
    );
    let tolerance = snap_tolerance(&window, cx);

    update_view(&window, cx, |list| list.keys.push("b"));
    assert_opacity(&window, cx, "b", 0.0, "the frame of the change");
    assert!(
        (top(&window, cx, "b") - (ROW + RISE)).abs() <= tolerance,
        "the frame of the change paints the child {RISE} px below its place"
    );

    let mut frames = 0;
    loop {
        let delivered = next_frame(&window, cx);
        if delivered == 0 {
            break;
        }
        assert_eq!(delivered, 1, "one frame per frame while entering");
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the enter comes to rest");
        let (progress, _) = spring_at(PRESENCE_SPRING, 0.0, 0.0, 1.0, frames);
        assert_opacity(&window, cx, "b", progress, &format!("frame {frames}"));
        let painted = top(&window, cx, "b");
        let expected = ROW + RISE * (1.0 - progress);
        assert!(
            (painted - expected).abs() <= tolerance,
            "frame {frames}: painted at {painted}, spring at {expected}"
        );
    }
    assert!(frames > 10, "the enter took {frames} frames");
    assert_eq!(
        opacity(&window, cx, "b"),
        Some(1.0),
        "the enter ends opaque"
    );
    assert_eq!(top(&window, cx, "b"), ROW, "the enter ends in place");
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn an_exiting_child_keeps_painting_until_its_exit_rests_then_drops(cx: &mut TestAppContext) {
    let window = open(cx, &["a", "b", "c"], false);
    let resting_c = painted_bounds(&window, cx, "c").unwrap();

    update_view(&window, cx, |list| list.keys.retain(|key| *key != "b"));
    assert_eq!(
        opacity(&window, cx, "b"),
        Some(1.0),
        "the frame of the removal paints the child as it was"
    );
    assert_eq!(
        top(&window, cx, "c"),
        2.0 * ROW,
        "the exiting child keeps its place"
    );

    let mut frames = 0;
    let mut painted_frames = 0;
    let mut dropped_at = None;
    loop {
        let delivered = next_frame(&window, cx);
        if delivered == 0 {
            break;
        }
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the exit comes to rest");
        assert!(dropped_at.is_none(), "no frame is requested after the drop");
        let (progress, velocity) = spring_at(PRESENCE_SPRING, 1.0, 0.0, 0.0, frames);
        if painted_bounds(&window, cx, "b").is_some() {
            painted_frames += 1;
            assert_opacity(&window, cx, "b", progress, &format!("frame {frames}"));
            assert_eq!(
                top(&window, cx, "c"),
                2.0 * ROW,
                "frame {frames}: c keeps its place"
            );
            continue;
        }
        dropped_at = Some(frames);
        assert!(
            progress.abs() <= UNIT_REST_DISTANCE && velocity.abs() <= 0.02,
            "frame {frames}: dropped at {progress} moving {velocity}/s, before the exit rests"
        );
        let moved_c = painted_bounds(&window, cx, "c").unwrap();
        assert_eq!(
            moved_c.origin.y.0, ROW,
            "the sibling takes the dropped place"
        );
        let swept = resting_c.union(&moved_c);
        assert!(
            last_damage(&window, cx).is_none_or(|damage| contains(damage, swept)),
            "the frame of the drop repaints where the sibling moved from and to"
        );
    }
    assert!(
        painted_frames > 10,
        "the exit painted {painted_frames} frames"
    );
    assert_eq!(
        dropped_at,
        Some(frames),
        "the child drops in the last frame"
    );
    assert!(painted_bounds(&window, cx, "b").is_none());
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn a_child_readded_mid_exit_reverses_from_its_value_and_velocity(cx: &mut TestAppContext) {
    const FIRST_LEG: u32 = 6;
    let window = open(cx, &["a", "b"], false);
    update_view(&window, cx, |list| list.keys.retain(|key| *key != "b"));
    for _ in 0..FIRST_LEG {
        next_frame(&window, cx);
    }
    let (value, velocity) = spring_at(PRESENCE_SPRING, 1.0, 0.0, 0.0, FIRST_LEG);
    assert!(velocity < -1.0, "the exit is mid-flight at {velocity}/s");
    assert_opacity(&window, cx, "b", value, "mid-exit");

    update_view(&window, cx, |list| list.keys.push("b"));
    assert_opacity(&window, cx, "b", value, "the frame of the re-add");
    for frame in 1..=12 {
        next_frame(&window, cx);
        let (expected, _) = spring_at(PRESENCE_SPRING, value, velocity, 1.0, frame);
        assert_opacity(
            &window,
            cx,
            "b",
            expected,
            &format!("frame {frame} after the re-add"),
        );
    }
    settle(&window, cx);
    assert_eq!(
        opacity(&window, cx, "b"),
        Some(1.0),
        "the reversal ends opaque"
    );
    assert_eq!(top(&window, cx, "b"), ROW, "the child keeps its place");
}

#[gpui::test]
fn siblings_in_layout_transitions_do_not_jump_when_an_exiting_child_drops(cx: &mut TestAppContext) {
    let window = open(cx, &["a", "b", "c"], true);
    let tolerance = snap_tolerance(&window, cx);
    update_view(&window, cx, |list| list.keys.retain(|key| *key != "b"));

    let mut previous = top(&window, cx, "c");
    let mut dropped = false;
    let mut frames = 0;
    while next_frame(&window, cx) > 0 {
        frames += 1;
        assert!(frames < MAX_FRAMES, "the motion comes to rest");
        let painted = top(&window, cx, "c");
        if !dropped && painted_bounds(&window, cx, "b").is_none() {
            dropped = true;
            assert!(
                (painted - previous).abs() <= tolerance,
                "the frame b drops paints c where it was: {previous} -> {painted}"
            );
        }
        previous = painted;
    }
    assert!(dropped, "b drops");
    assert_eq!(previous, ROW, "c ends in the place of b");
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn reduced_motion_adds_and_drops_children_in_the_frame_of_the_change(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = open(cx, &["a", "b"], false);
    update_view(&window, cx, |list| list.keys = vec!["a", "c"]);
    assert!(
        painted_bounds(&window, cx, "b").is_none(),
        "b drops at once"
    );
    assert_eq!(opacity(&window, cx, "c"), Some(1.0), "c enters opaque");
    assert_eq!(top(&window, cx, "c"), ROW, "c enters in place");
    assert_eq!(next_frame(&window, cx), 0, "no frame is requested");
}

/// A crossfade showing the row of `key`.
struct Swap {
    key: &'static str,
}

impl Render for Swap {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let key = self.key;
        crossfade("swap", key, move |_: &mut Window, _: &mut App| row(key))
    }
}

fn open_swap(cx: &mut TestAppContext, key: &'static str) -> WindowHandle<Swap> {
    let window = cx.open_window(size(px(200.0), px(400.0)), move |_, _| Swap { key });
    cx.run_until_parked();
    window
}

#[gpui::test]
fn a_crossfade_paints_both_children_stacked_during_the_swap_and_one_at_rest(
    cx: &mut TestAppContext,
) {
    let window = open_swap(cx, "a");
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "the first child appears at rest"
    );

    update_view(&window, cx, |swap| swap.key = "b");
    assert_opacity(&window, cx, "a", 1.0, "the frame of the swap");
    assert_opacity(&window, cx, "b", 0.0, "the frame of the swap");
    let stacked = |window: &WindowHandle<Swap>, cx: &mut TestAppContext, frame: u32| {
        let old = painted_bounds(window, cx, "a")?;
        let new = painted_bounds(window, cx, "b").expect("the new child is painted");
        assert_eq!(
            old.origin, new.origin,
            "frame {frame}: the children are stacked"
        );
        Some(())
    };
    assert!(
        stacked(&window, cx, 0).is_some(),
        "both children paint in the frame of the swap"
    );

    let mut frames = 0;
    let mut both = 0;
    loop {
        let delivered = next_frame(&window, cx);
        if delivered == 0 {
            break;
        }
        assert_eq!(delivered, 1, "one frame per frame while fading");
        frames += 1;
        assert!(frames < MAX_FRAMES as u32, "the crossfade comes to rest");
        let (incoming, _) = spring_at(PRESENCE_SPRING, 0.0, 0.0, 1.0, frames);
        assert_opacity(&window, cx, "b", incoming, &format!("frame {frames}"));
        if stacked(&window, cx, frames).is_some() {
            both += 1;
            let (outgoing, _) = spring_at(PRESENCE_SPRING, 1.0, 0.0, 0.0, frames);
            assert_opacity(&window, cx, "a", outgoing, &format!("frame {frames}"));
        }
    }
    assert!(both > 10, "both children painted for {both} frames");
    assert!(
        painted_bounds(&window, cx, "a").is_none(),
        "the old child drops at rest"
    );
    assert_eq!(
        opacity(&window, cx, "b"),
        Some(1.0),
        "the new child ends opaque"
    );
    assert_eq!(
        idle_requests(&window, cx),
        0,
        "at rest, no frame is requested"
    );
}

#[gpui::test]
fn reduced_motion_swaps_a_crossfade_in_the_frame_of_the_change(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_motion_policy(MotionPolicy::REDUCED));
    let window = open_swap(cx, "a");
    update_view(&window, cx, |swap| swap.key = "b");
    assert!(
        painted_bounds(&window, cx, "a").is_none(),
        "the old child drops at once"
    );
    assert_eq!(
        opacity(&window, cx, "b"),
        Some(1.0),
        "the new child shows opaque"
    );
    assert_eq!(next_frame(&window, cx), 0, "no frame is requested");
}
