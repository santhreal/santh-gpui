//! WHY: a window runs frames at the refresh rate of the CRTC that shows the
//! largest part of it. The class closed here is a window paced by another
//! CRTC: by the first CRTC RandR lists whatever the window's position (a
//! window on a 60 Hz monitor beside a 240 Hz one drew four frames per
//! displayed frame and, under Fifo, blocked on every one), by a CRTC that
//! shows a smaller part of a window that straddles two, or by a CRTC that
//! shows none of it. Every case is derived from the CRTCs of each layout.
//! Not covered: the RandR queries and the configure and screen-change events
//! that re-read them.

use std::time::Duration;

use gpui::{Bounds, point, size};

use super::{CrtcRate, interval_at};

fn rect(x: i32, y: i32, width: i32, height: i32) -> Bounds<i32> {
    Bounds {
        origin: point(x, y),
        size: size(width, height),
    }
}

fn crtc(bounds: Bounds<i32>, hz: u64) -> CrtcRate {
    CrtcRate {
        bounds,
        interval: Duration::from_micros(1_000_000 / hz),
    }
}

/// CRTC layouts, each listing a fast CRTC first: side by side at different
/// heights, stacked, offset with a gap between them, and three in a row.
fn layouts() -> Vec<Vec<CrtcRate>> {
    vec![
        vec![
            crtc(rect(0, 0, 1920, 1080), 240),
            crtc(rect(1920, 0, 3840, 2160), 60),
        ],
        vec![
            crtc(rect(0, 0, 2560, 1440), 144),
            crtc(rect(0, 1440, 2560, 1440), 60),
        ],
        vec![
            crtc(rect(0, 500, 1920, 1080), 165),
            crtc(rect(2000, 0, 1280, 1024), 75),
        ],
        vec![
            crtc(rect(0, 0, 1920, 1080), 240),
            crtc(rect(1920, 0, 1920, 1080), 60),
            crtc(rect(3840, 0, 1920, 1080), 120),
        ],
    ]
}

/// A rectangle inside `bounds`, `inset` device pixels from each edge.
fn inside(bounds: Bounds<i32>, inset: i32) -> Bounds<i32> {
    rect(
        bounds.origin.x + inset,
        bounds.origin.y + inset,
        bounds.size.width - 2 * inset,
        bounds.size.height - 2 * inset,
    )
}

#[test]
fn a_window_inside_a_crtc_runs_at_that_crtcs_rate() {
    for crtcs in layouts() {
        for shown in &crtcs {
            for window in [
                shown.bounds,
                inside(shown.bounds, 1),
                inside(shown.bounds, 200),
            ] {
                assert_eq!(
                    interval_at(&crtcs, window),
                    Some(shown.interval),
                    "{window:?} on {crtcs:?}"
                );
            }
        }
    }
}

#[test]
fn a_window_across_two_crtcs_runs_at_the_rate_of_the_one_that_shows_more() {
    for crtcs in layouts() {
        for (index, left) in crtcs.iter().enumerate() {
            let Some(right) = crtcs
                .iter()
                .find(|other| other.bounds.origin.x == left.bounds.right())
            else {
                continue;
            };
            // A window 1000 device pixels wide on the rows both CRTCs show,
            // with `on_left` of its columns on the left CRTC.
            let top = left.bounds.origin.y.max(right.bounds.origin.y);
            let bottom = left.bounds.bottom().min(right.bounds.bottom());
            for (on_left, expected) in [
                (999, left.interval),
                (501, left.interval),
                (499, right.interval),
                (1, right.interval),
            ] {
                let window = rect(left.bounds.right() - on_left, top, 1000, bottom - top);
                assert_eq!(
                    interval_at(&crtcs, window),
                    Some(expected),
                    "{on_left} of 1000 columns on CRTC {index} of {crtcs:?}"
                );
            }
        }
    }
}

#[test]
fn an_even_split_and_a_window_no_crtc_shows_run_at_the_first_crtcs_rate() {
    for mut crtcs in layouts() {
        for _ in 0..crtcs.len() {
            let first = crtcs[0];
            assert_eq!(
                interval_at(&crtcs, rect(-5000, -5000, 800, 600)),
                Some(first.interval),
                "a window off every CRTC of {crtcs:?}"
            );
            if let Some(right) = crtcs
                .iter()
                .find(|other| other.bounds.origin.x == first.bounds.right())
            {
                let top = first.bounds.origin.y.max(right.bounds.origin.y);
                let window = rect(first.bounds.right() - 500, top, 1000, 400);
                assert_eq!(
                    interval_at(&crtcs, window),
                    Some(first.interval),
                    "a window split evenly on {crtcs:?}"
                );
            }
            crtcs.rotate_left(1);
        }
    }
}

#[test]
fn without_crtcs_there_is_no_interval() {
    assert_eq!(interval_at(&[], rect(0, 0, 800, 600)), None);
}
