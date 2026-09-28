//! Smooth tail following: a list that follows its tail moves to content that
//! grows at its end on a spring instead of jumping to it.

use std::ops::Range;

use crate::{
    App, Bounds, ListOffset, Pixels, Window,
    motion::{ElementSpring, PIXEL_REST_DISTANCE},
    px,
};

use super::smooth_wheel::WHEEL_SPRING;

#[cfg(test)]
mod tests;

/// The follow motion of one list.
///
/// `lag` is how far the end of the content lies below the end of the
/// viewport, carried to zero on [`WHEEL_SPRING`]. `shown` is the scroll top
/// the last layout that followed the tail displayed. The next following
/// layout measures from it how far the end moved since, adds that distance
/// to the lag at the frame instant and keeps the spring's velocity, so
/// content that keeps arriving moves on without a stall.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FollowEase {
    lag: ElementSpring,
    shown: Option<ListOffset>,
}

impl FollowEase {
    /// A motion at rest with nothing shown.
    pub(crate) fn new() -> Self {
        Self {
            lag: ElementSpring::at_rest(0.0, WHEEL_SPRING, PIXEL_REST_DISTANCE),
            shown: None,
        }
    }

    /// The scroll top the last following layout displayed, if the list has
    /// followed since the motion last ended.
    pub(crate) fn shown(&self) -> Option<ListOffset> {
        self.shown
    }

    /// Records `shown` as the scroll top a following layout displayed.
    pub(crate) fn record(&mut self, shown: ListOffset) {
        self.shown = Some(shown);
    }

    /// Moves the shown scroll top with a splice that replaced the items in
    /// `old_range` with `count` items. An item inserted at the shown index
    /// goes before it. A shown item the splice replaced keeps its index when
    /// a new item takes that index, and ends the motion when none does.
    pub(crate) fn splice(&mut self, old_range: &Range<usize>, count: usize) {
        let Some(shown) = self.shown.as_mut() else {
            return;
        };
        if old_range.end <= shown.item_ix {
            shown.item_ix = shown.item_ix - old_range.len() + count;
        } else if old_range.start <= shown.item_ix && shown.item_ix - old_range.start >= count {
            self.cancel();
        }
    }

    /// Samples the lag of a following layout at the current frame instant.
    ///
    /// `behind` is how far the end of the content lies below the end of the
    /// viewport when the list displays [`Self::shown`] again, as laid out
    /// this frame. The distance the end moved since the last sample is added
    /// to the lag without changing its velocity, so the lag never exceeds
    /// `behind`: the spring only moves it toward zero. Returns the lag to
    /// display. Under reduced motion, or with no `behind`, the lag is zero.
    pub(crate) fn step(&mut self, behind: Option<Pixels>, cx: &App) -> Pixels {
        let Some(behind) = behind.filter(|behind| *behind > px(0.)) else {
            self.lag.snap(0.0);
            return px(0.);
        };
        let (policy, now) = (cx.motion_policy(), cx.frame_instant());
        let moved = behind.0 - self.lag.value();
        if moved != 0.0 {
            self.lag.displace(moved, policy, now);
        }
        self.lag.update(policy, now);
        let lag = self.lag.value();
        if lag <= 0.0 {
            self.lag.snap(0.0);
            return px(0.);
        }
        px(lag)
    }

    /// Ends the motion where the list is: the next following layout shows
    /// the end.
    pub(crate) fn cancel(&mut self) {
        self.lag.snap(0.0);
        self.shown = None;
    }

    /// Whether the list is still moving toward its end.
    pub(crate) fn is_moving(&self) -> bool {
        self.lag.is_moving()
    }

    /// While the list moves, declares `bounds`, its viewport, as damage and
    /// requests the next frame scoped to it. Called from prepaint or paint.
    pub(crate) fn request_frame(&self, bounds: Bounds<Pixels>, window: &mut Window) {
        if self.is_moving() {
            window.declare_damage(bounds);
            window.request_animation_frame_at_paint();
        }
    }
}
