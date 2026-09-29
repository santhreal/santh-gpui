//! Smooth wheel scrolling: a line wheel tick moves the target offset of a
//! scroll state, and the offset follows the target on a spring.

use std::time::Duration;

use crate::{
    App, Bounds, Pixels, Point, ScrollDelta, Window,
    motion::{ElementSpring, PIXEL_REST_DISTANCE, SpringConfig},
    point, px,
};

#[cfg(test)]
mod tests;

/// The spring the offset of a scroll state with smooth wheel scrolling follows
/// its wheel target on, and a list with smooth tail following follows its end
/// on: critically damped with a 120 ms response, a natural frequency of
/// 2π / 0.12 s (stiffness 2741.6, damping 104.7, unit mass).
pub const WHEEL_SPRING: SpringConfig = SpringConfig::new(2741.557, 104.719_76, 1.0);

/// How far into its motion the first frame of a motion from rest samples
/// [`WHEEL_SPRING`], however soon after the tick the frame draws: one 60 Hz
/// frame. A tick after an idle stretch draws a frame at once; sampled at the
/// tick, that frame repeats the frame before it, and every frame of the
/// motion shows one frame late.
pub(crate) const WHEEL_LEAD: Duration = Duration::from_micros(16_667);

/// Whether a wheel event with `delta` eases: a line delta, as from a mouse
/// wheel, under full motion. A pixel delta, as from a touchpad, applies at
/// once.
pub(crate) fn eases(delta: &ScrollDelta, cx: &App) -> bool {
    matches!(delta, ScrollDelta::Lines(_)) && !cx.motion_policy().reduced()
}

/// The wheel motion of one scroll state.
///
/// `remaining` is the distance, per axis in the offset coordinates of the
/// scroll state, from the offset to the wheel target as of the last step or
/// tick. One [`ElementSpring`] per axis carries that distance to zero. A tick
/// during a motion adds its distance to the spring's value at the instant of
/// the tick and keeps the spring's velocity, so consecutive ticks accumulate
/// without a stall. A tick from rest leaves its distance in `start`, and the
/// next step starts the springs with it [`WHEEL_LEAD`] before that step.
/// `written` is the offset the scroll state held after the last tick or step:
/// an offset that differs was written by something else, which ends the
/// motion where it is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WheelEase<P> {
    x: ElementSpring,
    y: ElementSpring,
    remaining: Point<Pixels>,
    start: Option<Point<Pixels>>,
    written: Option<P>,
}

impl<P: Copy + PartialEq> WheelEase<P> {
    /// A motion at rest.
    pub(crate) fn new() -> Self {
        let rest = ElementSpring::at_rest(0.0, WHEEL_SPRING, PIXEL_REST_DISTANCE);
        Self {
            x: rest,
            y: rest,
            remaining: Point::default(),
            start: None,
            written: None,
        }
    }

    /// The wheel target of a scroll state at `offset` that holds `at`:
    /// `offset` plus the remaining distance, or `offset` itself when something
    /// else wrote the offset since the last tick or step.
    pub(crate) fn target(&self, offset: Point<Pixels>, at: P) -> Point<Pixels> {
        if self.written == Some(at) {
            offset + self.remaining
        } else {
            offset
        }
    }

    /// Moves the wheel target of a scroll state that holds `at` by `delta`,
    /// already clamped to the content: at the current frame instant during a
    /// motion, and at the next step from rest.
    pub(crate) fn retarget(&mut self, delta: Point<Pixels>, at: P, cx: &App) {
        if self.written != Some(at) {
            self.cancel();
        }
        if self.x.is_moving() || self.y.is_moving() {
            let (policy, now) = (cx.motion_policy(), cx.frame_instant());
            self.x.displace(delta.x.0, policy, now);
            self.y.displace(delta.y.0, policy, now);
        } else {
            self.start = Some(self.start.unwrap_or_default() + delta);
        }
        self.remaining = self.remaining + delta;
        self.written = Some(at);
    }

    /// Samples the motion at the current frame instant and returns the
    /// distance the offset of a scroll state that holds `at` moves this
    /// frame. Returns `None` at rest, and ends the motion and returns `None`
    /// when something else wrote the offset.
    pub(crate) fn step(&mut self, at: P, cx: &App) -> Option<Point<Pixels>> {
        if !self.is_moving() {
            return None;
        }
        if self.written != Some(at) {
            self.cancel();
            return None;
        }
        let (policy, now) = (cx.motion_policy(), cx.frame_instant());
        if let Some(start) = self.start.take() {
            let then = now.before(WHEEL_LEAD);
            self.x.displace(start.x.0, policy, then);
            self.y.displace(start.y.0, policy, then);
        }
        self.x.update(policy, now);
        self.y.update(policy, now);
        let remaining = point(px(self.x.value()), px(self.y.value()));
        let step = self.remaining - remaining;
        self.remaining = remaining;
        Some(step)
    }

    /// Records `at` as the offset the scroll state holds after a step.
    pub(crate) fn record(&mut self, at: P) {
        self.written = Some(at);
    }

    /// Whether the offset is still moving toward the target.
    pub(crate) fn is_moving(&self) -> bool {
        self.start.is_some() || self.x.is_moving() || self.y.is_moving()
    }

    /// Ends the motion where the offset is.
    pub(crate) fn cancel(&mut self) {
        self.x.snap(0.0);
        self.y.snap(0.0);
        self.remaining = Point::default();
        self.start = None;
    }

    /// While the offset moves, declares `bounds`, the viewport of the scroll
    /// state, as damage and requests the next frame scoped to it. Called from
    /// prepaint or paint.
    pub(crate) fn request_frame(&self, bounds: Bounds<Pixels>, window: &mut Window) {
        if self.is_moving() {
            window.declare_damage(bounds);
            window.request_animation_frame_at_paint();
        }
    }
}
