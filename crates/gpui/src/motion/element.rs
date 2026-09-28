//! The spring an element moves one of its values with, sampled at the frame
//! instant.

use super::{Animator, FrameInstant, MotionModel, MotionPolicy, SpringConfig};

/// Distance from the target, in pixels, at which a pixel spring lands.
pub(crate) const PIXEL_REST_DISTANCE: f32 = 0.02;
/// Speed, in rest distances per second, under which a spring within its rest
/// distance of the target lands.
const REST_SPEED_PER_DISTANCE: f32 = 20.0;

/// One value of an element moved toward a target by a spring.
///
/// The spring is evaluated in closed form from the state where its current
/// motion started, so a sample depends on the frame instant and not on how
/// frames divide the time. A new target starts a motion from the value and
/// velocity sampled at the instant of the change.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ElementSpring {
    animator: Animator<FrameInstant>,
    config: SpringConfig,
    rest_distance: f32,
}

impl ElementSpring {
    /// A spring at rest on `value` that lands within `rest_distance` of its
    /// target.
    pub(crate) fn at_rest(value: f32, config: SpringConfig, rest_distance: f32) -> Self {
        Self {
            animator: Animator::at_rest(value),
            config,
            rest_distance,
        }
    }

    /// The value at the last [`Self::update`].
    pub(crate) fn value(&self) -> f32 {
        self.animator.value()
    }

    /// Whether the spring was moving at the last [`Self::update`] or started
    /// since.
    pub(crate) fn is_moving(&self) -> bool {
        !self.animator.is_at_rest()
    }

    /// Sets the spring parameters of later motions.
    pub(crate) fn set_config(&mut self, config: SpringConfig) {
        self.config = config;
    }

    /// Moves the value by `delta` at `now` without changing its velocity or
    /// target, as when the frame the value is measured in moves by `-delta`.
    /// Under reduced motion, lands on the target.
    pub(crate) fn displace(&mut self, delta: f32, policy: MotionPolicy, now: FrameInstant) {
        let target = self.animator.target();
        if policy.reduced() {
            self.animator.snap(target);
            return;
        }
        let sample = self.animator.sample(now);
        self.animator.start(
            sample.value + delta,
            sample.velocity,
            target,
            MotionModel::Spring(self.config),
            policy,
            now,
        );
    }

    /// Places the spring at rest on `value`.
    pub(crate) fn snap(&mut self, value: f32) {
        self.animator.snap(value);
    }

    /// Samples the spring at `now` and returns whether it is still moving. A
    /// sample within the rest distance of the target at a speed under
    /// [`REST_SPEED_PER_DISTANCE`] rest distances per second, or any sample
    /// under reduced motion, lands on the target.
    pub(crate) fn update(&mut self, policy: MotionPolicy, now: FrameInstant) -> bool {
        let target = self.animator.target();
        if policy.reduced() {
            self.animator.snap(target);
            return false;
        }
        let sample = self.animator.update(now);
        if sample.at_rest {
            return false;
        }
        if (sample.value - target).abs() <= self.rest_distance
            && sample.velocity.abs() <= self.rest_distance * REST_SPEED_PER_DISTANCE
        {
            self.animator.snap(target);
            return false;
        }
        true
    }
}
