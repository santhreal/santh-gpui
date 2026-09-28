//! Motion math from the `motion` crate, bound to GPUI.
//!
//! Every public item of the `motion` crate is re-exported here: easing curves,
//! springs, tweens, motion roles, presets, loader waveforms, and
//! [`MotionPolicy`]. GPUI adds three bindings:
//!
//! - [`FrameInstant`]: the [`Timestamp`] of the clock GPUI animations sample,
//!   read with [`App::frame_instant`](crate::App::frame_instant).
//! - The app-wide [`MotionPolicy`], read with
//!   [`App::motion_policy`](crate::App::motion_policy) and set with
//!   [`App::set_motion_policy`](crate::App::set_motion_policy).
//!   [`App::reduce_motion`](crate::App::reduce_motion) reads its reduced flag,
//!   which follows the reduced-motion preference of the operating system.
//! - [`MotionDriver`], [`MotionFrame`], and [`Advance`]: frame driving for a
//!   view. The view requests the next animation frame while one of its motion
//!   values moves and requests none once every value is at rest. A value
//!   tracked with [`MotionFrame::track_within`] scopes that frame to the
//!   bounds it repaints.
//!
//! # Example
//!
//! ```ignore
//! struct Panel {
//!     driver: MotionDriver,
//!     width: Animator<FrameInstant>,
//! }
//!
//! impl Render for Panel {
//!     fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
//!         let mut frame = self.driver.begin(cx);
//!         frame.track(&mut self.width);
//!         self.driver.end(frame, window);
//!         div().w(px(self.width.value()))
//!     }
//! }
//! ```

mod clock;
mod drive;
mod element;
mod policy;
#[cfg(test)]
mod system_preference_tests;
#[cfg(test)]
mod tests;

pub use ::motion::*;
pub use clock::FrameInstant;
pub use drive::{Advance, MotionDriver, MotionFrame};
pub(crate) use element::{ElementSpring, PIXEL_REST_DISTANCE, UNIT_REST_DISTANCE};
