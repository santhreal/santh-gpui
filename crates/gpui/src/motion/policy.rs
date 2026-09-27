//! The app-wide motion policy.

use super::MotionPolicy;
use crate::{App, Global};

/// Storage of the app-wide policy. Private so that every change goes through
/// [`App::set_motion_policy`], which redraws the windows.
struct AppMotionPolicy(MotionPolicy);

impl Global for AppMotionPolicy {}

/// The last reduced-motion preference the operating system reported. Private
/// so that every report goes through [`App::report_system_reduce_motion`].
struct SystemReduceMotion(bool);

impl Global for SystemReduceMotion {}

impl App {
    /// The motion policy of the app: reduced motion and the duration scale.
    ///
    /// The reduced flag follows the reduced-motion preference of the
    /// operating system, read with [`Platform::reduce_motion`]: the app starts
    /// with the flag set to the system preference, and each change of the
    /// preference sets the flag to the new value and keeps the duration scale.
    /// On Linux the app starts with the flag unset when the XDG desktop portal
    /// has not answered yet; the answer of the portal applies as a change.
    /// [`App::set_reduce_motion`] and [`App::set_motion_policy`] set the flag
    /// until the next change of the system preference. The duration scale is
    /// 1 until [`App::set_motion_policy`] sets another. An app that combines
    /// its own setting with the system preference reads the preference with
    /// [`App::system_reduce_motion`].
    ///
    /// [`Platform::reduce_motion`]: crate::Platform::reduce_motion
    pub fn motion_policy(&self) -> MotionPolicy {
        self.try_global::<AppMotionPolicy>()
            .map_or(MotionPolicy::DEFAULT, |policy| policy.0)
    }

    /// Sets the motion policy of the app. A changed policy redraws every
    /// window.
    pub fn set_motion_policy(&mut self, policy: MotionPolicy) {
        if self.motion_policy() != policy {
            self.set_global(AppMotionPolicy(policy));
            self.refresh_windows();
        }
    }

    /// Whether reduced motion is on: non-essential animations, such as loading
    /// spinners, render a static state instead of animating. Reads the reduced
    /// flag of [`App::motion_policy`].
    pub fn reduce_motion(&self) -> bool {
        self.motion_policy().reduced()
    }

    /// Turns reduced motion on or off and keeps the duration scale of
    /// [`App::motion_policy`].
    pub fn set_reduce_motion(&mut self, reduce_motion: bool) {
        self.set_motion_policy(self.motion_policy().with_reduced(reduce_motion));
    }

    /// The last reduced-motion preference the operating system reported,
    /// independent of the flag [`App::set_reduce_motion`] set. `false` until
    /// the platform reports a preference.
    pub fn system_reduce_motion(&self) -> bool {
        self.try_global::<SystemReduceMotion>()
            .is_some_and(|preference| preference.0)
    }

    /// Records a report of the system preference. A preference that differs
    /// from the recorded one, or the first report, sets the reduced flag to the
    /// preference; a repeated report keeps the flag the app set.
    pub(crate) fn report_system_reduce_motion(&mut self, reduce_motion: bool) {
        let recorded = self
            .try_global::<SystemReduceMotion>()
            .map(|preference| preference.0);
        if recorded != Some(reduce_motion) {
            self.set_global(SystemReduceMotion(reduce_motion));
            self.set_reduce_motion(reduce_motion);
        }
    }
}
