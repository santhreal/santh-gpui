//! WHY: an X11 window runs frames only while GPUI has frame demand for it.
//! The class closed here is a frame source that outlives its demand or
//! misses it: a timer left registered after a frame that leaves no demand
//! (the window wakes the process every refresh interval while nothing on
//! screen changes), two timers for one window (frames at twice the rate),
//! a timer for a window the X server does not show, demand dropped while
//! a frame runs or while the window is hidden (the window stops drawing
//! until the next input), and frames that outpace the display. The model
//! check runs every sequence of transitions up to a fixed length against
//! the timer the event loop would hold. Not covered: the X server's map
//! and visibility events that drive `show` and `hide`, and the calloop
//! registration itself.

use std::time::{Duration, Instant};

use super::{Pacer, Phase, next_deadline};

const INTERVAL: Duration = Duration::from_millis(16);

/// A pacer of a window the X server does not show yet.
fn hidden_pacer() -> Pacer {
    let mut pacer = Pacer::new();
    pacer.interval = INTERVAL;
    pacer
}

#[test]
fn a_window_draws_when_shown_and_parks_after_a_frame_without_demand() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    assert_eq!(
        pacer.show(t0),
        Some(t0),
        "a window that becomes visible draws at once"
    );
    pacer.begin_frame(t0);
    assert_eq!(
        pacer.end_frame(t0, t0 + Duration::from_millis(2)),
        None,
        "a frame that leaves no demand drops its timer"
    );
    assert_eq!(pacer.phase, Phase::Idle);
}

#[test]
fn demand_while_a_frame_runs_keeps_the_timer_on_the_frame_grid() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    pacer.show(t0);
    pacer.begin_frame(t0);
    assert_eq!(
        pacer.wake(t0),
        None,
        "a running frame registers no second timer"
    );
    assert_eq!(
        pacer.end_frame(t0, t0 + Duration::from_millis(3)),
        Some(t0 + INTERVAL),
        "demand left by a frame runs the next frame one interval after the last"
    );
    assert_eq!(pacer.phase, Phase::Armed);
}

#[test]
fn demand_after_an_idle_stretch_draws_at_once_and_demand_within_an_interval_waits() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    pacer.show(t0);
    pacer.begin_frame(t0);
    pacer.end_frame(t0, t0);

    let soon = t0 + Duration::from_millis(5);
    assert_eq!(
        pacer.wake(soon),
        Some(t0 + INTERVAL),
        "frames never outpace the display"
    );
    pacer.begin_frame(t0 + INTERVAL);
    pacer.end_frame(t0 + INTERVAL, t0 + INTERVAL);

    let later = t0 + Duration::from_secs(2);
    assert_eq!(
        pacer.wake(later),
        Some(later),
        "demand after an idle stretch draws at once"
    );
}

#[test]
fn repeated_demand_registers_one_timer() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    pacer.show(t0);
    for _ in 0..3 {
        assert_eq!(pacer.wake(t0), None);
    }
    assert_eq!(pacer.phase, Phase::Armed);
}

#[test]
fn demand_while_hidden_draws_when_the_window_is_shown() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    assert_eq!(pacer.wake(t0), None, "a hidden window registers no timer");
    assert_eq!(pacer.phase, Phase::Idle);
    let shown = t0 + Duration::from_millis(40);
    assert_eq!(pacer.show(shown), Some(shown));
    assert_eq!(
        pacer.show(shown),
        None,
        "a second show registers no second timer"
    );
}

#[test]
fn hiding_removes_an_armed_timer_and_a_running_frame_drops_its_own() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    pacer.show(t0);
    assert!(pacer.hide(), "an armed timer of a hidden window is removed");
    assert_eq!(pacer.phase, Phase::Idle);

    pacer.show(t0);
    pacer.begin_frame(t0);
    pacer.wake(t0);
    assert!(
        !pacer.hide(),
        "the running frame's timer is not removed under it"
    );
    assert_eq!(
        pacer.end_frame(t0, t0),
        None,
        "a frame of a window hidden while it ran drops its timer despite demand"
    );
}

#[test]
fn a_timer_that_failed_to_register_leaves_the_window_able_to_arm_again() {
    let t0 = Instant::now();
    let mut pacer = hidden_pacer();
    assert!(pacer.show(t0).is_some());
    pacer.disarm();
    assert_eq!(pacer.wake(t0), Some(t0));
}

#[test]
fn a_frame_that_overran_skips_the_grid_points_it_missed() {
    let t0 = Instant::now();
    let cases = [
        (Duration::ZERO, INTERVAL),
        (Duration::from_millis(1), INTERVAL),
        (INTERVAL, INTERVAL * 2),
        (INTERVAL * 2 + Duration::from_millis(8), INTERVAL * 3),
    ];
    for (elapsed, expected) in cases {
        assert_eq!(
            next_deadline(t0, t0 + elapsed, INTERVAL),
            t0 + expected,
            "{elapsed:?} after the deadline"
        );
    }
    let now = t0 + Duration::from_millis(7);
    assert_eq!(next_deadline(t0, now, Duration::ZERO), now);
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Wake,
    Show,
    Hide,
    /// The registered timer fires; the frame leaves demand or none.
    Frame {
        demand: bool,
    },
    /// Registering the timer of the previous `wake` or `show` failed.
    RegisterFails,
}

const STEPS: [Step; 6] = [
    Step::Wake,
    Step::Show,
    Step::Hide,
    Step::Frame { demand: false },
    Step::Frame { demand: true },
    Step::RegisterFails,
];

/// The event loop as the pacer drives it: the timer it holds, and whether
/// the window has demand no frame has served yet.
struct Model {
    pacer: Pacer,
    timer: Option<Instant>,
    unserved: bool,
    /// The previous step registered a new timer.
    registered: bool,
    now: Instant,
}

impl Model {
    fn register(&mut self, deadline: Option<Instant>, trace: &[Step]) {
        if let Some(deadline) = deadline {
            assert!(self.timer.is_none(), "a second timer after {trace:?}");
            self.timer = Some(deadline);
            self.registered = true;
        }
    }

    fn apply(&mut self, step: Step, trace: &[Step]) {
        self.now += Duration::from_millis(5);
        let registered = std::mem::take(&mut self.registered);
        match step {
            Step::Wake => {
                self.unserved = true;
                let deadline = self.pacer.wake(self.now);
                self.register(deadline, trace);
            }
            Step::Show => {
                // A window that becomes visible draws; showing a shown
                // window is no demand.
                self.unserved |= !self.pacer.visible;
                let deadline = self.pacer.show(self.now);
                self.register(deadline, trace);
            }
            Step::Hide => {
                if self.pacer.hide() {
                    assert!(
                        self.timer.take().is_some(),
                        "removed no timer after {trace:?}"
                    );
                }
            }
            Step::Frame { demand } => {
                let Some(deadline) = self.timer else {
                    return;
                };
                self.now = self.now.max(deadline);
                self.pacer.begin_frame(self.now);
                self.unserved = false;
                if demand {
                    self.unserved = true;
                    assert_eq!(
                        self.pacer.wake(self.now),
                        None,
                        "a timer inside a frame after {trace:?}"
                    );
                }
                self.timer = self.pacer.end_frame(deadline, self.now);
                if let Some(next) = self.timer {
                    assert!(
                        next > self.now,
                        "a timer at or before the frame that set it after {trace:?}"
                    );
                    assert!(
                        next - self.now <= INTERVAL,
                        "a timer more than an interval out after {trace:?}"
                    );
                }
            }
            Step::RegisterFails => {
                if !registered {
                    return;
                }
                // The failure is logged and the demand waits for the next
                // wake.
                self.timer = None;
                self.unserved = false;
                self.pacer.disarm();
            }
        }
        self.check(trace);
    }

    fn check(&self, trace: &[Step]) {
        let armed = matches!(self.pacer.phase, Phase::Armed);
        assert_eq!(
            armed,
            self.timer.is_some(),
            "the pacer and the event loop disagree on the timer after {trace:?}"
        );
        if !self.pacer.visible {
            assert!(
                self.timer.is_none(),
                "a hidden window holds a timer after {trace:?}"
            );
        }
        if self.pacer.visible && self.unserved {
            assert!(
                self.timer.is_some(),
                "a visible window with unserved demand holds no timer after {trace:?}"
            );
        }
        if !self.unserved {
            assert!(
                self.timer.is_none(),
                "a window with no demand holds a timer after {trace:?}"
            );
        }
    }
}

/// Every sequence of `DEPTH` steps, from a window not yet shown.
#[test]
fn every_transition_sequence_holds_one_timer_exactly_while_a_shown_window_has_demand() {
    const DEPTH: u32 = 7;
    let t0 = Instant::now();
    let total = STEPS.len().pow(DEPTH);
    for index in 0..total {
        let mut rest = index;
        let mut trace = Vec::with_capacity(DEPTH as usize);
        let mut model = Model {
            pacer: hidden_pacer(),
            timer: None,
            unserved: false,
            registered: false,
            now: t0,
        };
        for _ in 0..DEPTH {
            let step = STEPS[rest % STEPS.len()];
            rest /= STEPS.len();
            trace.push(step);
            model.apply(step, &trace);
        }
    }
}
