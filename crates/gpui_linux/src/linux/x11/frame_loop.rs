//! Frame pacing for X11 windows. A window runs frames while GPUI has frame
//! demand for it (a dirty view, a next-frame callback, a renderer that asks
//! to redraw), at no more than the refresh rate of the CRTC that shows the
//! largest part of the window. A window with no demand, or one the X server
//! does not show, registers no timer and costs the event loop no wakeups.
//!
//! Each CRTC runs its own mode. A window paced at another CRTC's faster
//! rate draws frames its display never shows and, under Fifo presentation,
//! blocks the thread that draws until the display releases a swapchain
//! image.

use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use calloop::{
    LoopHandle, RegistrationToken,
    timer::{TimeoutAction, Timer},
};
use gpui::Bounds;
use x11rb::protocol::xproto;

use super::X11Client;

/// The frame interval of a display whose refresh rate RandR does not report.
pub(crate) const DEFAULT_FRAME_INTERVAL: Duration = Duration::from_micros(1_000_000 / 60);

/// One CRTC that shows a mode: the rectangle of the root window it scans
/// out, in device pixels, and the frame interval of its mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CrtcRate {
    pub(crate) bounds: Bounds<i32>,
    pub(crate) interval: Duration,
}

/// The frame interval of the CRTC that shows the largest part of `window`,
/// a rectangle of the root window in device pixels. Of CRTCs that show
/// equal parts, and for a window no CRTC shows, the first CRTC's. `None`
/// without CRTCs.
pub(crate) fn interval_at(crtcs: &[CrtcRate], window: Bounds<i32>) -> Option<Duration> {
    crtcs
        .iter()
        .enumerate()
        .max_by_key(|(index, crtc)| (overlap(crtc.bounds, window), std::cmp::Reverse(*index)))
        .map(|(_, crtc)| crtc.interval)
}

/// The area two rectangles share, in square device pixels.
fn overlap(a: Bounds<i32>, b: Bounds<i32>) -> i64 {
    let shared = |a_start: i32, a_len: i32, b_start: i32, b_len: i32| {
        let start = i64::from(a_start.max(b_start));
        let end =
            (i64::from(a_start) + i64::from(a_len)).min(i64::from(b_start) + i64::from(b_len));
        (end - start).max(0)
    };
    shared(a.origin.x, a.size.width, b.origin.x, b.size.width)
        * shared(a.origin.y, a.size.height, b.origin.y, b.size.height)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No frame timer is registered.
    Idle,
    /// A frame timer is registered and has not fired.
    Armed,
    /// The frame timer is running a frame.
    Ticking,
    /// Demand arrived while a frame ran: the timer fires again.
    Rearm,
}

/// The pacing state of one window, apart from the event loop. Each
/// transition returns the timer change the caller applies, so every
/// sequence of transitions is testable without an X server.
#[derive(Clone, Copy, Debug)]
struct Pacer {
    visible: bool,
    phase: Phase,
    /// The frame interval of the CRTC that shows the window.
    interval: Duration,
    last_frame: Option<Instant>,
}

impl Pacer {
    const fn new() -> Self {
        Self {
            visible: false,
            phase: Phase::Idle,
            interval: DEFAULT_FRAME_INTERVAL,
            last_frame: None,
        }
    }

    /// Records frame demand. Returns the deadline of a timer to register.
    fn wake(&mut self, now: Instant) -> Option<Instant> {
        if !self.visible {
            return None;
        }
        match self.phase {
            Phase::Idle => {
                self.phase = Phase::Armed;
                // Demand after an idle stretch draws at once. Demand within
                // an interval of the last frame waits the interval out, so
                // frames never outpace the display however fast demand
                // arrives.
                Some(
                    self.last_frame
                        .map_or(now, |last| (last + self.interval).max(now)),
                )
            }
            Phase::Ticking => {
                self.phase = Phase::Rearm;
                None
            }
            Phase::Armed | Phase::Rearm => None,
        }
    }

    /// The X server shows the window. Returns the deadline of a timer to
    /// register: a window that becomes visible draws.
    fn show(&mut self, now: Instant) -> Option<Instant> {
        if self.visible {
            return None;
        }
        self.visible = true;
        self.wake(now)
    }

    /// The X server stops showing the window. Returns true when the
    /// registered timer is to be removed; a running frame drops its own
    /// timer when it ends.
    fn hide(&mut self) -> bool {
        self.visible = false;
        if self.phase == Phase::Armed {
            self.phase = Phase::Idle;
            true
        } else {
            false
        }
    }

    /// The timer `wake` or `show` returned could not be registered.
    fn disarm(&mut self) {
        if self.phase == Phase::Armed {
            self.phase = Phase::Idle;
        }
    }

    fn begin_frame(&mut self, now: Instant) {
        self.phase = Phase::Ticking;
        self.last_frame = Some(now);
    }

    /// Returns the next deadline of the timer that ran the frame for
    /// `deadline`, or `None` to drop that timer.
    fn end_frame(&mut self, deadline: Instant, now: Instant) -> Option<Instant> {
        if self.visible && self.phase == Phase::Rearm {
            self.phase = Phase::Armed;
            Some(next_deadline(deadline, now, self.interval))
        } else {
            self.phase = Phase::Idle;
            None
        }
    }
}

/// The first point of the grid `deadline + k * interval` after `now`. A
/// frame that overran its interval skips the grid points it missed.
fn next_deadline(deadline: Instant, now: Instant, interval: Duration) -> Instant {
    let interval_ns = interval.as_nanos();
    if interval_ns == 0 {
        return now;
    }
    let elapsed_ns = now.saturating_duration_since(deadline).as_nanos();
    // In 1..=interval_ns; RandR intervals are far below u64::MAX nanoseconds.
    let until_next = interval_ns - elapsed_ns % interval_ns;
    now + Duration::from_nanos(u64::try_from(until_next).unwrap_or(u64::MAX))
}

/// A window's frame source: a one-shot calloop timer registered while the
/// window has frame demand, kept at the refresh interval while each frame
/// leaves demand behind, and dropped by the first frame that leaves none.
pub(crate) struct FrameLoop {
    x_window: xproto::Window,
    loop_handle: LoopHandle<'static, X11Client>,
    pacer: Cell<Pacer>,
    timer: Cell<Option<RegistrationToken>>,
}

impl FrameLoop {
    pub(crate) fn new(
        x_window: xproto::Window,
        loop_handle: LoopHandle<'static, X11Client>,
    ) -> Self {
        Self {
            x_window,
            loop_handle,
            pacer: Cell::new(Pacer::new()),
            timer: Cell::new(None),
        }
    }

    fn update<R>(&self, transition: impl FnOnce(&mut Pacer) -> R) -> R {
        let mut pacer = self.pacer.get();
        let result = transition(&mut pacer);
        self.pacer.set(pacer);
        result
    }

    /// Requests a frame. A window the X server does not show draws when it
    /// is next shown instead.
    pub(crate) fn wake(&self) {
        if let Some(deadline) = self.update(|pacer| pacer.wake(Instant::now())) {
            self.arm(deadline);
        }
    }

    /// Starts frames, at `interval`, for a window the X server shows.
    pub(crate) fn show(&self, interval: Duration) {
        self.set_interval(interval);
        if let Some(deadline) = self.update(|pacer| pacer.show(Instant::now())) {
            self.arm(deadline);
        }
    }

    /// Paces the window's later frames at `interval`, the frame interval of
    /// the CRTC that now shows it.
    pub(crate) fn set_interval(&self, interval: Duration) {
        if self.pacer.get().interval != interval {
            log::info!(
                "x11: window {:#x} runs a frame every {interval:?}",
                self.x_window
            );
            self.update(|pacer| pacer.interval = interval);
        }
    }

    /// Stops frames for a window the X server does not show, or one that is
    /// being destroyed.
    pub(crate) fn hide(&self) {
        if self.update(Pacer::hide)
            && let Some(token) = self.timer.take()
        {
            self.loop_handle.remove(token);
        }
    }

    pub(crate) fn begin_frame(&self) {
        self.update(|pacer| pacer.begin_frame(Instant::now()));
    }

    /// Ends the frame the timer ran for `deadline` and returns what the
    /// timer does next.
    pub(crate) fn end_frame(&self, deadline: Instant) -> TimeoutAction {
        match self.update(|pacer| pacer.end_frame(deadline, Instant::now())) {
            Some(next) => TimeoutAction::ToInstant(next),
            None => {
                self.timer.set(None);
                TimeoutAction::Drop
            }
        }
    }

    fn arm(&self, deadline: Instant) {
        let x_window = self.x_window;
        match self.loop_handle.insert_source(
            Timer::from_deadline(deadline),
            move |deadline, _, client| client.run_frame(x_window, deadline),
        ) {
            Ok(token) => self.timer.set(Some(token)),
            Err(err) => {
                log::error!("X11: failed to schedule a frame: {}", err.error);
                self.update(Pacer::disarm);
            }
        }
    }
}

#[cfg(test)]
mod crtc_tests;
#[cfg(test)]
mod tests;
