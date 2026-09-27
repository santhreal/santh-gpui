//! The reduced-motion answers of the XDG desktop portal on the main thread.
//!
//! WHY: closes the classes "the first read of the reduced-motion preference
//! waits for the portal", which held the main thread before the first window
//! for up to 100 ms on a cold start, "an answer that arrived before the first
//! read is not applied", and "an answer after the first read is lost, applies
//! without the change callback, or ends the following of later answers".
//! Every case runs [`SystemReduceMotion::follow`], which the first read and
//! the change-callback registration of the platform call, on a real
//! [`LinuxDispatcher`] and calloop main loop. Not caught: the portal reads,
//! tested in `xdg_desktop_portal`, and a caller that blocks before the first
//! read.

use std::{
    cell::Cell,
    rc::Rc,
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

use gpui::{ForegroundExecutor, RunnableVariant};

use super::SystemReduceMotion;
use crate::linux::{LinuxDispatcher, PriorityQueueCalloopReceiver};

/// Longest wait for an answer to reach the main thread.
const DELIVERY_DEADLINE: Duration = Duration::from_secs(5);

/// A calloop main loop that runs the tasks of its foreground executor.
struct MainLoop {
    event_loop: calloop::EventLoop<'static, ()>,
    executor: ForegroundExecutor,
}

impl MainLoop {
    fn new() -> Self {
        let (sender, receiver) = PriorityQueueCalloopReceiver::<RunnableVariant>::new();
        let event_loop = calloop::EventLoop::try_new().expect("create the calloop main loop");
        event_loop
            .handle()
            .insert_source(receiver, |event, _, _| {
                if let calloop::channel::Event::Msg(runnable) = event {
                    runnable.run();
                }
            })
            .expect("insert the main-thread task source");
        let executor = ForegroundExecutor::new(Arc::new(LinuxDispatcher::new(sender)));
        Self {
            event_loop,
            executor,
        }
    }

    /// Runs main-thread tasks until `done` holds. Fails after
    /// [`DELIVERY_DEADLINE`].
    fn run_until(&mut self, what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + DELIVERY_DEADLINE;
        while !done() {
            let now = Instant::now();
            assert!(now < deadline, "{what} within {DELIVERY_DEADLINE:?}");
            self.event_loop
                .dispatch(Some(deadline - now), &mut ())
                .expect("dispatch the calloop main loop");
        }
    }
}

/// A preference state with a change callback that counts its calls.
fn counted_state() -> (Rc<SystemReduceMotion>, Rc<Cell<usize>>) {
    let state = Rc::new(SystemReduceMotion::default());
    let calls = Rc::new(Cell::new(0));
    *state.on_change.borrow_mut() = Some(Box::new({
        let calls = calls.clone();
        move || calls.set(calls.get() + 1)
    }));
    (state, calls)
}

#[test]
fn an_answer_that_arrived_before_the_first_read_applies_at_the_read() {
    let main_loop = MainLoop::new();
    for reduce_motion in [false, true] {
        let (state, calls) = counted_state();
        let (sender, answers) = smol::channel::unbounded();
        sender.try_send(reduce_motion).unwrap();

        state.follow(answers, &main_loop.executor);

        assert_eq!(
            state.reduce_motion.get(),
            reduce_motion,
            "the first read after the portal answered {reduce_motion}"
        );
        assert_eq!(calls.get(), usize::from(reduce_motion));
    }
}

#[test]
fn the_first_read_returns_before_a_late_answer_which_then_applies_as_a_change() {
    /// Delay of the portal answer after the first read starts, unless the
    /// read returns first. A read that waits for the answer receives it.
    const PORTAL_DELAY: Duration = Duration::from_millis(50);

    let mut main_loop = MainLoop::new();
    let (state, calls) = counted_state();
    let (sender, answers) = smol::channel::unbounded();
    let (read_started, read_started_rx) = mpsc::channel::<()>();
    let (read_returned, read_returned_rx) = mpsc::channel::<()>();
    let portal = thread::spawn({
        let sender = sender.clone();
        move || {
            read_started_rx.recv().unwrap();
            let returned_first = read_returned_rx.recv_timeout(PORTAL_DELAY).is_ok();
            sender.try_send(true).unwrap();
            returned_first
        }
    });

    read_started.send(()).unwrap();
    state.follow(answers, &main_loop.executor);
    let first_read = state.reduce_motion.get();
    read_returned.send(()).ok();

    assert!(
        portal.join().unwrap(),
        "the first read returned before the portal answered {PORTAL_DELAY:?} later"
    );
    assert!(!first_read, "the first read before the portal answered");
    assert_eq!(calls.get(), 0);

    main_loop.run_until("the late answer applies", || state.reduce_motion.get());
    assert_eq!(calls.get(), 1, "calls after the late answer");

    sender.try_send(false).unwrap();
    main_loop.run_until("a change after the late answer applies", || {
        !state.reduce_motion.get()
    });
    assert_eq!(calls.get(), 2, "calls after a change after the late answer");
}
