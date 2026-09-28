//! The first surface configure runs on a worker so the drawing thread never
//! waits on `vkCreateSwapchainKHR`. These tests pin the contract the
//! platforms build on:
//!
//! - `is_done` reports the worker without blocking, and reads done by the
//!   time the worker's notify arrives, so a platform that presents on the
//!   notify never finds the configure still running and stalls.
//! - The notify runs once per configure, after it returns, panics included,
//!   so a window waiting for it always gets its frame.
//! - `finish` and drop wait for the worker, so no later configure,
//!   `destroy`, or window teardown overlaps the first one.
//!
//! They do not create a GPU surface: how the renderer and the X11 window use
//! these is not covered here.

use super::{Configuring, Stage};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const BOUND: Duration = Duration::from_secs(5);
/// How long a notify that should not arrive is waited for.
const QUIET: Duration = Duration::from_millis(50);

/// A configure that runs until `release` is sent, or for `BOUND`: a defect
/// that joins the worker before the release fails the test instead of
/// hanging it.
fn gated() -> (Configuring, mpsc::Sender<()>) {
    let (release, wait) = mpsc::channel::<()>();
    let configuring = Configuring::spawn(move || {
        wait.recv_timeout(BOUND).ok();
    })
    .expect("spawn the configure worker");
    (configuring, release)
}

/// A configure that sets `ran` after `delay`.
fn slow(delay: Duration) -> (Configuring, Arc<AtomicBool>) {
    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let configuring = Configuring::spawn(move || {
        std::thread::sleep(delay);
        flag.store(true, Ordering::SeqCst);
    })
    .expect("spawn the configure worker");
    (configuring, ran)
}

/// A notify that sends on the returned receiver.
fn counting() -> (super::Notify, mpsc::Receiver<()>) {
    let (sender, receiver) = mpsc::channel();
    (
        Arc::new(move || {
            sender.send(()).ok();
        }),
        receiver,
    )
}

#[test]
fn a_surface_without_a_configure_is_done() {
    assert!(Configuring::done().is_done());
}

#[test]
fn is_done_stays_false_until_the_configure_returns() {
    let (mut configuring, release) = gated();

    std::thread::sleep(Duration::from_millis(20));
    assert!(
        !configuring.is_done(),
        "done while the configure still runs"
    );
    release.send(()).expect("release the configure");

    let start = Instant::now();
    while !configuring.is_done() {
        assert!(
            start.elapsed() < BOUND,
            "a returned configure never reads as done"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(configuring.is_done(), "done does not stay done");
}

#[test]
fn the_notify_runs_once_after_the_configure_returns() {
    let (configuring, release) = gated();
    let (notify, notified) = counting();
    configuring.when_done(notify);

    assert!(
        notified.recv_timeout(QUIET).is_err(),
        "notified while the configure still runs"
    );
    release.send(()).expect("release the configure");
    notified
        .recv_timeout(BOUND)
        .expect("the configure returned without a notify");
    assert!(notified.recv_timeout(QUIET).is_err(), "notified twice");
}

#[test]
fn the_configure_is_done_when_its_notify_arrives() {
    let (mut configuring, release) = gated();
    let (sender, notified) = mpsc::channel();
    // The worker stays alive after it notifies: a check on the thread's exit
    // instead of the configure's return reads not done here, and a platform
    // presenting on the notify would skip the frame and never be woken.
    configuring.when_done(Arc::new(move || {
        sender.send(()).ok();
        std::thread::sleep(Duration::from_millis(200));
    }));
    release.send(()).expect("release the configure");
    notified.recv_timeout(BOUND).expect("never notified");
    assert!(
        configuring.is_done(),
        "the notify arrived before the configure read as done"
    );
}

#[test]
fn a_notify_after_the_configure_returned_is_not_called() {
    let (configuring, release) = gated();
    release.send(()).expect("release the configure");
    let start = Instant::now();
    while matches!(
        *configuring.worker.as_ref().expect("worker").stage.lock(),
        Stage::Running(_)
    ) {
        assert!(start.elapsed() < BOUND, "the configure never returned");
        std::thread::sleep(Duration::from_millis(1));
    }

    let (notify, notified) = counting();
    configuring.when_done(notify);
    assert!(notified.recv_timeout(QUIET).is_err());
}

#[test]
fn finish_waits_for_the_configure() {
    let (mut configuring, ran) = slow(Duration::from_millis(50));
    configuring.finish();
    assert!(
        ran.load(Ordering::SeqCst),
        "finish returned before the configure"
    );
    assert!(configuring.is_done());
}

#[test]
fn drop_waits_for_the_configure() {
    let (configuring, ran) = slow(Duration::from_millis(50));
    drop(configuring);
    assert!(
        ran.load(Ordering::SeqCst),
        "drop returned before the configure"
    );
}

#[test]
fn a_panicking_configure_notifies_and_resumes_in_finish() {
    let (release, wait) = mpsc::channel::<()>();
    let mut configuring = Configuring::spawn(move || {
        wait.recv_timeout(BOUND).ok();
        panic!("swapchain lost");
    })
    .expect("spawn the configure worker");
    let (notify, notified) = counting();
    configuring.when_done(notify);
    release.send(()).expect("release the configure");
    notified
        .recv_timeout(BOUND)
        .expect("a panicked configure left its window waiting");

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| configuring.finish()))
        .expect_err("finish swallowed the configure's panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"swapchain lost"));
    assert!(configuring.is_done(), "a joined worker is not done");
}

#[test]
fn dropping_a_panicked_configure_does_not_panic() {
    let configuring =
        Configuring::spawn(|| panic!("swapchain lost")).expect("spawn the configure worker");
    let start = Instant::now();
    drop(configuring);
    assert!(start.elapsed() < BOUND);
}
