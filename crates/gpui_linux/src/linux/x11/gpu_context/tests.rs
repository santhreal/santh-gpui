//! WHY: an X11 client creates its GPU context on a thread that reads the
//! client's X connection, and its first window adopts that context, whose
//! adapter was selected without a surface. The classes closed here are a
//! thread that outlives the connection it reads (the client's setup fails,
//! or the client ends before its first window, and the connection closes
//! while the thread creates the instance on it), and a first window that
//! keeps the unchecked context although creating its renderer reported an
//! error. Not covered: the order of the X connection's setup and the
//! driver's load, which iris's tests/startup.rs checks against a fresh
//! Xvfb; the adapter selection; the error scope around the renderer's
//! creation, which needs a GPU device; and the surface's configure, which
//! the window does not wait for and whose failure the renderer's trial
//! (gpui_wgpu `wgpu_renderer/trial`) turns into a new context.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, mpsc},
    time::Duration,
};

use anyhow::anyhow;
use gpui_wgpu::wgpu;
use parking_lot::Mutex;

use super::{Pending, adopted};

/// Records its release in a log it shares with the test.
struct Released(Arc<Mutex<Vec<&'static str>>>);

impl Drop for Released {
    fn drop(&mut self) {
        self.0.lock().push("released");
    }
}

#[test]
fn a_dropped_pending_result_ends_its_thread_before_it_releases_what_the_thread_reads() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (started, running) = mpsc::channel();
    let thread_log = Arc::clone(&log);
    let pending = Pending::spawn("test", Released(Arc::clone(&log)), move || {
        started.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        thread_log.lock().push("thread ended");
    })
    .unwrap();
    running.recv().unwrap();
    drop(pending);
    assert_eq!(*log.lock(), ["thread ended", "released"]);
}

#[test]
fn a_joined_pending_result_is_the_value_of_its_thread_and_continues_its_panic() {
    assert_eq!(Pending::spawn("test", (), || 7).unwrap().join(), 7);
    let panicking =
        Pending::spawn("test", (), || -> u32 { panic!("the driver panicked") }).unwrap();
    let joined = catch_unwind(AssertUnwindSafe(|| panicking.join()));
    let panic = joined.expect_err("the join returned instead of continuing the panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"the driver panicked"));
}

fn validation_error() -> wgpu::Error {
    wgpu::Error::Validation {
        source: "render pipeline creation failed".into(),
        description: "the adapter does not support the pipeline's format".into(),
    }
}

#[test]
fn a_window_adopts_the_threads_context_only_if_its_renderer_reported_no_error() {
    assert_eq!(adopted(Ok(1), None).unwrap(), 1);
    // Every other outcome rebuilds the context against the surface: a
    // validation error the renderer's creation did not return, and a
    // failed creation whether or not it also reported a validation error.
    let reported = validation_error().to_string();
    for (renderer, validation, expected) in [
        (Ok(1), Some(validation_error()), reported.as_str()),
        (
            Err(anyhow!("incompatible surface")),
            None,
            "incompatible surface",
        ),
        (
            Err(anyhow!("incompatible surface")),
            Some(validation_error()),
            reported.as_str(),
        ),
    ] {
        let error = adopted(renderer, validation).expect_err("an unchecked context was adopted");
        assert_eq!(error.to_string(), expected);
    }
}
