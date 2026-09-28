//! WHY: the first window of an X11 client adopts a GPU context whose
//! adapter was selected without a surface, and does not wait for its
//! surface's first configure. These tests close three defect classes of
//! the trial that replaces that wait:
//!
//! - a context kept although the first configure on it failed, which
//!   leaves the window drawing failed frames on an adapter that cannot
//!   present to it;
//! - a context rejected before its configure is read, or rejected after a
//!   configure on it has succeeded, which throws away a working device;
//! - a renderer on a context selected by configuring its surface that is
//!   rejected when its own configure fails, which rebuilds the context on
//!   every failed configure instead of counting failed frames.
//!
//! Not covered here: the renderer reading its configure's result and
//! recovering a rejected context, which `window_tests` drives on an X
//! server.

use super::Trial;

/// Every trial. A new variant fails to compile in `listed` until it is
/// added here, and so until these tests decide it.
const TRIALS: [Trial; 3] = [Trial::Settled, Trial::Pending, Trial::Rejected];

#[allow(dead_code)]
fn listed(trial: Trial) {
    match trial {
        Trial::Settled | Trial::Pending | Trial::Rejected => {}
    }
}

#[test]
fn only_a_window_renderer_on_an_untested_context_is_on_trial() {
    for window_surface in [false, true] {
        for surface_tested in [false, true] {
            let expected = if window_surface && !surface_tested {
                Trial::Pending
            } else {
                Trial::Settled
            };
            assert_eq!(
                Trial::new(window_surface, surface_tested),
                expected,
                "window_surface={window_surface} surface_tested={surface_tested}"
            );
        }
    }
}

#[test]
fn a_pending_trial_rejects_the_context_only_if_no_configure_on_it_succeeded() {
    for (failed, surface_tested, expected) in [
        (true, false, Trial::Rejected),
        (true, true, Trial::Settled),
        (false, false, Trial::Settled),
        (false, true, Trial::Settled),
    ] {
        assert_eq!(
            Trial::Pending.settle(failed, surface_tested),
            expected,
            "failed={failed} surface_tested={surface_tested}"
        );
    }
}

#[test]
fn a_decided_trial_does_not_change() {
    for trial in TRIALS {
        if trial == Trial::Pending {
            continue;
        }
        for failed in [false, true] {
            for surface_tested in [false, true] {
                assert_eq!(
                    trial.settle(failed, surface_tested),
                    trial,
                    "failed={failed} surface_tested={surface_tested}"
                );
            }
        }
    }
}
