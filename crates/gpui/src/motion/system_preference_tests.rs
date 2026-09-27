//! The reduced-motion preference of the operating system in the app motion
//! policy.
//!
//! WHY: closes the classes "the app starts with full motion while the system
//! requests reduced motion", "a change of the system preference does not reach
//! the app motion policy", "a system change resets the duration scale", "a
//! system change leaves windows drawn with the previous policy", and "a
//! repeated system report overrides the reduced flag the app set". Every case
//! runs through the test platform, which reports the preference through
//! [`crate::Platform::on_reduce_motion_change`] also when it is unchanged, as
//! the platform contract allows. Not caught: a platform that reads its
//! operating-system setting wrongly or never reports a change; each platform
//! crate tests its own mapping.

use std::sync::Arc;

use super::MotionPolicy;
use crate::{
    App, BackgroundExecutor, Context, ForegroundExecutor, IntoElement, Render, TestAppContext,
    TestDispatcher, TestPlatform, Window, WindowHandle, div, px, size,
};

/// A view that records the reduced flag of the app motion policy at each
/// render.
#[derive(Default)]
struct PolicyView {
    rendered: Vec<bool>,
}

impl Render for PolicyView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.rendered.push(cx.reduce_motion());
        div()
    }
}

fn open_policy_view(cx: &mut TestAppContext) -> WindowHandle<PolicyView> {
    let window = cx.open_window(size(px(100.0), px(100.0)), |_, _| PolicyView::default());
    cx.run_until_parked();
    window
}

/// The reduced flags the view rendered with, oldest first.
fn rendered(window: &WindowHandle<PolicyView>, cx: &mut TestAppContext) -> Vec<bool> {
    window
        .update(cx, |view, _, _| view.rendered.clone())
        .unwrap()
}

#[test]
fn the_app_starts_with_the_reduced_flag_of_the_system() {
    for reduce_motion in [false, true] {
        let dispatcher = Arc::new(TestDispatcher::new(0));
        let platform = TestPlatform::new(
            BackgroundExecutor::new(dispatcher.clone()),
            ForegroundExecutor::new(dispatcher),
        );
        platform.simulate_reduce_motion_change(reduce_motion);
        let app = App::new_app(
            platform,
            Arc::new(()),
            http_client::FakeHttpClient::with_404_response(),
        );
        assert_eq!(
            app.borrow().motion_policy(),
            MotionPolicy::DEFAULT.with_reduced(reduce_motion),
            "an app started while the system reduce-motion preference is {reduce_motion}"
        );
    }
}

#[gpui::test]
fn a_system_change_sets_the_reduced_flag_keeps_the_duration_scale_and_redraws(
    cx: &mut TestAppContext,
) {
    let slow = MotionPolicy::DEFAULT.with_duration_scale(4.0);
    cx.update(|cx| cx.set_motion_policy(slow));
    let window = open_policy_view(cx);

    for reduce_motion in [true, false, true, false] {
        let before = rendered(&window, cx).len();
        cx.simulate_reduce_motion_change(reduce_motion);
        cx.run_until_parked();
        assert_eq!(
            cx.update(|cx| cx.motion_policy()),
            slow.with_reduced(reduce_motion),
            "a system change to {reduce_motion} sets the reduced flag and keeps the duration scale"
        );
        assert_eq!(
            rendered(&window, cx)[before..],
            [reduce_motion],
            "a system change to {reduce_motion} redraws the window once with the new flag"
        );
    }
}

#[gpui::test]
fn a_repeated_system_report_neither_changes_the_policy_nor_redraws(cx: &mut TestAppContext) {
    let window = open_policy_view(cx);
    for reduce_motion in [false, true] {
        cx.simulate_reduce_motion_change(reduce_motion);
        cx.run_until_parked();
        let before = rendered(&window, cx).len();
        cx.simulate_reduce_motion_change(reduce_motion);
        cx.run_until_parked();
        assert_eq!(cx.update(|cx| cx.reduce_motion()), reduce_motion);
        assert_eq!(
            rendered(&window, cx).len(),
            before,
            "a repeated system report of {reduce_motion} redraws nothing"
        );
    }
}

#[gpui::test]
fn the_reduced_flag_the_app_sets_holds_until_the_system_preference_changes(
    cx: &mut TestAppContext,
) {
    for system in [true, false] {
        cx.simulate_reduce_motion_change(system);
        cx.update(|cx| cx.set_reduce_motion(!system));

        cx.simulate_reduce_motion_change(system);
        cx.run_until_parked();
        assert_eq!(
            cx.update(|cx| cx.reduce_motion()),
            !system,
            "a repeated system report of {system} keeps the flag the app set"
        );

        cx.simulate_reduce_motion_change(!system);
        cx.simulate_reduce_motion_change(system);
        cx.run_until_parked();
        assert_eq!(
            cx.update(|cx| cx.reduce_motion()),
            system,
            "the next system change sets the flag to the system preference {system}"
        );
    }
}
