#![cfg_attr(target_family = "wasm", no_main)]

//! A scrollable div, a uniform list and a list that ease mouse wheel ticks on
//! a spring. A touchpad scrolls them at once. The list follows its tail: a
//! tick up detaches it, and scrolling back to the end attaches it again.

use gpui::{
    App, Bounds, Context, Div, FollowMode, ListAlignment, ListState, ScrollHandle,
    UniformListScrollHandle, Window, WindowBounds, WindowOptions, div, list, prelude::*, px, rgb,
    size, uniform_list,
};
use gpui_platform::application;

const ROWS: usize = 200;
const ROW_HEIGHT: f32 = 32.0;

struct SmoothWheelExample {
    scroll: ScrollHandle,
    uniform: UniformListScrollHandle,
    list: ListState,
}

impl SmoothWheelExample {
    fn new() -> Self {
        let scroll = ScrollHandle::new();
        scroll.set_smooth_wheel(true);
        let uniform = UniformListScrollHandle::new();
        uniform.set_smooth_wheel(true);
        let list = ListState::new(ROWS, ListAlignment::Top, px(200.0));
        list.set_smooth_wheel(true);
        list.set_follow_mode(FollowMode::Tail);
        Self {
            scroll,
            uniform,
            list,
        }
    }
}

fn row(ix: usize, height: f32) -> Div {
    div()
        .flex_shrink_0()
        .h(px(height))
        .px_3()
        .border_b_1()
        .border_color(rgb(0xd0d0d0))
        .child(format!("Row {ix}"))
}

fn column(title: &'static str, body: impl IntoElement) -> Div {
    div()
        .flex()
        .flex_col()
        .flex_1()
        .h_full()
        .bg(rgb(0xffffff))
        .child(div().p_2().border_b_1().border_color(rgb(0xa0a0a0)).child(title))
        .child(body)
}

impl Render for SmoothWheelExample {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .size_full()
            .gap_2()
            .p_2()
            .bg(rgb(0xf0f0f0))
            .text_color(rgb(0x202020))
            .child(column(
                "div",
                div()
                    .id("div")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .children((0..ROWS).map(|ix| row(ix, ROW_HEIGHT))),
            ))
            .child(column(
                "uniform_list",
                uniform_list("uniform", ROWS, |range, _, _| {
                    range.map(|ix| row(ix, ROW_HEIGHT)).collect()
                })
                .track_scroll(&self.uniform)
                .flex_1(),
            ))
            .child(column(
                "list, following its tail",
                list(self.list.clone(), |ix, _, _| {
                    row(ix, ROW_HEIGHT + (ix % 3) as f32 * 12.0).into_any_element()
                })
                .flex_1(),
            ))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.0), px(480.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| SmoothWheelExample::new()),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
