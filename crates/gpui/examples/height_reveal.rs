#![cfg_attr(target_family = "wasm", no_main)]

//! Sections that open and close on a spring. Click a header to open or close
//! its section; click "Add a line" inside an open section to grow it.

use gpui::{
    App, Bounds, Context, SharedString, Window, WindowBounds, WindowOptions, div, height_reveal,
    prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Section {
    title: SharedString,
    lines: usize,
    open: bool,
}

struct HeightRevealExample {
    sections: Vec<Section>,
}

impl HeightRevealExample {
    fn new() -> Self {
        let sections = ["Overview", "Details", "History"]
            .into_iter()
            .enumerate()
            .map(|(index, title)| Section {
                title: title.into(),
                lines: 2 + index,
                open: index == 0,
            })
            .collect();
        Self { sections }
    }
}

impl Render for HeightRevealExample {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let sections = self.sections.iter().enumerate().map(|(index, section)| {
            let header = div()
                .id(("header", index))
                .px_3()
                .py_2()
                .bg(rgb(0x2a2f3a))
                .text_color(rgb(0xd0d4dc))
                .child(format!(
                    "{} {}",
                    if section.open { "v" } else { ">" },
                    section.title
                ))
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.sections[index].open = !this.sections[index].open;
                    cx.notify();
                }));

            let body = div()
                .flex()
                .flex_col()
                .gap_1()
                .p_3()
                .bg(rgb(0x1c1f26))
                .text_color(rgb(0xa8aeb8))
                .children((0..section.lines).map(|line| format!("Line {}", line + 1)))
                .child(
                    div()
                        .id(("add", index))
                        .text_color(rgb(0x6fa8ff))
                        .child("Add a line")
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.sections[index].lines += 1;
                            cx.notify();
                        })),
                );

            div()
                .flex()
                .flex_col()
                .rounded_md()
                .overflow_hidden()
                .child(header)
                // Keyed by the section, so each keeps its own motion.
                .child(height_reveal(("body", index), section.open, body))
        });

        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .bg(rgb(0x14161b))
            .children(sections)
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(420.0), px(520.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| HeightRevealExample::new()),
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
