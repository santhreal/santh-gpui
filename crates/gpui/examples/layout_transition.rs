#![cfg_attr(target_family = "wasm", no_main)]

//! Cards that move to their new places on a spring when their order or the
//! space above them changes. Click a card to move it to the front; click the
//! banner to show or hide it.

use gpui::{
    App, Bounds, Context, Hsla, MotionExt as _, SharedString, Window, WindowBounds, WindowOptions,
    div, hsla, prelude::*, px, rgb, size,
};
use gpui_platform::application;

struct Card {
    id: usize,
    title: SharedString,
    color: Hsla,
}

struct LayoutTransitionExample {
    cards: Vec<Card>,
    banner: bool,
}

impl LayoutTransitionExample {
    fn new() -> Self {
        let cards = (0..6)
            .map(|id| Card {
                id,
                title: format!("Card {}", id + 1).into(),
                color: hsla(id as f32 / 6.0, 0.6, 0.55, 1.0),
            })
            .collect();
        Self {
            cards,
            banner: true,
        }
    }
}

impl Render for LayoutTransitionExample {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let banner = div()
            .id("banner")
            .h(if self.banner { px(64.0) } else { px(24.0) })
            .w_full()
            .px_3()
            .flex()
            .items_center()
            .bg(rgb(0x2a2f3a))
            .text_color(rgb(0xd0d4dc))
            .child(if self.banner {
                "Banner: click to collapse"
            } else {
                "Click to expand"
            })
            .on_click(cx.listener(|this, _, _, cx| {
                this.banner = !this.banner;
                cx.notify();
            }));

        let cards = self.cards.iter().enumerate().map(|(index, card)| {
            div()
                .id(card.id)
                .w(px(140.0))
                .h(px(if index == 0 { 96.0 } else { 64.0 }))
                .rounded_md()
                .p_2()
                .bg(card.color)
                .text_color(rgb(0xffffff))
                .child(card.title.clone())
                .on_click(cx.listener(move |this, _, _, cx| {
                    let card = this.cards.remove(index);
                    this.cards.insert(0, card);
                    cx.notify();
                }))
                // Keyed by the card, so the card moves wherever it lands.
                .animate_layout(("card", card.id))
        });

        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_3()
            .bg(rgb(0x14161b))
            .child(banner)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .gap_3()
                    .p_3()
                    .children(cards),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(520.0), px(420.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| LayoutTransitionExample::new()),
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
