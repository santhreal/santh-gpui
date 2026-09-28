#![cfg_attr(target_family = "wasm", no_main)]

//! Notes that fade in when added and fade out when removed, while the notes
//! around them move to their new places on a spring, and a note count that
//! crossfades when it changes. Click "Add note" to add one; click a note to
//! remove it.

use gpui::{
    App, Bounds, Context, Hsla, MotionExt as _, SharedString, Window, WindowBounds, WindowOptions,
    crossfade, div, hsla, prelude::*, presence, px, rgb, size,
};
use gpui_platform::application;

struct Note {
    id: usize,
    title: SharedString,
    color: Hsla,
}

struct PresenceExample {
    notes: Vec<Note>,
    next_id: usize,
}

impl PresenceExample {
    fn new() -> Self {
        let mut example = Self {
            notes: Vec::new(),
            next_id: 0,
        };
        for _ in 0..3 {
            example.add_note();
        }
        example
    }

    fn add_note(&mut self) {
        let id = self.next_id;
        self.next_id += 1;
        self.notes.push(Note {
            id,
            title: format!("Note {}", id + 1).into(),
            color: hsla((id % 7) as f32 / 7.0, 0.55, 0.5, 1.0),
        });
    }
}

impl Render for PresenceExample {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let this = cx.weak_entity();
        let add = div()
            .id("add")
            .px_3()
            .py_2()
            .rounded_md()
            .bg(rgb(0x2a2f3a))
            .text_color(rgb(0xd0d4dc))
            .child("Add note")
            .on_click(cx.listener(|this, _, _, cx| {
                this.add_note();
                cx.notify();
            }));

        let notes = self.notes.iter().map(|note| {
            let (id, title, color, this) = (note.id, note.title.clone(), note.color, this.clone());
            // The builder runs every frame the note is painted, including
            // while it fades out after removal.
            let build = move |_: &mut Window, _: &mut App| {
                let this = this.clone();
                div()
                    .id(("note", id))
                    .w(px(240.0))
                    .h(px(40.0))
                    .px_3()
                    .flex()
                    .items_center()
                    .rounded_md()
                    .bg(color)
                    .text_color(rgb(0xffffff))
                    .child(title.clone())
                    .on_click(move |_, _, cx| {
                        let _ = this.update(cx, |example, cx| {
                            example.notes.retain(|note| note.id != id);
                            cx.notify();
                        });
                    })
                    .animate_layout(("note-layout", id))
            };
            (("note", id), build)
        });

        let count = self.notes.len();
        // Each count is its own key, so a change fades the previous count
        // out beneath the new one.
        let summary = crossfade("count", count, move |_: &mut Window, _: &mut App| {
            div()
                .text_color(rgb(0xd0d4dc))
                .child(format!("{count} notes"))
        });

        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_3()
            .p_3()
            .bg(rgb(0x14161b))
            .child(
                div()
                    .flex()
                    .gap_3()
                    .items_center()
                    .child(add)
                    .child(summary),
            )
            .child(presence("notes").flex().flex_col().gap_2().children(notes))
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(320.0), px(480.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| PresenceExample::new()),
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
