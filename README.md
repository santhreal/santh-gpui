# santh-gpui

santh-gpui is the GPU-accelerated UI framework of every Santh desktop
application: iris, the Veyyon desktop, and aria. It is derived from
[Zed](https://github.com/zed-industries/zed)'s GPUI; this repository holds
the framework crates and their dependencies only, with the upstream history.

## Packages

| Package | Contents |
| ------- | -------- |
| `gpui` | Application and window contexts, entities, elements, layout, input, text, animation. |
| `motion` | Easing curves, springs, tweens, and motion tokens with no GPUI dependency; `gpui::motion` re-exports it with the frame clock and driver. |
| `gpui_platform` | `application()` and `headless()`: the platform for the current OS. |
| `gpui_linux` | X11 and Wayland windows, input, clipboard, and display integration. |
| `gpui_macos`, `gpui_apple` | AppKit windows and the Metal renderer. |
| `gpui_windows` | Win32 windows and the DirectX 11 renderer. |
| `gpui_wgpu` | The wgpu (Vulkan) renderer used on Linux, the cosmic-text text system, and offscreen rendering. |
| `gpui_web` | The browser platform. |
| `gpui_tokio` | A Tokio runtime driven from GPUI executors. |
| `gpui_macros` | `IntoElement`, `Render`, and action derives. |

`collections`, `refineable`, `sum_tree`, `scheduler`, `util`, `path`,
`http_client`, and `media` are internal dependencies of the packages above.

## Use

Pin one revision of this repository for every GPUI package:

```toml
[dependencies]
gpui = { git = "https://github.com/santhreal/santh-gpui.git", rev = "<rev>" }
gpui_platform = { git = "https://github.com/santhreal/santh-gpui.git", rev = "<rev>", features = ["x11", "wayland"] }
```

`gpui_platform` enables no Linux window system by default; enable `x11`,
`wayland`, or both. Add `gpui_wgpu` and `gpui_tokio` at the same revision
when you use offscreen rendering or Tokio.

```rust
use gpui::{App, Context, Window, WindowOptions, div, prelude::*};

struct Hello;

impl Render for Hello {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child("Hello")
    }
}

fn main() {
    gpui_platform::application().run(|cx: &mut App| {
        cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| Hello))
            .expect("open window");
    });
}
```

`crates/gpui/examples` has runnable examples: `cargo run -p gpui --example hello_world`.

## Rules for applications

- Depend on this repository only. Do not depend on crates.io `gpui`,
  upstream Zed, another fork, or a vendored copy of any package here.
- Framework fixes, platform fixes, render primitives, animation, and
  performance work land here on `main`; applications then move their `rev`.
- An application does not carry a patch to a package here.

## Build

The toolchain is pinned in `rust-toolchain.toml`.

```sh
cargo check --workspace --all-targets
cargo test -p gpui
```

Linux builds need the X11, Wayland, xkbcommon, fontconfig, and Vulkan loader
development packages; `.github/workflows/ci.yml` lists them.

## License

The packages are licensed under Apache-2.0 (`LICENSE-APACHE`). The fonts in
`assets/fonts` are licensed under the SIL Open Font License 1.1; each font
directory holds its license.
