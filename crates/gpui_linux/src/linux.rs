#[cfg(any(feature = "wayland", feature = "x11"))]
mod clipboard_file_list;
mod dispatcher;
#[cfg(any(feature = "wayland", feature = "x11"))]
mod gpu_context;
mod headless;
mod keyboard;
mod platform;
mod system_notifications;
#[cfg(any(feature = "wayland", feature = "x11"))]
mod text_system;
#[cfg(feature = "wayland")]
mod wayland;
#[cfg(feature = "x11")]
mod x11;

#[cfg(any(feature = "wayland", feature = "x11"))]
mod xdg_desktop_portal;

#[cfg(all(test, feature = "wayland", feature = "x11"))]
mod windowed_platform_tests;

pub use dispatcher::*;
pub(crate) use headless::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
#[cfg(any(feature = "wayland", feature = "x11"))]
pub(crate) use text_system::*;
#[cfg(feature = "wayland")]
pub(crate) use wayland::*;
#[cfg(feature = "x11")]
pub(crate) use x11::*;

use std::rc::Rc;

/// Returns the default platform implementation for the current OS: headless
/// when `headless` is set or no display server is named, otherwise connected
/// to the display server. Panics when that connection fails;
/// [`windowed_platform`] returns the failure instead.
pub fn current_platform(headless: bool) -> Rc<dyn gpui::Platform> {
    if headless || gpui::guess_compositor() == "Headless" {
        return Rc::new(LinuxPlatform {
            inner: HeadlessClient::new(),
        });
    }
    windowed_platform().unwrap_or_else(|err| panic!("{err:#}"))
}

/// Returns the platform connected to the display server `WAYLAND_DISPLAY` or
/// `DISPLAY` points at, Wayland first. Fails when neither is set, when
/// `ZED_HEADLESS` is set, or when the display server does not accept the
/// connection, so a windowed application can report why it has no window
/// instead of running headless.
pub fn windowed_platform() -> anyhow::Result<Rc<dyn gpui::Platform>> {
    #[cfg(feature = "x11")]
    use anyhow::Context as _;

    match gpui::guess_compositor() {
        #[cfg(feature = "wayland")]
        "Wayland" => Ok(Rc::new(LinuxPlatform {
            inner: WaylandClient::new()?,
        })),

        #[cfg(feature = "x11")]
        "X11" => Ok(Rc::new(LinuxPlatform {
            inner: X11Client::new().context("cannot connect to the X server at DISPLAY")?,
        })),

        "Headless" if std::env::var_os("ZED_HEADLESS").is_some() => {
            anyhow::bail!("ZED_HEADLESS is set, so no window can open")
        }
        "Headless" => anyhow::bail!("neither WAYLAND_DISPLAY nor DISPLAY is set"),
        _ => unreachable!(
            r#"At least one of the "wayland" or "x11" features must be enabled on gpui_linux or gpui_platform."#
        ),
    }
}
