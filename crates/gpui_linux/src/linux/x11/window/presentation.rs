//! How an X11 window presents: the present modes its surface prefers, and
//! the `_NET_WM_BYPASS_COMPOSITOR` hint the resolved mode needs.

use gpui_wgpu::wgpu::PresentMode;

/// The present modes a window prefers, first preferred first. Its surface
/// presents with the first one it offers, else `Fifo`.
///
/// `Fifo` blocks the drawing thread in `get_current_texture` until the
/// display releases a swapchain image, with input queued behind it,
/// whenever frames outpace the display. Under a compositor that presents
/// below the rate of the window's CRTC, as beside a monitor of a lower
/// refresh rate, that is most of every frame. `Mailbox` replaces the queued
/// image instead. NVIDIA's X11 surfaces offer no `Mailbox`; under a
/// compositor they present `Immediate`, which the compositor copies into
/// its own vsynced frame, so the present does not tear. Without a
/// compositor an `Immediate` present scans out mid-refresh and tears, so
/// such a surface presents `Fifo`.
pub(super) fn present_modes(compositor_present: bool) -> &'static [PresentMode] {
    if compositor_present {
        &[PresentMode::Mailbox, PresentMode::Immediate]
    } else {
        &[PresentMode::Mailbox]
    }
}

/// The `_NET_WM_BYPASS_COMPOSITOR` value of a window that presents with
/// `mode`, `None` for no hint. A compositor unredirects a fullscreen window
/// by default, and an unredirected window scans out its own presents. The
/// value 2 keeps the window composited, so its `Immediate` presents stay
/// copied and do not tear.
pub(super) fn bypass_compositor(mode: PresentMode) -> Option<u32> {
    (mode == PresentMode::Immediate).then_some(2)
}

#[cfg(test)]
mod tests;
