//! WHY: X11 windows preferred `Mailbox` alone. NVIDIA's X11 surfaces offer
//! no `Mailbox`, so they presented `Fifo`, and under a compositor that
//! presents at 60 Hz a window on a 240 Hz monitor blocked its drawing
//! thread in `get_current_texture` for about 14 ms of every frame, input
//! queued behind it. These tests resolve each window state against every
//! set of modes a surface can offer through the renderer's own selection,
//! and check that a window never blocks on `Fifo` under a compositor while
//! a non-blocking mode is offered, never presents a tearing mode without
//! one, and sets the compositor hint exactly when it presents `Immediate`.
//!
//! Not covered: whether a compositor honors the hint, and whether a
//! surface's `Immediate` presents tear under a given compositor.

use super::{bypass_compositor, present_modes};
use gpui_wgpu::{select_present_mode, wgpu::PresentMode};

/// The modes a surface can offer beside `Fifo`, which every surface offers.
const OFFERABLE: [PresentMode; 3] = [
    PresentMode::FifoRelaxed,
    PresentMode::Immediate,
    PresentMode::Mailbox,
];

/// Every set of modes a surface can offer.
fn offers() -> impl Iterator<Item = Vec<PresentMode>> {
    (0..1u32 << OFFERABLE.len()).map(|bits| {
        std::iter::once(PresentMode::Fifo)
            .chain(
                OFFERABLE
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| bits & (1 << index) != 0)
                    .map(|(_, mode)| *mode),
            )
            .collect()
    })
}

/// Modes whose presents reach the screen mid-refresh where no compositor
/// copies them.
fn tears(mode: PresentMode) -> bool {
    matches!(mode, PresentMode::Immediate | PresentMode::FifoRelaxed)
}

#[test]
fn a_composited_window_never_blocks_while_a_nonblocking_mode_is_offered() {
    for offered in offers() {
        let mode = select_present_mode(present_modes(true), &offered);
        let expected = if offered.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else if offered.contains(&PresentMode::Immediate) {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        };
        assert_eq!(mode, expected, "under a compositor, offered {offered:?}");
    }
}

#[test]
fn a_window_without_a_compositor_never_tears() {
    for offered in offers() {
        let mode = select_present_mode(present_modes(false), &offered);
        assert!(
            !tears(mode),
            "without a compositor, offered {offered:?}, presents tearing {mode:?}"
        );
        let expected = if offered.contains(&PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else {
            PresentMode::Fifo
        };
        assert_eq!(mode, expected, "without a compositor, offered {offered:?}");
    }
}

#[test]
fn only_an_immediate_present_keeps_the_window_composited() {
    assert_eq!(bypass_compositor(PresentMode::Immediate), Some(2));
    for mode in [
        PresentMode::AutoVsync,
        PresentMode::AutoNoVsync,
        PresentMode::Fifo,
        PresentMode::FifoRelaxed,
        PresentMode::Mailbox,
    ] {
        assert_eq!(bypass_compositor(mode), None, "{mode:?}");
    }
}
