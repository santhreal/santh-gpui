//! The present mode a window surface runs, over every preference and every
//! set of modes a surface can offer.
//!
//! A surface runs its preferred mode when it offers that mode and `Fifo`
//! otherwise, so a configure never requests a mode the surface lacks, which
//! wgpu rejects. Catches a preference applied without checking the surface,
//! a fallback other than `Fifo`, and a supported preference dropped for
//! `Fifo`. Does not check which mode each platform prefers.

use super::present_mode;
use wgpu::PresentMode;

/// The modes a surface can report. Every surface offers `Fifo`.
const OFFERABLE: [PresentMode; 3] = [
    PresentMode::FifoRelaxed,
    PresentMode::Immediate,
    PresentMode::Mailbox,
];

/// Every preference a window can state, the automatic modes included.
const PREFERENCES: [Option<PresentMode>; 7] = [
    None,
    Some(PresentMode::AutoVsync),
    Some(PresentMode::AutoNoVsync),
    Some(PresentMode::Fifo),
    Some(PresentMode::FifoRelaxed),
    Some(PresentMode::Immediate),
    Some(PresentMode::Mailbox),
];

/// Every set of modes a surface can offer: `Fifo` plus each subset of
/// [`OFFERABLE`].
fn offers() -> Vec<Vec<PresentMode>> {
    (0..1u32 << OFFERABLE.len())
        .map(|bits| {
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
        .collect()
}

#[test]
fn a_surface_runs_its_preference_when_offered_and_fifo_otherwise() {
    for supported in offers() {
        for preferred in PREFERENCES {
            let expected = match preferred {
                Some(mode) if supported.contains(&mode) => mode,
                _ => PresentMode::Fifo,
            };
            assert_eq!(
                present_mode(preferred, &supported),
                expected,
                "preferring {preferred:?} on a surface offering {supported:?}"
            );
        }
    }
}
