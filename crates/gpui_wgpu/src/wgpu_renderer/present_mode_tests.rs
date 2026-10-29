//! The present mode a window surface runs, over every ordered preference
//! list and every set of modes a surface can offer.
//!
//! A surface runs the first preferred mode it offers and `Fifo` when it
//! offers none, so a configure never requests a mode the surface lacks,
//! which wgpu rejects. Catches a preference applied without checking the
//! surface, a fallback other than `Fifo`, a later preference chosen over an
//! earlier offered one, and an offered preference dropped for `Fifo`. Does
//! not check which modes each platform prefers.

use super::select_present_mode;
use wgpu::PresentMode;

/// The modes a surface can report. Every surface offers `Fifo`.
const OFFERABLE: [PresentMode; 3] = [
    PresentMode::FifoRelaxed,
    PresentMode::Immediate,
    PresentMode::Mailbox,
];

/// Every mode a window can prefer, the automatic modes included.
const PREFERABLE: [PresentMode; 6] = [
    PresentMode::AutoVsync,
    PresentMode::AutoNoVsync,
    PresentMode::Fifo,
    PresentMode::FifoRelaxed,
    PresentMode::Immediate,
    PresentMode::Mailbox,
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

/// Every ordered list of distinct [`PREFERABLE`] modes, the empty list
/// included.
fn preferences() -> Vec<Vec<PresentMode>> {
    fn extend(prefix: &mut Vec<PresentMode>, out: &mut Vec<Vec<PresentMode>>) {
        out.push(prefix.clone());
        for mode in PREFERABLE {
            if !prefix.contains(&mode) {
                prefix.push(mode);
                extend(prefix, out);
                prefix.pop();
            }
        }
    }
    let mut out = Vec::new();
    extend(&mut Vec::new(), &mut out);
    out
}

#[test]
fn a_surface_runs_its_first_offered_preference_and_fifo_otherwise() {
    let preferences = preferences();
    // 1 + 6 + 30 + 120 + 360 + 720 + 720 ordered lists of distinct modes.
    assert_eq!(preferences.len(), 1957);
    for offered in offers() {
        for preferred in &preferences {
            let mode = select_present_mode(preferred, &offered);
            let context = format!("preferring {preferred:?} on a surface offering {offered:?}");
            assert!(offered.contains(&mode), "{context} ran unoffered {mode:?}");
            match preferred.iter().position(|mode| offered.contains(mode)) {
                Some(first) => assert_eq!(mode, preferred[first], "{context}"),
                None => assert_eq!(mode, PresentMode::Fifo, "{context}"),
            }
        }
    }
}
