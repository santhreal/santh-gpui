//! X11 window placement and configure handling that does not need a server
//! connection: the `WM_NORMAL_HINTS` a new window sets, the background pixel
//! it is created with, and what a `ConfigureNotify` changes.

use gpui::{Bounds, DevicePixels, Pixels, Point, Size};
use x11rb::properties::{WmSizeHints, WmSizeHintsSpecification};

/// `WM_NORMAL_HINTS` for a new window at `origin` (root device pixels).
///
/// The position hint places the window where the application put it: a
/// window manager applies its own placement to a window without one, which
/// only a move after the map would undo. Sizes are in device pixels, so the
/// logical `min_size` is multiplied by `scale_factor`. `max_texture_size`
/// bounds the window to what the renderer can draw.
pub(super) fn normal_hints(
    origin: Point<DevicePixels>,
    min_size: Option<Size<Pixels>>,
    scale_factor: f32,
    max_texture_size: u32,
) -> WmSizeHints {
    let mut hints = WmSizeHints::new();
    hints.position = Some((
        WmSizeHintsSpecification::ProgramSpecified,
        origin.x.0,
        origin.y.0,
    ));
    hints.min_size = min_size.map(|size| {
        (
            (f32::from(size.width) * scale_factor).round() as i32,
            (f32::from(size.height) * scale_factor).round() as i32,
        )
    });
    hints.max_size = Some((max_texture_size as i32, max_texture_size as i32));
    hints
}

/// The background pixel a new window is created with, `None` for no
/// background.
///
/// A mapped window shows its background until its first frame is presented.
/// With no background, a compositor draws the new backing pixmap as the
/// server filled it, black on most servers. An ARGB pixel of 0 is fully
/// transparent to a compositor, so a transparent window on a 32-bit visual
/// shows nothing until the frame lands. Without a compositor the server
/// draws that pixel as black, and no background leaves the screen as it was.
pub(super) fn creation_background_pixel(
    transparent: bool,
    compositor_present: bool,
    argb_visual: bool,
) -> Option<u32> {
    (transparent && compositor_present && argb_visual).then_some(0)
}

/// Whether a new window's map waits for its surface's first configure,
/// which runs on a worker. Under a compositor, a window without a
/// background pixel shows its unfilled backing, black, until the first frame
/// is presented; it maps once the frame can be. Every other window shows
/// nothing before its first frame, maps at once, and draws when the
/// configure returns.
pub(super) fn map_waits_for_configure(
    transparent: bool,
    compositor_present: bool,
    argb_visual: bool,
) -> bool {
    compositor_present
        && creation_background_pixel(transparent, compositor_present, argb_visual).is_none()
}

/// What a `ConfigureNotify` changes on a window whose bounds are `current`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ConfigureChange {
    /// The size changed. The reported origin is relative to the window
    /// manager's frame on a resize, so only the size is read.
    Resized(Size<Pixels>),
    /// The size is unchanged and the window moved to this root position.
    Moved(Point<Pixels>),
    /// Neither changed, as on a restack or a repeated move request: nothing
    /// to lay out or draw.
    Unchanged,
}

pub(super) fn configure_change(
    current: Bounds<Pixels>,
    reported: Bounds<Pixels>,
) -> ConfigureChange {
    if reported.size != current.size {
        ConfigureChange::Resized(reported.size)
    } else if reported.origin != current.origin {
        ConfigureChange::Moved(reported.origin)
    } else {
        ConfigureChange::Unchanged
    }
}

/// The root position of an XInput 2 event, whose root coordinates are
/// 16.16 fixed point, rounded toward negative infinity to whole pixels.
pub(crate) fn xi_root_position(root_x: i32, root_y: i32) -> (i32, i32) {
    (root_x >> 16, root_y >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{point, px, size};
    use x11rb::x11_utils::Serialize as _;

    /// ICCCM 4.1.2.3 flag bits.
    const P_POSITION: u32 = 1 << 2;
    const P_MIN_SIZE: u32 = 1 << 4;
    const P_MAX_SIZE: u32 = 1 << 5;

    /// The WM_NORMAL_HINTS property as the window manager reads it: 18
    /// CARD32 words, flags first, then x, y, width, height, min and max.
    fn words(hints: &WmSizeHints) -> Vec<u32> {
        hints
            .serialize()
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect()
    }

    /// A window opens at the position the application gave it: the hint
    /// carries the program-specified position, including a negative origin
    /// on a monitor left of or above the primary one, and a window at the
    /// root origin is not moved off it.
    #[test]
    fn position_hint_is_the_requested_origin() {
        for (x, y) in [(0, 0), (640, 360), (-1920, 0), (3840, -200)] {
            let hints = normal_hints(point(DevicePixels(x), DevicePixels(y)), None, 1.0, 8192);
            let words = words(&hints);
            assert_eq!(words.len(), 18);
            assert_eq!(words[0] & P_POSITION, P_POSITION, "origin {x},{y}");
            assert_eq!((words[1] as i32, words[2] as i32), (x, y));
        }
    }

    /// The minimum size is in device pixels, so a window manager holds the
    /// logical minimum at every scale, fractional scales rounded.
    #[test]
    fn min_size_hint_is_in_device_pixels() {
        let min = size(px(300.0), px(201.0));
        for (scale, expected) in [
            (1.0_f32, (300, 201)),
            (1.5, (450, 302)),
            (2.0, (600, 402)),
            (1.25, (375, 251)),
        ] {
            let hints = normal_hints(
                point(DevicePixels(0), DevicePixels(0)),
                Some(min),
                scale,
                8192,
            );
            let words = words(&hints);
            assert_eq!(words[0] & P_MIN_SIZE, P_MIN_SIZE, "scale {scale}");
            assert_eq!(
                (words[5] as i32, words[6] as i32),
                expected,
                "scale {scale}"
            );
        }
    }

    #[test]
    fn max_size_hint_is_the_texture_limit_and_min_size_is_optional() {
        let hints = normal_hints(point(DevicePixels(10), DevicePixels(20)), None, 2.0, 16384);
        let words = words(&hints);
        assert_eq!(words[0] & P_MIN_SIZE, 0);
        assert_eq!(words[0] & P_MAX_SIZE, P_MAX_SIZE);
        assert_eq!((words[7], words[8]), (16384, 16384));
    }

    /// Only a transparent window on an ARGB visual under a compositor gets
    /// the transparent background pixel; every other combination has none.
    #[test]
    fn background_pixel_only_for_composited_transparent_argb_windows() {
        for transparent in [false, true] {
            for compositor in [false, true] {
                for argb in [false, true] {
                    let expected = (transparent && compositor && argb).then_some(0);
                    assert_eq!(
                        creation_background_pixel(transparent, compositor, argb),
                        expected,
                        "transparent {transparent} compositor {compositor} argb {argb}"
                    );
                }
            }
        }
    }

    /// Under a compositor, only a window whose backing shows black before
    /// its first frame, an opaque one or one without an ARGB visual, waits
    /// for the configure to map. Without a compositor none waits.
    #[test]
    fn only_a_composited_window_without_background_waits_to_map() {
        for transparent in [false, true] {
            for compositor in [false, true] {
                for argb in [false, true] {
                    assert_eq!(
                        map_waits_for_configure(transparent, compositor, argb),
                        compositor && !(transparent && argb),
                        "transparent {transparent} compositor {compositor} argb {argb}"
                    );
                }
            }
        }
    }

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
    }

    #[test]
    fn configure_classifies_resize_move_and_no_change() {
        let current = bounds(100.0, 50.0, 800.0, 600.0);
        let cases = [
            (current, ConfigureChange::Unchanged),
            (
                bounds(140.0, 90.0, 800.0, 600.0),
                ConfigureChange::Moved(point(px(140.0), px(90.0))),
            ),
            // A resize reports a frame-relative origin; the size alone is read.
            (
                bounds(0.0, 0.0, 801.0, 600.0),
                ConfigureChange::Resized(size(px(801.0), px(600.0))),
            ),
            (
                bounds(100.0, 50.0, 800.0, 599.0),
                ConfigureChange::Resized(size(px(800.0), px(599.0))),
            ),
        ];
        for (reported, expected) in cases {
            assert_eq!(
                configure_change(current, reported),
                expected,
                "{reported:?}"
            );
        }
    }

    /// XI2 root coordinates are 16.16 fixed point. Fractional pixels
    /// round toward negative infinity, left of the primary monitor too.
    #[test]
    fn xi_root_position_reads_fixed_point() {
        let fp = |whole: i32, frac: i32| (whole << 16) | frac;
        assert_eq!(xi_root_position(fp(0, 0), fp(0, 0)), (0, 0));
        assert_eq!(
            xi_root_position(fp(1234, 0x8000), fp(56, 0xffff)),
            (1234, 56)
        );
        assert_eq!(xi_root_position(fp(-1, 0x8000), fp(-300, 0)), (-1, -300));
        assert_eq!(xi_root_position(fp(3839, 0), fp(2159, 0)), (3839, 2159));
    }
}
