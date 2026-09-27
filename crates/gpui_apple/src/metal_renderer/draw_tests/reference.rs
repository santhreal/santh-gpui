//! CPU reference math for the draw tests: the documented layer mask
//! composite, the edge fade mask value, and polygon coverage of path
//! outlines.

use super::{EDGE, LOGICAL_SIZE};
use gpui::{Bounds, EdgeFadeMask, Hsla, Path, Pixels, ScaledPixels, point, px, size, white};

pub(super) type Rgba = [f32; 4];

/// A drawn frame with unorm RGBA channels, row by row.
pub(super) struct Image {
    pub(super) width: i32,
    pub(super) pixels: Vec<Rgba>,
}

impl Image {
    /// The pixel at `(x, y)`, clamped to the frame edge.
    pub(super) fn at(&self, x: i32, y: i32) -> Rgba {
        let height = self.pixels.len() as i32 / self.width;
        let (x, y) = (x.clamp(0, self.width - 1), y.clamp(0, height - 1));
        self.pixels[(y * self.width + x) as usize]
    }
}

/// `src` drawn over `dst` with the quad blend state.
pub(super) fn over(dst: Rgba, src: Rgba) -> Rgba {
    let a = src[3];
    let [r, g, b] = [0, 1, 2].map(|c| src[c] * a + dst[c] * (1.0 - a));
    [r, g, b, (a + dst[3]).min(1.0)]
}

pub(super) fn rgba(color: Hsla) -> Rgba {
    let color = color.to_rgb();
    [color.r, color.g, color.b, color.a]
}

/// The documented `EndLayerMask` composite of the premultiplied `layer` onto
/// `dst` at mask value `m`.
pub(super) fn composite(dst: Rgba, layer: Rgba, m: f32) -> Rgba {
    let [r, g, b] = [0, 1, 2].map(|c| layer[c] * m + dst[c] * (1.0 - layer[3] * m));
    [r, g, b, (layer[3] * m + dst[3]).min(1.0)]
}

/// The documented mask value `m` of `mask` at device pixel `(x, y)`: zero
/// when the pixel center is outside `mask.bounds`, else `r * r`, with `r`
/// the minimum over the edges of `mask.fade_bounds` with a nonzero band of
/// the clamped distance from the pixel center to that edge over the band.
pub(super) fn fade_mask(mask: &EdgeFadeMask, x: i32, y: i32) -> f32 {
    let p = (x as f32 + 0.5, y as f32 + 0.5);
    let bounds = &mask.bounds;
    let inside = p.0 >= bounds.left().0
        && p.0 < bounds.right().0
        && p.1 >= bounds.top().0
        && p.1 < bounds.bottom().0;
    if !inside {
        return 0.0;
    }
    let fade = &mask.fade_bounds;
    let bands = &mask.bands;
    let r = [
        (bands.top.0, p.1 - fade.top().0),
        (bands.right.0, fade.right().0 - p.0),
        (bands.bottom.0, fade.bottom().0 - p.1),
        (bands.left.0, p.0 - fade.left().0),
    ]
    .into_iter()
    .filter(|(band, _)| *band > 0.0)
    .map(|(band, distance)| (distance / band).clamp(0.0, 1.0))
    .fold(1.0, f32::min);
    r * r
}

/// A closed outline drawn as a gpui path, with the polygon it encloses, in
/// logical pixels.
pub(super) struct Outline {
    path: Path<Pixels>,
    polygon: Vec<(f32, f32)>,
}

impl Outline {
    pub(super) fn new(start: (f32, f32)) -> Self {
        Self {
            path: Path::new(point(px(start.0), px(start.1))),
            polygon: vec![start],
        }
    }

    pub(super) fn line_to(mut self, to: (f32, f32)) -> Self {
        self.path.line_to(point(px(to.0), px(to.1)));
        self.polygon.push(to);
        self
    }

    pub(super) fn curve_to(mut self, to: (f32, f32), ctrl: (f32, f32)) -> Self {
        let from = self.polygon[self.polygon.len() - 1];
        self.path
            .curve_to(point(px(to.0), px(to.1)), point(px(ctrl.0), px(ctrl.1)));
        self.polygon.extend((1..=64).map(|step| {
            let t = step as f32 / 64.0;
            let u = 1.0 - t;
            (
                u * u * from.0 + 2.0 * u * t * ctrl.0 + t * t * to.0,
                u * u * from.1 + 2.0 * u * t * ctrl.1 + t * t * to.1,
            )
        }));
        self
    }

    /// A square with corners rounded by quadratic curves.
    pub(super) fn rounded_square(lo: f32, hi: f32, radius: f32) -> Self {
        Self::new((lo + radius, lo))
            .line_to((hi - radius, lo))
            .curve_to((hi, lo + radius), (hi, lo))
            .line_to((hi, hi - radius))
            .curve_to((hi - radius, hi), (hi, hi))
            .line_to((lo + radius, hi))
            .curve_to((lo, hi - radius), (lo, hi))
            .line_to((lo, lo + radius))
            .curve_to((lo + radius, lo), (lo, lo))
    }

    /// The outline as a white clip path in device pixels at `scale`.
    pub(super) fn clip_path(&self, scale: f32) -> Path<ScaledPixels> {
        let mut path = self.path.clone();
        path.color = white().into();
        path.content_mask = Bounds::new(
            point(px(0.0), px(0.0)),
            size(px(LOGICAL_SIZE), px(LOGICAL_SIZE)),
        )
        .into();
        path.scale(scale)
    }

    /// Coverage of device pixel `(x, y)` at `scale`: `Some(1.0)` or
    /// `Some(0.0)` when its center is more than [`EDGE`] device pixels from
    /// the outline, `None` when antialiased.
    pub(super) fn coverage(&self, scale: f32, x: i32, y: i32) -> Option<f32> {
        let p = ((x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale);
        let mut inside = false;
        let mut distance = f32::INFINITY;
        for (index, a) in self.polygon.iter().enumerate() {
            let b = self.polygon[(index + 1) % self.polygon.len()];
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0) {
                inside = !inside;
            }
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let length = dx * dx + dy * dy;
            let t = if length == 0.0 {
                0.0
            } else {
                (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / length).clamp(0.0, 1.0)
            };
            distance = distance.min((p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy));
        }
        (distance * scale > EDGE).then_some(if inside { 1.0 } else { 0.0 })
    }
}
