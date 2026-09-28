//! CPU reference math for the draw tests: the documented backdrop blur,
//! layer mask composite, and edge fade, the rounded rectangle signed
//! distance, and polygon coverage of path outlines.

use super::{EDGE, SIZE, window_mask};
use gpui::{
    BackdropBlur, Bounds, ContentMask, Corners, EdgeFadeMask, Edges, Hsla, LayerMask, Path, Pixels,
    ScaledPixels, point, px, white,
};

pub(super) type Rgba = [f32; 4];

/// A drawn frame with unorm RGBA channels, row by row.
pub(super) struct Image(pub(super) Vec<Rgba>);

impl Image {
    /// The pixel at `(x, y)`, clamped to the frame edge.
    pub(super) fn at(&self, x: i32, y: i32) -> Rgba {
        self.0[(y.clamp(0, SIZE - 1) * SIZE + x.clamp(0, SIZE - 1)) as usize]
    }

    /// Bilinear sample at window position `p`, clamped to the frame edge.
    pub(super) fn sample(&self, p: (f32, f32)) -> Rgba {
        let (x, y) = (p.0 - 0.5, p.1 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let row = |y| lerp(self.at(x0, y), self.at(x0 + 1, y), tx);
        lerp(row(y0), row(y0 + 1), ty)
    }
}

pub(super) fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
    [0, 1, 2, 3].map(|c| a[c] + (b[c] - a[c]) * t)
}

/// The rounded rectangle signed distance of `quad_sdf` in the shaders.
pub(super) fn quad_sdf(
    p: (f32, f32),
    bounds: &Bounds<ScaledPixels>,
    radii: &Corners<ScaledPixels>,
) -> f32 {
    let half = (bounds.size.width.0 / 2.0, bounds.size.height.0 / 2.0);
    let to_point = (
        p.0 - bounds.origin.x.0 - half.0,
        p.1 - bounds.origin.y.0 - half.1,
    );
    let radius = match (to_point.0 < 0.0, to_point.1 < 0.0) {
        (true, true) => radii.top_left,
        (true, false) => radii.bottom_left,
        (false, true) => radii.top_right,
        (false, false) => radii.bottom_right,
    }
    .0;
    let corner = (
        to_point.0.abs() - half.0 + radius,
        to_point.1.abs() - half.1 + radius,
    );
    if radius == 0.0 {
        corner.0.max(corner.1)
    } else {
        corner.0.max(0.0).hypot(corner.1.max(0.0)) + corner.0.max(corner.1).min(0.0) - radius
    }
}

pub(super) fn coverage(sdf: f32) -> f32 {
    (0.5 - sdf).clamp(0.0, 1.0)
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

/// The documented result of drawing `blur` over `frame` at pixel `(x, y)`.
pub(super) fn expected_blur(frame: &Image, blur: &BackdropBlur, x: i32, y: i32) -> Rgba {
    const RINGS: [(f32, f32, f32); 3] = [(0.38, 0.637, 0.0), (0.70, 0.216, 0.5), (1.0, 0.044, 0.0)];
    let p = (x as f32 + 0.5, y as f32 + 0.5);
    let covered = coverage(quad_sdf(p, &blur.bounds, &blur.corner_radii))
        * coverage(quad_sdf(
            p,
            &blur.content_mask.bounds,
            &blur.content_mask.corner_radii,
        ));
    let r = blur.blur_radius.0;
    let mut color = frame.sample(p);
    if r > 0.0 {
        let mut weight = 1.0;
        for (k, w, phase) in RINGS {
            for i in 0..8 {
                let theta = (i as f32 + phase) * std::f32::consts::FRAC_PI_4;
                let tap = frame.sample((p.0 + r * k * theta.cos(), p.1 + r * k * theta.sin()));
                color = [0, 1, 2, 3].map(|c| color[c] + tap[c] * w);
                weight += w;
            }
        }
        color = color.map(|c| c / weight);
    }
    let luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
    let tint = rgba(blur.tint);
    let [r, g, b] = [0, 1, 2].map(|c| {
        let saturated = (luminance + blur.saturation * (color[c] - luminance)).clamp(0.0, 1.0);
        saturated + (tint[c] - saturated) * tint[3]
    });
    over(frame.at(x, y), [r, g, b, color[3] * covered])
}

/// A closed outline drawn as a gpui path, with the polygon it encloses.
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

    pub(super) fn clip_path(&self) -> Path<ScaledPixels> {
        self.path(white(), window_mask())
    }

    /// The outline as a path filled with `color` and clipped by
    /// `content_mask`.
    pub(super) fn path(
        &self,
        color: Hsla,
        content_mask: ContentMask<ScaledPixels>,
    ) -> Path<ScaledPixels> {
        let mut path = self.path.clone();
        path.color = color.into();
        let mut path = path.scale(1.0);
        path.content_mask = content_mask;
        path
    }

    /// Coverage of pixel `(x, y)`: `Some(1.0)` or `Some(0.0)` when its center
    /// is more than [`EDGE`] from the outline, `None` when antialiased.
    pub(super) fn coverage(&self, x: i32, y: i32) -> Option<f32> {
        let p = (x as f32 + 0.5, y as f32 + 0.5);
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
        (distance > EDGE).then_some(if inside { 1.0 } else { 0.0 })
    }
}

/// The documented `EndLayerMask` composite of `layer` onto `dst` at mask
/// value `m`.
pub(super) fn composite(dst: Rgba, layer: Rgba, m: f32) -> Rgba {
    let [r, g, b] = [0, 1, 2].map(|c| layer[c] * m + dst[c] * (1.0 - layer[3] * m));
    [r, g, b, (layer[3] * m + dst[3]).min(1.0)]
}

/// An edge fade in logical pixels, drawn at `scale` device pixels per
/// logical pixel.
pub(super) struct Fade {
    /// The faded region.
    pub(super) bounds: Bounds<Pixels>,
    /// The width of the ramp inside each edge of `bounds`.
    pub(super) bands: Edges<Pixels>,
    /// The region the layer is composited into.
    pub(super) clip: Bounds<Pixels>,
    pub(super) scale: f32,
}

impl Fade {
    /// The layer mask `Window::with_edge_fade` records for this fade.
    pub(super) fn mask(&self) -> LayerMask {
        LayerMask::EdgeFade(EdgeFadeMask::new(
            self.bounds.scale(self.scale),
            self.bands.scale(self.scale),
            self.clip.scale(self.scale),
        ))
    }

    /// The documented mask value `m = r * r` at the center of device pixel
    /// `(x, y)`, zero outside `clip`. Computed in logical pixels, where the
    /// ratio of a distance to a band is the same as in device pixels.
    pub(super) fn m(&self, x: i32, y: i32) -> f32 {
        let p = ((x as f32 + 0.5) / self.scale, (y as f32 + 0.5) / self.scale);
        let [left, top, right, bottom] = sides(&self.clip);
        if p.0 < left || p.0 >= right || p.1 < top || p.1 >= bottom {
            return 0.0;
        }
        let [left, top, right, bottom] = sides(&self.bounds);
        let r = [
            (p.1 - top, self.bands.top),
            (right - p.0, self.bands.right),
            (bottom - p.1, self.bands.bottom),
            (p.0 - left, self.bands.left),
        ]
        .map(|(d, band)| (d, f32::from(band)))
        .into_iter()
        .filter(|(_, band)| *band > 0.0)
        .map(|(d, band)| (d / band).clamp(0.0, 1.0))
        .fold(1.0, f32::min);
        r * r
    }
}

/// The left, top, right, and bottom sides of `bounds`.
fn sides(bounds: &Bounds<Pixels>) -> [f32; 4] {
    [bounds.left(), bounds.top(), bounds.right(), bounds.bottom()].map(f32::from)
}
