//! CPU time of a window scene of 1600x1000 pixels: three panels, each a
//! layer, holding rows of text as glyph sprites over 256 atlas tiles, row
//! backgrounds and markers as bordered quads, icons, underlines, a chart of
//! 64-vertex paths, and a popover at z-index 1 with a shadow. The first line
//! of the report prints the primitive counts and the size of a `Primitive`.
//!
//! `paint` clears a scene and inserts every primitive: a frame that renders
//! every view. `replay` clears a scene and replays the whole paint log of the
//! previous finished frame into it: a frame that reuses every view. The two
//! scenes of `replay` alternate the way a window's rendered and next frames
//! do. Each case then finishes the scene, timed on its own.
//!
//! Each case runs 100 warm-up frames, then 2000 measured frames. The report
//! prints the median and the 90th percentile of recording, of finishing, and
//! of the two together, three times.

use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, ContentMask, Corners, DevicePixels, Edges,
    Hsla, MonochromeSprite, Path, Point, PolychromeSprite, Quad, ScaledPixels, Scene, Shadow,
    TileId, TransformationMatrix, Underline, point, px, size,
};
use std::time::{Duration, Instant};

const WIDTH: f32 = 1600.0;
const HEIGHT: f32 = 1000.0;
const ROW: f32 = 20.0;
const GLYPH: f32 = 9.0;
const TILES: u32 = 256;
const WARM_UP: usize = 100;
const FRAMES: usize = 2000;

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn clip(bounds: Bounds<ScaledPixels>) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds,
        corner_radii: Corners::default(),
    }
}

fn quad(bounds: Bounds<ScaledPixels>, mask: ContentMask<ScaledPixels>, hue: f32) -> Quad {
    Quad {
        order: 0,
        border_style: Default::default(),
        bounds,
        content_mask: mask,
        background: gpui::solid_background(Hsla {
            h: hue,
            s: 0.4,
            l: 0.2,
            a: 1.0,
        }),
        border_color: Hsla::white(),
        corner_radii: Corners::all(ScaledPixels(3.0)),
        border_widths: Edges::all(ScaledPixels(1.0)),
        transformation: TransformationMatrix::unit(),
    }
}

fn tile(kind: AtlasTextureKind, id: u32) -> AtlasTile {
    AtlasTile {
        texture_id: AtlasTextureId { index: 0, kind },
        tile_id: TileId(id),
        padding: 0,
        bounds: Bounds::new(
            point(DevicePixels(0), DevicePixels(0)),
            size(DevicePixels(9), DevicePixels(16)),
        ),
    }
}

/// One row of `glyphs` glyph sprites starting at `(x, y)`.
fn text(scene: &mut Scene, x: f32, y: f32, glyphs: u32, mask: ContentMask<ScaledPixels>) {
    for glyph in 0..glyphs {
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: rect(x + glyph as f32 * GLYPH, y + 2.0, GLYPH, 16.0),
            content_mask: mask,
            color: Hsla::white(),
            tile: tile(
                AtlasTextureKind::Monochrome,
                (glyph * 7 + (y as u32)) % TILES,
            ),
            transformation: TransformationMatrix::unit(),
        });
    }
}

/// A panel of `rows` rows of `glyphs` glyphs, drawn inside a layer.
fn panel(scene: &mut Scene, x: f32, width: f32, rows: u32, glyphs: u32, icons: bool) {
    let bounds = rect(x, 0.0, width, HEIGHT);
    let mask = clip(bounds);
    scene.push_layer(bounds);
    scene.insert_primitive(quad(bounds, mask, x / WIDTH));
    for row in 0..rows {
        let y = row as f32 * ROW;
        scene.insert_primitive(quad(rect(x, y, width, ROW), mask, row as f32 / 64.0));
        scene.insert_primitive(quad(rect(x + 2.0, y + 4.0, 4.0, 12.0), mask, 0.3));
        let mut text_x = x + 8.0;
        if icons {
            scene.insert_primitive(PolychromeSprite {
                order: 0,
                pad: 0,
                grayscale: false.into(),
                opacity: 1.0,
                bounds: rect(text_x, y + 2.0, 16.0, 16.0),
                content_mask: mask,
                corner_radii: Corners::default(),
                tile: tile(AtlasTextureKind::Polychrome, row % 12),
                transformation: TransformationMatrix::unit(),
            });
            text_x += 20.0;
        }
        text(scene, text_x, y, glyphs, mask);
        if row % 5 == 0 {
            scene.insert_primitive(Underline {
                order: 0,
                pad: 0,
                bounds: rect(text_x, y + 17.0, glyphs as f32 * GLYPH, 1.0),
                content_mask: mask,
                color: Hsla::red(),
                thickness: ScaledPixels(1.0),
                wavy: (row % 10 == 0).into(),
                transformation: TransformationMatrix::unit(),
            });
        }
    }
    scene.pop_layer();
}

/// A chart of `count` paths of 64 vertices in the right panel.
fn chart(scene: &mut Scene, count: u32) {
    for index in 0..count {
        let y = 600.0 + (index % 10) as f32 * 30.0;
        let mut path = Path::new(Point::new(px(1210.0), px(y)));
        for step in 1..64 {
            let x = 1210.0 + step as f32 * 6.0;
            let wave = ((step + index) % 8) as f32 * 3.0;
            path.line_to(Point::new(px(x), px(y + wave)));
        }
        path.color = gpui::solid_background(Hsla::blue().opacity(0.5));
        path.content_mask = ContentMask {
            bounds: Bounds::new(point(px(1200.0), px(0.0)), size(px(400.0), px(HEIGHT))),
            corner_radii: Corners::default(),
        };
        scene.insert_primitive(path.scale(1.0));
    }
}

fn popover(scene: &mut Scene) {
    let bounds = rect(500.0, 300.0, 400.0, 90.0);
    let mask = clip(rect(0.0, 0.0, WIDTH, HEIGHT));
    scene.push_z_index(1);
    scene.insert_primitive(Shadow {
        order: 0,
        blur_radius: ScaledPixels(12.0),
        bounds,
        corner_radii: Corners::all(ScaledPixels(6.0)),
        content_mask: mask,
        color: Hsla::black().opacity(0.4),
        element_bounds: bounds,
        element_corner_radii: Corners::all(ScaledPixels(6.0)),
        inset: 0,
        pad: 0,
        transformation: TransformationMatrix::unit(),
    });
    scene.insert_primitive(quad(bounds, mask, 0.6));
    for row in 0..3 {
        text(scene, 510.0, 305.0 + row as f32 * ROW, 40, mask);
    }
    scene.pop_z_index();
}

fn paint(scene: &mut Scene) {
    scene.insert_primitive(quad(
        rect(0.0, 0.0, WIDTH, HEIGHT),
        clip(rect(0.0, 0.0, WIDTH, HEIGHT)),
        0.0,
    ));
    panel(scene, 0.0, 400.0, 48, 24, true);
    panel(scene, 400.0, 800.0, 50, 80, false);
    panel(scene, 1200.0, 400.0, 28, 36, false);
    chart(scene, 30);
    popover(scene);
}

fn percentile(samples: &mut [Duration], fraction: f64) -> Duration {
    samples.sort_unstable();
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

#[derive(Default)]
struct Phases {
    record: Vec<Duration>,
    finish: Vec<Duration>,
    frame: Vec<Duration>,
}

impl Phases {
    fn push(&mut self, frame: usize, start: Instant, recorded: Instant, finished: Instant) {
        if frame >= WARM_UP {
            self.record.push(recorded - start);
            self.finish.push(finished - recorded);
            self.frame.push(finished - start);
        }
    }

    fn report(mut self, name: &str) {
        let micros = |duration: Duration| duration.as_secs_f64() * 1e6;
        let column = |samples: &mut Vec<Duration>| {
            format!(
                "median {:>7.1}us p90 {:>7.1}us",
                micros(percentile(samples, 0.5)),
                micros(percentile(samples, 0.9))
            )
        };
        println!(
            "{name:<7} record {} | finish {} | frame {}",
            column(&mut self.record),
            column(&mut self.finish),
            column(&mut self.frame),
        );
    }
}

fn measure_paint() -> Phases {
    let mut scene = Scene::default();
    let mut phases = Phases::default();
    for frame in 0..WARM_UP + FRAMES {
        let start = Instant::now();
        scene.clear();
        paint(&mut scene);
        let recorded = Instant::now();
        scene.finish();
        phases.push(frame, start, recorded, Instant::now());
    }
    phases
}

fn measure_replay() -> Phases {
    let mut previous = Scene::default();
    paint(&mut previous);
    previous.finish();
    let mut next = Scene::default();
    let mut phases = Phases::default();
    for frame in 0..WARM_UP + FRAMES {
        let start = Instant::now();
        next.clear();
        next.replay(0..previous.len(), &previous);
        let recorded = Instant::now();
        next.finish();
        phases.push(frame, start, recorded, Instant::now());
        assert_eq!(
            next.len(),
            previous.len(),
            "frame {frame} replayed a partial log"
        );
        std::mem::swap(&mut previous, &mut next);
    }
    phases
}

fn main() {
    let mut scene = Scene::default();
    paint(&mut scene);
    scene.finish();
    println!(
        "quads {} monochrome sprites {} polychrome sprites {} underlines {} paths {} shadows {} operations {} primitive size {} bytes",
        scene.quads.len(),
        scene.monochrome_sprites.len(),
        scene.polychrome_sprites.len(),
        scene.underlines.len(),
        scene.paths.len(),
        scene.shadows.len(),
        scene.len(),
        std::mem::size_of::<gpui::Primitive>(),
    );
    for _ in 0..3 {
        measure_paint().report("paint");
        measure_replay().report("replay");
    }
}
