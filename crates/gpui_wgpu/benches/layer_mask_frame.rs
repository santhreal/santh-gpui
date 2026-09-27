//! Frame time of the headless wgpu renderer on a 1600x1000 scene of 2000
//! bordered, rounded quads and 40 paths, drawn whole every frame: without a
//! layer mask, inside an edge fade of 48 pixels at the top and bottom edges,
//! and inside a path clip of the window's rectangle.
//!
//! Each case draws 50 warm-up frames, then 500 measured frames. A frame's CPU
//! time is `WgpuRenderer::draw` (encoding and submission); its total time
//! also waits for the device to finish the submission. The report prints the
//! median and the 90th percentile of both.

use gpui::{
    Bounds, ContentMask, Corners, DevicePixels, EdgeFadeMask, Edges, Hsla, LayerMask, Path, Point,
    Quad, ScaledPixels, Scene, TransformationMatrix, point, px, size,
};
use gpui_wgpu::{WgpuContext, WgpuRenderer};
use std::time::{Duration, Instant};

const WIDTH: f32 = 1600.0;
const HEIGHT: f32 = 1000.0;
const WARM_UP: usize = 50;
const FRAMES: usize = 500;

fn window() -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size(ScaledPixels(WIDTH), ScaledPixels(HEIGHT)),
    )
}

fn content(scene: &mut Scene) {
    let mask = ContentMask {
        bounds: window(),
        corner_radii: Corners::default(),
    };
    for row in 0..50 {
        for column in 0..40 {
            scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: Bounds::new(
                    point(
                        ScaledPixels(column as f32 * 40.0 + 2.0),
                        ScaledPixels(row as f32 * 20.0 + 2.0),
                    ),
                    size(ScaledPixels(36.0), ScaledPixels(16.0)),
                ),
                content_mask: mask,
                background: gpui::solid_background(Hsla {
                    h: (row * 40 + column) as f32 / 2000.0,
                    s: 0.6,
                    l: 0.5,
                    a: 0.9,
                }),
                border_color: Hsla::white(),
                corner_radii: Corners::all(ScaledPixels(4.0)),
                border_widths: Edges::all(ScaledPixels(1.0)),
                transformation: TransformationMatrix::unit(),
            });
        }
    }
    for index in 0..40 {
        let x = (index % 10) as f32 * 160.0 + 20.0;
        let y = (index / 10) as f32 * 250.0 + 20.0;
        let mut path = Path::new(Point::new(px(x), px(y)));
        path.line_to(Point::new(px(x + 120.0), px(y)));
        path.curve_to(
            Point::new(px(x), px(y + 200.0)),
            Point::new(px(x + 120.0), px(y + 200.0)),
        );
        path.line_to(Point::new(px(x), px(y)));
        path.color = gpui::solid_background(Hsla::white().opacity(0.5));
        path.content_mask = ContentMask {
            bounds: Bounds::new(point(px(0.0), px(0.0)), size(px(WIDTH), px(HEIGHT))),
            corner_radii: Corners::default(),
        };
        scene.insert_primitive(path.scale(1.0));
    }
}

struct Timing {
    cpu: Vec<Duration>,
    total: Vec<Duration>,
}

fn measure(context: &WgpuContext, scene: &Scene) -> Timing {
    let mut renderer = WgpuRenderer::new_offscreen(
        context,
        size(DevicePixels(WIDTH as i32), DevicePixels(HEIGHT as i32)),
    )
    .expect("offscreen renderer");
    let mut timing = Timing {
        cpu: Vec::with_capacity(FRAMES),
        total: Vec::with_capacity(FRAMES),
    };
    for frame in 0..WARM_UP + FRAMES {
        let start = Instant::now();
        assert!(renderer.draw(scene), "frame {frame} failed");
        let encoded = start.elapsed();
        context
            .device
            .poll(gpui_wgpu::wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .expect("device completes the frame");
        let finished = start.elapsed();
        if frame >= WARM_UP {
            timing.cpu.push(encoded);
            timing.total.push(finished);
        }
    }
    timing
}

fn percentile(samples: &mut [Duration], fraction: f64) -> Duration {
    samples.sort_unstable();
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn report(name: &str, mut timing: Timing) {
    println!(
        "{name:<24} cpu median {:>8.1?} p90 {:>8.1?} | total median {:>8.1?} p90 {:>8.1?}",
        percentile(&mut timing.cpu, 0.5),
        percentile(&mut timing.cpu, 0.9),
        percentile(&mut timing.total, 0.5),
        percentile(&mut timing.total, 0.9),
    );
}

fn main() {
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    println!("adapter: {:?}", context.adapter.get_info().name);

    let mut plain = Scene::default();
    content(&mut plain);
    plain.finish();

    let mut faded = Scene::default();
    faded.push_layer_mask(LayerMask::EdgeFade(EdgeFadeMask::new(
        window(),
        Edges {
            top: ScaledPixels(48.0),
            right: ScaledPixels(0.0),
            bottom: ScaledPixels(48.0),
            left: ScaledPixels(0.0),
        },
        window(),
    )));
    content(&mut faded);
    faded.pop_layer_mask();
    faded.finish();

    let mut clipped = Scene::default();
    let mut clip = Path::new(Point::new(px(0.0), px(0.0)));
    clip.line_to(Point::new(px(WIDTH), px(0.0)));
    clip.line_to(Point::new(px(WIDTH), px(HEIGHT)));
    clip.line_to(Point::new(px(0.0), px(HEIGHT)));
    clip.line_to(Point::new(px(0.0), px(0.0)));
    clip.color = gpui::solid_background(Hsla::white());
    clip.content_mask = ContentMask {
        bounds: Bounds::new(point(px(0.0), px(0.0)), size(px(WIDTH), px(HEIGHT))),
        corner_radii: Corners::default(),
    };
    clipped.push_layer_mask(LayerMask::Path(clip.scale(1.0)));
    content(&mut clipped);
    clipped.pop_layer_mask();
    clipped.finish();

    for _ in 0..3 {
        report("no layer mask", measure(&context, &plain));
        report("edge fade", measure(&context, &faded));
        report("path clip", measure(&context, &clipped));
    }
}
