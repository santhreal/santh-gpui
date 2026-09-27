//! Creation time of headless wgpu renderers of one device: the first
//! renderer, a second renderer of the same target configuration, and a
//! transparency change of the second renderer to opaque and back.
//!
//! Each run creates a device, creates its two renderers, changes the
//! transparency, and drops them all. The report prints the median and the
//! 90th percentile of each measurement over the runs.

use gpui::{DevicePixels, size};
use gpui_wgpu::{WgpuContext, WgpuRenderer};
use std::time::{Duration, Instant};

const RUNS: usize = 30;

fn percentile(samples: &mut [Duration], fraction: f64) -> Duration {
    samples.sort_unstable();
    samples[((samples.len() - 1) as f64 * fraction).round() as usize]
}

fn report(name: &str, samples: &mut [Duration]) {
    println!(
        "{name:<28} median {:>9.2?} p90 {:>9.2?} ({} runs)",
        percentile(samples, 0.5),
        percentile(samples, 0.9),
        samples.len(),
    );
}

fn main() {
    let instance = WgpuContext::surfaceless_instance();
    let extent = size(DevicePixels(1600), DevicePixels(1000));
    let mut first = Vec::with_capacity(RUNS);
    let mut second = Vec::with_capacity(RUNS);
    let mut transparency = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        let context =
            WgpuContext::new_surfaceless(instance.clone(), None).expect("surfaceless context");
        if run == 0 {
            println!("adapter: {:?}", context.adapter.get_info().name);
        }

        let start = Instant::now();
        let first_renderer = WgpuRenderer::new_offscreen(&context, extent).expect("first renderer");
        first.push(start.elapsed());

        let start = Instant::now();
        let mut second_renderer =
            WgpuRenderer::new_offscreen(&context, extent).expect("second renderer");
        second.push(start.elapsed());

        let start = Instant::now();
        second_renderer.update_transparency(false);
        second_renderer.update_transparency(true);
        transparency.push(start.elapsed());

        drop(second_renderer);
        drop(first_renderer);
    }
    report("first renderer", &mut first);
    report("second renderer", &mut second);
    report("opaque and back", &mut transparency);
}
