//! A tile's pixels are on its texture as soon as `get_or_insert_with`
//! returns, in the texture's byte order, at the tile's origin: the atlas
//! writes them through the queue and keeps no copy for a later frame to
//! flush. These tests read the texture back without drawing a frame, so an
//! upload deferred to the renderer, written at the wrong origin or row
//! pitch, or left in BGRA order on an RGBA texture fails them. They do not
//! measure how many copies an upload makes.

use super::*;
use gpui::{ImageId, RenderImageParams, RenderSvgParams};
use std::sync::Arc;

fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let context =
        crate::WgpuContext::new_surfaceless(crate::WgpuContext::surfaceless_instance(), None)?;
    Ok((Arc::clone(&context.device), Arc::clone(&context.queue)))
}

fn image_key(image_id: usize) -> AtlasKey {
    AtlasKey::Image(RenderImageParams {
        image_id: ImageId(image_id),
        frame_index: 0,
    })
}

fn size(width: i32, height: i32) -> Size<DevicePixels> {
    Size {
        width: DevicePixels(width),
        height: DevicePixels(height),
    }
}

/// `width * height * bytes_per_pixel` bytes, no two alike within 251.
fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

fn insert(atlas: &WgpuAtlas, key: &AtlasKey, size: Size<DevicePixels>, bytes: &[u8]) -> AtlasTile {
    atlas
        .get_or_insert_with(key, &mut || Ok(Some((size, Cow::Owned(bytes.to_vec())))))
        .expect("allocation should succeed")
        .expect("callback returns Some")
}

/// The tile's texels as the GPU holds them, read with a copy submitted
/// after the insert and no frame drawn.
fn read_tile(atlas: &WgpuAtlas, tile: &AtlasTile) -> Vec<u8> {
    let lock = atlas.0.lock();
    let texture = &lock.storage[tile.texture_id];
    let bytes_per_pixel = texture.bytes_per_pixel() as u32;
    let width = tile.bounds.size.width.0 as u32;
    let height = tile.bounds.size.height.0 as u32;
    let row = width * bytes_per_pixel;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = lock.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas_tile_readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = lock
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture.texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: tile.bounds.origin.x.0 as u32,
                y: tile.bounds.origin.y.0 as u32,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    lock.queue.submit(std::iter::once(encoder.finish()));
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    lock.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback completes");
    receiver
        .recv()
        .expect("map callback runs")
        .expect("readback buffer maps");
    let mapped = slice.get_mapped_range();
    mapped
        .chunks(padded as usize)
        .flat_map(|line| &line[..row as usize])
        .copied()
        .collect()
}

fn bgra_to_rgba(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .flat_map(|p| [p[2], p[1], p[0], p[3]])
        .collect()
}

#[test]
fn inserted_image_tiles_are_on_the_texture_before_any_frame() -> anyhow::Result<()> {
    for format in [
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ] {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, format);
        // Two tiles in one texture, of widths whose rows are not a multiple
        // of the copy alignment: the second sits away from the origin.
        let (a_size, b_size) = (size(3, 2), size(5, 3));
        let a_bytes = pattern(1, 3 * 2 * 4);
        let b_bytes = pattern(101, 5 * 3 * 4);
        let a = insert(&atlas, &image_key(1), a_size, &a_bytes);
        let b = insert(&atlas, &image_key(2), b_size, &b_bytes);
        assert_eq!(a.texture_id, b.texture_id, "{format:?}");
        assert_ne!(a.bounds.origin, b.bounds.origin, "{format:?}");

        let expected = |bytes: &[u8]| match format {
            wgpu::TextureFormat::Rgba8Unorm => bgra_to_rgba(bytes),
            _ => bytes.to_vec(),
        };
        assert_eq!(
            read_tile(&atlas, &a),
            expected(&a_bytes),
            "{format:?} tile a"
        );
        assert_eq!(
            read_tile(&atlas, &b),
            expected(&b_bytes),
            "{format:?} tile b"
        );
    }
    Ok(())
}

#[test]
fn inserted_monochrome_tiles_are_on_the_texture_one_byte_per_pixel() -> anyhow::Result<()> {
    let (device, queue) = test_device_and_queue()?;
    let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
    let key = AtlasKey::Svg(RenderSvgParams {
        path: "icon.svg".into(),
        size: size(7, 4),
    });
    let bytes = pattern(9, 7 * 4);
    let tile = insert(&atlas, &key, size(7, 4), &bytes);
    assert_eq!(read_tile(&atlas, &tile), bytes);
    Ok(())
}

#[test]
fn a_cached_tile_keeps_its_first_pixels() -> anyhow::Result<()> {
    let (device, queue) = test_device_and_queue()?;
    let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
    let first = pattern(3, 2 * 2 * 4);
    let tile = insert(&atlas, &image_key(1), size(2, 2), &first);
    let again = atlas
        .get_or_insert_with(&image_key(1), &mut || panic!("a cached key builds nothing"))?
        .expect("cached tile");
    assert_eq!(again, tile);
    assert_eq!(read_tile(&atlas, &tile), first);
    Ok(())
}

#[test]
fn a_tile_removed_right_after_its_insert_leaves_the_atlas_usable() -> anyhow::Result<()> {
    let (device, queue) = test_device_and_queue()?;
    let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
    insert(&atlas, &image_key(1), size(1, 1), &[0, 0, 0, 255]);
    // The only tile goes, and its texture with it, while its write is
    // still queued.
    atlas.remove(&image_key(1));
    let bytes = pattern(7, 4 * 4 * 4);
    let tile = insert(&atlas, &image_key(2), size(4, 4), &bytes);
    assert_eq!(read_tile(&atlas, &tile), bytes);
    Ok(())
}

#[test]
fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
    let (device, queue) = test_device_and_queue()?;
    let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

    let small = size(64, 64);
    let big = size(700, 700);
    let insert_zeroed = |key: &AtlasKey, size: Size<DevicePixels>| {
        let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
        insert(&atlas, key, size, &vec![0u8; byte_count])
    };

    let keeper_tile = insert_zeroed(&image_key(1), small);
    let tile_a = insert_zeroed(&image_key(2), big);
    assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

    atlas.remove(&image_key(2));
    let tile_b = insert_zeroed(&image_key(3), big);
    assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    Ok(())
}

#[test]
fn swizzle_upload_data_borrows_bgra_uploads() {
    let input = vec![0x10, 0x20, 0x30, 0x40];
    let out = swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm);
    assert!(matches!(out, Cow::Borrowed(_)));
    assert_eq!(&*out, &input[..]);
}

#[test]
fn swizzle_upload_data_converts_bgra_to_rgba() {
    let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
    assert_eq!(
        &*swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
        &[0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
    );
}
