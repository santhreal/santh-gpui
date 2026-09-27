//! Compares every constant buffer and structured buffer a shader entry point
//! reads with the Rust struct the renderer uploads into it.
//!
//! The entry points are read from the HLSL sources, compiled through the
//! debug-build compiler path, and reflected with `D3DReflect`. The test fails
//! on an entry point with no [`ShaderModule`], on a buffer with no Rust
//! layout, and on a stride, size, or member offset that differs from the Rust
//! struct. Members of types whose fields gpui keeps private, such as
//! `Background` and `AtlasTile`, are compared by their own offset and by the
//! offset of the member after them, not field by field.

use super::shader_resources::{ShaderModule, ShaderTarget, build_shader_blob};
use super::{BatchParams, GlobalParams, PathRasterizationSprite, PathSprite};
use crate::direct_write::GlyphLayerTextureParams;
use gpui::{
    BackdropBlur, EdgeFadeMask, MonochromeSprite, PolychromeSprite, Quad, Shadow, SubpixelSprite,
    Underline,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::{offset_of, size_of},
};
use windows::{
    Win32::Graphics::{
        Direct3D::{D3D_SIT_CBUFFER, D3D_SIT_STRUCTURED, Fxc::D3DReflect},
        Direct3D11::{
            D3D11_SHADER_BUFFER_DESC, D3D11_SHADER_DESC, D3D11_SHADER_INPUT_BIND_DESC,
            D3D11_SHADER_TYPE_DESC, D3D11_SHADER_VARIABLE_DESC, ID3D11ShaderReflection,
            ID3D11ShaderReflectionConstantBuffer, ID3D11ShaderReflectionType,
        },
    },
    core::Interface,
};

/// HLSL sources compiled into [`ShaderModule`] entry points.
const SHADER_SOURCES: [&str; 2] = [
    include_str!("../shaders.hlsl"),
    include_str!("../color_text_raster.hlsl"),
];

/// Byte layout of a Rust struct uploaded into a shader buffer: its size and
/// the offset of each member, keyed by the member's HLSL path.
struct Layout {
    size: usize,
    members: Vec<(&'static str, usize)>,
}

macro_rules! layout {
    ($ty:ty { $($hlsl:literal: $($field:ident).+),+ $(,)? }) => {
        Layout {
            size: size_of::<$ty>(),
            members: vec![$(($hlsl, offset_of!($ty, $($field).+))),+],
        }
    };
}

/// Byte layout of a reflected shader buffer.
struct Reflected {
    size: usize,
    /// Offset of every member and nested member, keyed by its dotted path.
    members: BTreeMap<String, usize>,
    /// Names of the members at the top level of the buffer.
    top_level: Vec<String>,
}

/// The Rust layout the renderer uploads into buffer `name` when it draws with
/// `module`, or `None` when it uploads nothing under that name.
fn rust_layout(module: ShaderModule, name: &str) -> Option<Layout> {
    match name {
        "GlobalParams" => {
            return Some(layout!(GlobalParams {
                "gamma_ratios": gamma_ratios,
                "global_viewport_size": viewport_size,
                "grayscale_enhanced_contrast": grayscale_enhanced_contrast,
                "subpixel_enhanced_contrast": subpixel_enhanced_contrast,
                "is_bgr": is_bgr,
                "global_pad": _pad,
            }));
        }
        "BatchParams" => {
            return Some(layout!(BatchParams {
                "batch_start_index": start_index,
                "batch_pad": _padding,
            }));
        }
        _ => {}
    }
    let (buffer, layout) = match module {
        ShaderModule::Quad => (
            "quads",
            layout!(Quad {
                "order": order,
                "border_style": border_style,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "background": background,
                "border_color": border_color,
                "corner_radii": corner_radii,
                "border_widths": border_widths,
                "transformation": transformation,
            }),
        ),
        ShaderModule::Shadow => (
            "shadows",
            layout!(Shadow {
                "order": order,
                "blur_radius": blur_radius,
                "bounds": bounds,
                "corner_radii": corner_radii,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "color": color,
                "element_bounds": element_bounds,
                "element_corner_radii": element_corner_radii,
                "inset": inset,
                "pad": pad,
                "transformation": transformation,
            }),
        ),
        ShaderModule::Underline => (
            "underlines",
            layout!(Underline {
                "order": order,
                "pad": pad,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "color": color,
                "thickness": thickness,
                "wavy": wavy,
                "transformation": transformation,
            }),
        ),
        ShaderModule::PathRasterization => (
            "path_rasterization_sprites",
            layout!(PathRasterizationSprite {
                "xy_position": xy_position,
                "st_position": st_position,
                "color": color,
                "bounds": bounds,
                "transformation": transformation,
            }),
        ),
        ShaderModule::PathSprite | ShaderModule::PathClipComposite => {
            ("path_sprites", layout!(PathSprite { "bounds": bounds }))
        }
        ShaderModule::MonochromeSprite => (
            "mono_sprites",
            layout!(MonochromeSprite {
                "order": order,
                "pad": pad,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "color": color,
                "tile": tile,
                "transformation": transformation,
            }),
        ),
        // Subpixel sprites are uploaded into the monochrome sprite layout.
        ShaderModule::SubpixelSprite => (
            "mono_sprites",
            layout!(SubpixelSprite {
                "order": order,
                "pad": pad,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "color": color,
                "tile": tile,
                "transformation": transformation,
            }),
        ),
        ShaderModule::PolychromeSprite => (
            "poly_sprites",
            layout!(PolychromeSprite {
                "order": order,
                "pad": pad,
                "grayscale": grayscale,
                "opacity": opacity,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "corner_radii": corner_radii,
                "tile": tile,
                "transformation": transformation,
            }),
        ),
        ShaderModule::EmojiRasterization => (
            "GlyphLayerTextureParams",
            layout!(GlyphLayerTextureParams {
                "bounds": bounds,
                "run_color": run_color,
                "gamma_ratios": gamma_ratios,
                "grayscale_enhanced_contrast": grayscale_enhanced_contrast,
                "_pad": _pad,
            }),
        ),
        ShaderModule::BackdropBlur => (
            "backdrop_blurs",
            layout!(BackdropBlur {
                "order": order,
                "pad": pad,
                "bounds": bounds,
                "content_mask": content_mask,
                "content_mask.bounds": content_mask.bounds,
                "content_mask.corner_radii": content_mask.corner_radii,
                "corner_radii": corner_radii,
                "blur_radius": blur_radius,
                "saturation": saturation,
                "tint": tint,
                "transformation": transformation,
            }),
        ),
        ShaderModule::EdgeFadeComposite => (
            "edge_fade_masks",
            layout!(EdgeFadeMask {
                "bounds": bounds,
                "fade_bounds": fade_bounds,
                "bands": bands,
                "bands.top": bands.top,
                "bands.right": bands.right,
                "bands.bottom": bands.bottom,
                "bands.left": bands.left,
            }),
        ),
    };
    (buffer == name).then_some(layout)
}

/// The module whose entry points are `{name}_vertex` and `{name}_fragment`.
fn module_named(name: &str) -> Option<ShaderModule> {
    Some(match name {
        "quad" => ShaderModule::Quad,
        "shadow" => ShaderModule::Shadow,
        "underline" => ShaderModule::Underline,
        "path_rasterization" => ShaderModule::PathRasterization,
        "path_sprite" => ShaderModule::PathSprite,
        "monochrome_sprite" => ShaderModule::MonochromeSprite,
        "subpixel_sprite" => ShaderModule::SubpixelSprite,
        "polychrome_sprite" => ShaderModule::PolychromeSprite,
        "emoji_rasterization" => ShaderModule::EmojiRasterization,
        "backdrop_blur" => ShaderModule::BackdropBlur,
        "path_clip_composite" => ShaderModule::PathClipComposite,
        "edge_fade_composite" => ShaderModule::EdgeFadeComposite,
        _ => return None,
    })
}

/// Names `x` of every `x_vertex(` and `x_fragment(` in the HLSL sources.
fn entry_point_names() -> BTreeSet<&'static str> {
    let mut names = BTreeSet::new();
    for source in SHADER_SOURCES {
        for suffix in ["_vertex(", "_fragment("] {
            for (end, _) in source.match_indices(suffix) {
                let start = source[..end]
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |index| index + 1);
                names.insert(&source[start..end]);
            }
        }
    }
    names
}

fn reflect(module: ShaderModule, target: ShaderTarget) -> ID3D11ShaderReflection {
    let blob = build_shader_blob(module, target)
        .unwrap_or_else(|error| panic!("compiling {module:?} {target:?}: {error:#}"));
    let mut reflection = std::ptr::null_mut();
    unsafe {
        D3DReflect(
            blob.GetBufferPointer(),
            blob.GetBufferSize(),
            &ID3D11ShaderReflection::IID,
            &mut reflection,
        )
        .unwrap_or_else(|error| panic!("reflecting {module:?} {target:?}: {error}"));
        ID3D11ShaderReflection::from_raw(reflection)
    }
}

/// Adds every member of `ty` at `base` bytes to `members`, under `prefix`.
fn flatten(
    ty: &ID3D11ShaderReflectionType,
    prefix: &str,
    base: usize,
    members: &mut BTreeMap<String, usize>,
) {
    let mut desc = D3D11_SHADER_TYPE_DESC::default();
    unsafe { ty.GetDesc(&mut desc) }.expect("reflecting a member type");
    for index in 0..desc.Members {
        let member = unsafe { ty.GetMemberTypeByIndex(index) }.expect("reflected member type");
        let name = unsafe { ty.GetMemberTypeName(index).to_string() }.expect("member name");
        let mut member_desc = D3D11_SHADER_TYPE_DESC::default();
        unsafe { member.GetDesc(&mut member_desc) }.expect("reflecting a member");
        let path = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}.{name}")
        };
        let offset = base + member_desc.Offset as usize;
        flatten(&member, &path, offset, members);
        members.insert(path, offset);
    }
}

fn reflect_buffer(buffer: &ID3D11ShaderReflectionConstantBuffer, structured: bool) -> Reflected {
    let mut desc = D3D11_SHADER_BUFFER_DESC::default();
    unsafe { buffer.GetDesc(&mut desc) }.expect("reflecting a buffer");
    let mut members = BTreeMap::new();
    let mut top_level = Vec::new();
    for index in 0..desc.Variables {
        let variable = unsafe { buffer.GetVariableByIndex(index) }.expect("reflected variable");
        let mut variable_desc = D3D11_SHADER_VARIABLE_DESC::default();
        unsafe { variable.GetDesc(&mut variable_desc) }.expect("reflecting a variable");
        let ty = unsafe { variable.GetType() }.expect("reflected variable type");
        if structured {
            // A structured buffer holds one `$Element` variable whose type is
            // the element struct.
            flatten(&ty, "", 0, &mut members);
            let mut ty_desc = D3D11_SHADER_TYPE_DESC::default();
            unsafe { ty.GetDesc(&mut ty_desc) }.expect("reflecting the element type");
            top_level.extend((0..ty_desc.Members).map(|index| {
                unsafe { ty.GetMemberTypeName(index).to_string() }.expect("member name")
            }));
        } else {
            let name = unsafe { variable_desc.Name.to_string() }.expect("variable name");
            let offset = variable_desc.StartOffset as usize;
            flatten(&ty, &name, offset, &mut members);
            members.insert(name.clone(), offset);
            top_level.push(name);
        }
    }
    Reflected {
        size: desc.Size as usize,
        members,
        top_level,
    }
}

/// Lists every difference between the reflected buffer and the Rust layout.
fn compare(context: &str, reflected: &Reflected, rust: &Layout, errors: &mut Vec<String>) {
    if reflected.size != rust.size {
        errors.push(format!(
            "{context}: HLSL size {} bytes, Rust size {} bytes",
            reflected.size, rust.size
        ));
    }
    for (path, offset) in &rust.members {
        match reflected.members.get(*path) {
            Some(reflected_offset) if reflected_offset == offset => {}
            Some(reflected_offset) => errors.push(format!(
                "{context}: `{path}` at HLSL offset {reflected_offset}, Rust offset {offset}"
            )),
            None => errors.push(format!("{context}: `{path}` missing from the HLSL struct")),
        }
    }
    for name in &reflected.top_level {
        if !rust.members.iter().any(|(path, _)| *path == name.as_str()) {
            errors.push(format!("{context}: HLSL member `{name}` has no Rust field"));
        }
    }
}

/// Catches a Rust primitive and its HLSL struct drifting apart: a field added,
/// removed, resized, or reordered on one side only, as when `ContentMask`
/// grew from 16 to 32 bytes. Draws read such a buffer at the wrong stride and
/// render garbage with no error. Does not catch a member of the same size and
/// offset with a different meaning, such as swapped `Corners` fields.
#[test]
fn every_shader_buffer_matches_its_rust_layout() {
    let mut errors = Vec::new();
    for name in entry_point_names() {
        let Some(module) = module_named(name) else {
            errors.push(format!("HLSL entry point `{name}` has no ShaderModule"));
            continue;
        };
        assert_eq!(module.as_str(), name, "ShaderModule::as_str for {module:?}");
        let mut checked = 0;
        for target in [ShaderTarget::Vertex, ShaderTarget::Fragment] {
            let reflection = reflect(module, target);
            let mut shader_desc = D3D11_SHADER_DESC::default();
            unsafe { reflection.GetDesc(&mut shader_desc) }.expect("reflecting a shader");
            for index in 0..shader_desc.BoundResources {
                let mut bind = D3D11_SHADER_INPUT_BIND_DESC::default();
                unsafe { reflection.GetResourceBindingDesc(index, &mut bind) }
                    .expect("reflecting a resource binding");
                let structured = bind.Type == D3D_SIT_STRUCTURED;
                if !structured && bind.Type != D3D_SIT_CBUFFER {
                    continue;
                }
                let buffer_name = unsafe { bind.Name.to_string() }.expect("buffer name");
                let context = format!("{module:?} {target:?} `{buffer_name}`");
                let Some(rust) = rust_layout(module, &buffer_name) else {
                    errors.push(format!("{context}: no Rust layout"));
                    continue;
                };
                let buffer = unsafe { reflection.GetConstantBufferByName(bind.Name) }
                    .expect("reflected buffer");
                let reflected = reflect_buffer(&buffer, structured);
                if structured && bind.NumSamples as usize != rust.size {
                    errors.push(format!(
                        "{context}: HLSL stride {} bytes, Rust size {} bytes",
                        bind.NumSamples, rust.size
                    ));
                }
                compare(&context, &reflected, &rust, &mut errors);
                checked += 1;
            }
        }
        if checked == 0 {
            errors.push(format!("{module:?} reads no buffer"));
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
