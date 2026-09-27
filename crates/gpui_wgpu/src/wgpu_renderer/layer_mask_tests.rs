//! Layer masks on the wgpu renderer, compared pixel by pixel against a CPU
//! model of the composite formula at 1x and 2x scale.
//!
//! The model draws every node of a test scene at each pixel center, stores
//! each target as 8-bit unorm after every draw, and composites a layer
//! through `dst.rgb = layer.rgb * m + dst.rgb * (1 - layer.a * m)`, with `m`
//! the edge fade `r * r` or the path coverage.
//!
//! Catches an edge fade composited through `r` instead of `r * r`, through
//! the maximum instead of the minimum across edges, or without one edge's
//! term; a batch inside a mask (paths, a backdrop blur, or anything drawn
//! after a nested mask closes) drawn onto the frame or an outer layer instead
//! of the innermost layer; an inner layer composited onto the frame instead
//! of its outer layer; a backdrop blur inside a mask that samples a layer
//! instead of the frame beneath every open layer; masked content that
//! reaches the frame outside the damage scissor; and layer textures
//! allocated by a frame without masks, released before
//! [`gpui::LAYER_IDLE_RELEASE_FRAMES`] frames without a use, or kept after
//! them. Does not catch a layer pass or a composite that alone omits the
//! damage scissor: layers are cleared whole and the other scissor keeps the
//! frame outside the damage unchanged, so only the extra fragment work
//! differs. Does not catch errors of path antialiasing at a path mask's
//! edges: every path here is axis aligned on whole device pixels, where
//! coverage is 0 or 1.

use super::WgpuRenderer;
use crate::WgpuContext;
use gpui::{
    BackdropBlur, Bounds, ContentMask, Corners, DevicePixels, EdgeFadeMask, Edges, Hsla,
    LAYER_IDLE_RELEASE_FRAMES, LayerMask, Path, Point, Quad, ScaledPixels, Scene,
    TransformationMatrix, px, rgb, size,
};

/// The logical width and height of every test frame.
const EXTENT: f32 = 100.0;

/// A premultiplied RGBA color with channels in `[0, 1]`.
type Color = [f32; 4];

const TRANSPARENT: Color = [0.0; 4];

/// An opaque color, as `0xRRGGBB`.
#[derive(Clone, Copy)]
struct Paint(u32);

const RED: Paint = Paint(0xff0000);
const GREEN: Paint = Paint(0x00ff00);
const BLUE: Paint = Paint(0x0000ff);

impl Paint {
    fn color(self) -> Color {
        let channel = |shift: u32| ((self.0 >> shift) & 0xff) as f32 / 255.0;
        [channel(16), channel(8), channel(0), 1.0]
    }

    fn hsla(self) -> Hsla {
        rgb(self.0).into()
    }
}

/// A rectangle in logical pixels.
#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
    Rect {
        x,
        y,
        width,
        height,
    }
}

const FULL: Rect = Rect {
    x: 0.0,
    y: 0.0,
    width: EXTENT,
    height: EXTENT,
};

impl Rect {
    fn device(self, scale: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            Point::new(ScaledPixels(self.x * scale), ScaledPixels(self.y * scale)),
            size(
                ScaledPixels(self.width * scale),
                ScaledPixels(self.height * scale),
            ),
        )
    }
}

/// Whether the device pixel center `center` lies inside `bounds`.
fn covers(bounds: Bounds<ScaledPixels>, center: (f32, f32)) -> bool {
    center.0 > bounds.left().0
        && center.0 < bounds.right().0
        && center.1 > bounds.top().0
        && center.1 < bounds.bottom().0
}

/// One node of a test scene; nodes paint in order.
#[derive(Clone)]
enum Node {
    /// An opaque quad.
    Quad(Rect, Paint),
    /// An opaque rectangle path.
    Path(Rect, Paint),
    /// A backdrop blur of radius zero and saturation zero: the luminance of
    /// the frame beneath every open layer.
    GrayBlur(Rect),
    /// A subtree composited through an edge fade of `fade` with `bands` as
    /// top, right, bottom, and left, clipped to the frame.
    Fade {
        fade: Rect,
        bands: [f32; 4],
        children: Vec<Node>,
    },
    /// A subtree composited through a rectangle path mask.
    Clip { clip: Rect, children: Vec<Node> },
}

fn fade(fade: Rect, bands: [f32; 4], children: Vec<Node>) -> Node {
    Node::Fade {
        fade,
        bands,
        children,
    }
}

fn clip(clip: Rect, children: Vec<Node>) -> Node {
    Node::Clip { clip, children }
}

fn edge_fade_mask(fade: Rect, [top, right, bottom, left]: [f32; 4], scale: f32) -> EdgeFadeMask {
    EdgeFadeMask::new(
        fade.device(scale),
        Edges {
            top: ScaledPixels(top * scale),
            right: ScaledPixels(right * scale),
            bottom: ScaledPixels(bottom * scale),
            left: ScaledPixels(left * scale),
        },
        FULL.device(scale),
    )
}

fn frame_content_mask(scale: f32) -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: FULL.device(scale),
        corner_radii: Corners::default(),
    }
}

fn rect_path(area: Rect, paint: Paint, scale: f32) -> Path<ScaledPixels> {
    let corner = |x: f32, y: f32| Point::new(px(x), px(y));
    let (left, top) = (area.x, area.y);
    let (right, bottom) = (area.x + area.width, area.y + area.height);
    let mut path = Path::new(corner(left, top));
    path.line_to(corner(right, top));
    path.line_to(corner(right, bottom));
    path.line_to(corner(left, bottom));
    path.line_to(corner(left, top));
    path.color = gpui::solid_background(paint.hsla());
    path.content_mask = ContentMask {
        bounds: Bounds::new(corner(0.0, 0.0), size(px(EXTENT), px(EXTENT))),
        corner_radii: Corners::default(),
    };
    path.scale(scale)
}

fn paint(scene: &mut Scene, nodes: &[Node], scale: f32) {
    for node in nodes {
        match node {
            Node::Quad(area, fill) => scene.insert_primitive(Quad {
                order: 0,
                border_style: Default::default(),
                bounds: area.device(scale),
                content_mask: frame_content_mask(scale),
                background: gpui::solid_background(fill.hsla()),
                border_color: Hsla::default(),
                corner_radii: Corners::default(),
                border_widths: Edges::default(),
                transformation: TransformationMatrix::unit(),
            }),
            Node::Path(area, fill) => scene.insert_primitive(rect_path(*area, *fill, scale)),
            Node::GrayBlur(area) => scene.insert_primitive(BackdropBlur {
                order: 0,
                pad: 0,
                bounds: area.device(scale),
                content_mask: frame_content_mask(scale),
                corner_radii: Corners::default(),
                blur_radius: ScaledPixels(0.0),
                saturation: 0.0,
                tint: Hsla::default(),
                transformation: TransformationMatrix::unit(),
            }),
            Node::Fade {
                fade,
                bands,
                children,
            } => {
                scene.push_layer_mask(LayerMask::EdgeFade(edge_fade_mask(*fade, *bands, scale)));
                paint(scene, children, scale);
                scene.pop_layer_mask();
            }
            Node::Clip { clip, children } => {
                scene.push_layer_mask(LayerMask::Path(rect_path(*clip, RED, scale)));
                paint(scene, children, scale);
                scene.pop_layer_mask();
            }
        }
    }
}

fn scene(nodes: &[Node], scale: f32) -> Scene {
    let mut scene = Scene::default();
    paint(&mut scene, nodes, scale);
    scene.finish();
    scene
}

/// The edge fade `m = r * r` at the device pixel center `center`, zero
/// outside the composited region.
fn edge_fade(mask: &EdgeFadeMask, center: (f32, f32)) -> f32 {
    if !covers(mask.bounds, center) {
        return 0.0;
    }
    let fade = mask.fade_bounds;
    let bands = mask.bands;
    let ratio = [
        (bands.top.0, center.1 - fade.top().0),
        (bands.right.0, fade.right().0 - center.0),
        (bands.bottom.0, fade.bottom().0 - center.1),
        (bands.left.0, center.0 - fade.left().0),
    ]
    .into_iter()
    .filter(|(band, _)| *band > 0.0)
    .map(|(band, distance)| (distance / band).clamp(0.0, 1.0))
    .fold(1.0, f32::min);
    ratio * ratio
}

/// `color` as stored in an 8-bit unorm target.
fn stored(color: Color) -> Color {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() / 255.0)
}

/// Composites the premultiplied `layer` onto `dst` through the mask value `m`.
fn composite(layer: Color, m: f32, dst: Color) -> Color {
    let coverage = layer[3] * m;
    [
        layer[0] * m + dst[0] * (1.0 - coverage),
        layer[1] * m + dst[1] * (1.0 - coverage),
        layer[2] * m + dst[2] * (1.0 - coverage),
        (coverage + dst[3]).min(1.0),
    ]
}

/// The color the model draws for `nodes` at the device pixel center
/// `center` onto `dst`. `backdrop` is the frame beneath every open layer,
/// `None` when `dst` is the frame.
fn model(
    nodes: &[Node],
    mut dst: Color,
    backdrop: Option<Color>,
    center: (f32, f32),
    scale: f32,
) -> Color {
    for node in nodes {
        dst = stored(match node {
            Node::Quad(area, fill) | Node::Path(area, fill) => {
                if covers(area.device(scale), center) {
                    fill.color()
                } else {
                    dst
                }
            }
            Node::GrayBlur(area) => {
                if covers(area.device(scale), center) {
                    let frame = backdrop.unwrap_or(dst);
                    let gray = frame[0] * 0.2126 + frame[1] * 0.7152 + frame[2] * 0.0722;
                    [gray, gray, gray, frame[3]]
                } else {
                    dst
                }
            }
            Node::Fade {
                fade,
                bands,
                children,
            } => {
                let layer = model(
                    children,
                    TRANSPARENT,
                    Some(backdrop.unwrap_or(dst)),
                    center,
                    scale,
                );
                let m = edge_fade(&edge_fade_mask(*fade, *bands, scale), center);
                composite(layer, m, dst)
            }
            Node::Clip { clip, children } => {
                let layer = model(
                    children,
                    TRANSPARENT,
                    Some(backdrop.unwrap_or(dst)),
                    center,
                    scale,
                );
                let m = if covers(clip.device(scale), center) {
                    1.0
                } else {
                    0.0
                };
                composite(layer, m, dst)
            }
        });
    }
    dst
}

fn renderer(scale: f32) -> WgpuRenderer {
    let extent = (EXTENT * scale) as i32;
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    WgpuRenderer::new_offscreen(&context, size(DevicePixels(extent), DevicePixels(extent)))
        .expect("offscreen renderer")
}

/// The pixels of the last frame whose device pixel center lies inside
/// `region` and whose channels differ from `expected` by more than 1/255
/// plus rounding, as `(x, y, actual, expected)`.
fn mismatches(
    renderer: &WgpuRenderer,
    scale: f32,
    region: Rect,
    expected: impl Fn((f32, f32)) -> Color,
) -> Vec<(usize, usize, [u8; 4], [f32; 4])> {
    let bytes = renderer.read_pixels().expect("read pixels");
    let extent = (EXTENT * scale) as usize;
    let region = region.device(scale);
    let mut mismatches = Vec::new();
    for y in 0..extent {
        for x in 0..extent {
            let center = (x as f32 + 0.5, y as f32 + 0.5);
            if !covers(region, center) {
                continue;
            }
            let offset = (y * extent + x) * 4;
            let actual: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
            let expected = expected(center).map(|channel| channel * 255.0);
            if actual
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (*actual as f32 - expected).abs() > 1.5)
            {
                mismatches.push((x, y, actual, expected));
            }
        }
    }
    mismatches
}

/// Draws `nodes` whole and asserts every pixel matches the model.
fn assert_frame(renderer: &mut WgpuRenderer, name: &str, nodes: &[Node], scale: f32) {
    assert!(renderer.draw(&scene(nodes, scale)), "{name}: draw failed");
    let wrong = mismatches(renderer, scale, FULL, |center| {
        model(nodes, TRANSPARENT, None, center, scale)
    });
    assert!(
        wrong.is_empty(),
        "{name} at scale {scale}: {} pixels differ from the model, first (x, y, actual, expected): {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)],
    );
}

const FADED: Rect = Rect {
    x: 10.0,
    y: 10.0,
    width: 80.0,
    height: 80.0,
};
const BAND: f32 = 20.0;
/// A horizontal stripe through the middle of the frame.
const STRIPE: Rect = Rect {
    x: 0.0,
    y: 40.0,
    width: EXTENT,
    height: 20.0,
};

fn cases() -> Vec<(&'static str, Vec<Node>)> {
    let base = || Node::Quad(FULL, BLUE);
    let red_fade = |bands: [f32; 4]| fade(FADED, bands, vec![Node::Quad(FULL, RED)]);
    let split_base = || {
        vec![
            Node::Quad(rect(0.0, 0.0, 50.0, EXTENT), RED),
            Node::Quad(rect(50.0, 0.0, 50.0, EXTENT), BLUE),
        ]
    };
    vec![
        ("top fade", vec![base(), red_fade([BAND, 0.0, 0.0, 0.0])]),
        ("right fade", vec![base(), red_fade([0.0, BAND, 0.0, 0.0])]),
        ("bottom fade", vec![base(), red_fade([0.0, 0.0, BAND, 0.0])]),
        ("left fade", vec![base(), red_fade([0.0, 0.0, 0.0, BAND])]),
        ("fade on every edge", vec![base(), red_fade([BAND; 4])]),
        (
            "bands wider than half the faded region",
            vec![
                base(),
                fade(
                    rect(20.0, 20.0, 60.0, 60.0),
                    [45.0, 40.0, 35.0, 50.0],
                    vec![Node::Quad(FULL, RED)],
                ),
            ],
        ),
        (
            "fade inside a path clip",
            vec![
                base(),
                clip(
                    rect(30.0, 0.0, 40.0, EXTENT),
                    vec![red_fade([BAND; 4]), Node::Quad(STRIPE, GREEN)],
                ),
            ],
        ),
        (
            "path clip inside a fade",
            vec![
                base(),
                fade(
                    FADED,
                    [BAND; 4],
                    vec![
                        clip(rect(30.0, 0.0, 40.0, EXTENT), vec![Node::Quad(FULL, RED)]),
                        Node::Quad(STRIPE, GREEN),
                    ],
                ),
            ],
        ),
        (
            "paths batch inside a fade",
            vec![
                base(),
                fade(
                    FADED,
                    [BAND; 4],
                    vec![Node::Path(FULL, RED), Node::Quad(STRIPE, GREEN)],
                ),
            ],
        ),
        (
            "backdrop blur inside a fade",
            [
                split_base(),
                vec![fade(
                    FADED,
                    [BAND; 4],
                    vec![
                        Node::Quad(FULL, GREEN),
                        Node::GrayBlur(rect(0.0, 20.0, EXTENT, 60.0)),
                        Node::Quad(rect(45.0, 45.0, 10.0, 10.0), RED),
                    ],
                )],
            ]
            .concat(),
        ),
        (
            "backdrop blur inside a fade inside a path clip",
            [
                split_base(),
                vec![clip(
                    rect(5.0, 5.0, 90.0, 90.0),
                    vec![
                        Node::Quad(FULL, GREEN),
                        fade(
                            FADED,
                            [BAND; 4],
                            vec![
                                Node::Quad(FULL, GREEN),
                                Node::GrayBlur(rect(0.0, 20.0, EXTENT, 60.0)),
                            ],
                        ),
                    ],
                )],
            ]
            .concat(),
        ),
    ]
}

#[test]
fn layer_masks_composite_as_the_model_at_scale_factors() {
    for scale in [1.0, 2.0] {
        let mut renderer = renderer(scale);
        for (name, nodes) in cases() {
            assert_frame(&mut renderer, name, &nodes, scale);
        }
    }
}

/// A partial frame repaints masked content inside the damage and keeps the
/// previous frame outside it.
#[test]
fn layer_masks_composite_inside_the_damage_scissor_at_scale_factors() {
    for scale in [1.0, 2.0] {
        let mut renderer = renderer(scale);
        let first = [
            Node::Quad(FULL, BLUE),
            fade(FADED, [BAND; 4], vec![Node::Quad(FULL, RED)]),
        ];
        assert_frame(&mut renderer, "first frame", &first, scale);

        let second = [
            Node::Quad(FULL, GREEN),
            clip(
                rect(20.0, 20.0, 60.0, 60.0),
                vec![fade(FADED, [BAND; 4], vec![Node::Quad(FULL, BLUE)])],
            ),
        ];
        let damage = rect(0.0, 0.0, 50.0, EXTENT);
        let mut partial = scene(&second, scale);
        partial.damage = Some(damage.device(scale));
        assert!(renderer.draw(&partial));

        let inside = mismatches(&renderer, scale, damage, |center| {
            model(&second, TRANSPARENT, None, center, scale)
        });
        let outside = mismatches(&renderer, scale, rect(50.0, 0.0, 50.0, EXTENT), |center| {
            model(&first, TRANSPARENT, None, center, scale)
        });
        assert!(
            inside.is_empty() && outside.is_empty(),
            "partial frame at scale {scale}: {} pixels inside the damage and {} outside differ, first: {:?} {:?}",
            inside.len(),
            outside.len(),
            inside.first(),
            outside.first(),
        );
    }
}

/// A frame without masks creates no layer texture; a layer is created at
/// the depth a frame first reaches and released after
/// `LAYER_IDLE_RELEASE_FRAMES` drawn frames that do not reach its depth,
/// deepest first; a renderer that released its layers draws masks again.
#[test]
fn layer_textures_are_created_on_demand_and_released_when_idle() {
    let scale = 1.0;
    let mut renderer = renderer(scale);
    let layers = |renderer: &WgpuRenderer| renderer.resources().layers.len();
    let plain = [Node::Quad(FULL, BLUE)];
    let one_mask = [
        Node::Quad(FULL, BLUE),
        fade(FADED, [BAND; 4], vec![Node::Quad(FULL, RED)]),
    ];
    let two_masks = [
        Node::Quad(FULL, BLUE),
        clip(
            rect(20.0, 20.0, 60.0, 60.0),
            vec![fade(FADED, [BAND; 4], vec![Node::Quad(FULL, RED)])],
        ),
    ];

    for _ in 0..3 {
        assert_frame(&mut renderer, "plain frame", &plain, scale);
    }
    assert_eq!(
        layers(&renderer),
        0,
        "a frame without masks created a layer"
    );

    assert_frame(&mut renderer, "two masks", &two_masks, scale);
    assert_eq!(layers(&renderer), 2);

    for frame in 1..LAYER_IDLE_RELEASE_FRAMES {
        assert!(renderer.draw(&scene(&one_mask, scale)));
        assert_eq!(layers(&renderer), 2, "released after {frame} idle frames");
    }
    assert_frame(&mut renderer, "one mask", &one_mask, scale);
    assert_eq!(
        layers(&renderer),
        1,
        "the second layer outlived {LAYER_IDLE_RELEASE_FRAMES} idle frames"
    );

    for frame in 1..LAYER_IDLE_RELEASE_FRAMES {
        assert!(renderer.draw(&scene(&plain, scale)));
        assert_eq!(layers(&renderer), 1, "released after {frame} idle frames");
    }
    assert_frame(&mut renderer, "plain frame after masks", &plain, scale);
    assert_eq!(
        layers(&renderer),
        0,
        "the first layer outlived {LAYER_IDLE_RELEASE_FRAMES} idle frames"
    );

    assert_frame(&mut renderer, "two masks after release", &two_masks, scale);
    assert_eq!(layers(&renderer), 2);
}
