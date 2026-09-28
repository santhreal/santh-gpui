// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point, px, radians, size,
};
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

/// One `(z_index, order)` pair per enclosing scope, root first, each packed
/// by `sort_step`. A layer's children extend the layer's own key, so they
/// sort after it and before the next sibling of the layer; two primitives in
/// one scope that do not overlap share an order and keep batching together.
type SortKey = SmallVec<[u64; 4]>;

/// `(z_index, order)` as one integer that orders as the pair does.
fn sort_step(z_index: i32, order: DrawOrder) -> u64 {
    (u64::from(z_index.cast_unsigned() ^ 0x8000_0000) << 32) | u64::from(order)
}

/// The paint scope a primitive is inserted into: the root, or one layer.
struct Scope {
    /// The key of the layer that opened this scope; empty at the root.
    prefix: SortKey,
    /// The z-index applied to primitives inserted directly in this scope.
    z_index: i32,
    /// Index into `Scene::bounds_trees`; a tree per open layer so that overlap
    /// inside a layer is ordered by paint sequence rather than by primitive kind.
    tree: usize,
}

/// Where one primitive sits in its kind's vector, with the key it sorts by.
struct Rank {
    key: SortKey,
    kind: PrimitiveKind,
    /// The atlas tile that orders sprites of one draw order; zero for every
    /// other kind.
    tile: u32,
    index: u32,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    bounds_trees: Vec<BoundsTree<ScaledPixels>>,
    scopes: Vec<Scope>,
    z_index_stack: Vec<i32>,
    ranks: Vec<Rank>,
    /// Scratch for `finish`: per kind, the position each primitive moves to,
    /// indexed by its position before the move.
    destinations: [Vec<u32>; PrimitiveKind::ALL.len()],
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub backdrop_blurs: Vec<BackdropBlur>,
    pub start_layer_masks: Vec<StartLayerMask>,
    pub end_layer_masks: Vec<EndLayerMask>,
    pub damage: Option<Bounds<ScaledPixels>>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        for tree in &mut self.bounds_trees {
            tree.clear();
        }
        self.scopes.clear();
        self.z_index_stack.clear();
        self.ranks.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.subpixel_sprites.clear();
        self.monochrome_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.backdrop_blurs.clear();
        self.start_layer_masks.clear();
        self.end_layer_masks.clear();
        self.damage = None;
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    fn scope(&mut self) -> &mut Scope {
        if self.scopes.is_empty() {
            if self.bounds_trees.is_empty() {
                self.bounds_trees.push(BoundsTree::default());
            }
            self.scopes.push(Scope {
                prefix: SmallVec::new(),
                z_index: 0,
                tree: 0,
            });
        }
        let last = self.scopes.len() - 1;
        &mut self.scopes[last]
    }

    /// The key a primitive or layer with `bounds` gets in the current scope.
    fn key_for(&mut self, bounds: Bounds<ScaledPixels>) -> SortKey {
        let scope = self.scope();
        let tree = scope.tree;
        let z_index = scope.z_index;
        let mut key = scope.prefix.clone();
        let order = self.bounds_trees[tree].insert(bounds);
        key.push(sort_step(z_index, order));
        key
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.open_scope(bounds);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.close_scope();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Opens the scope of a layer with `bounds` without recording a paint
    /// operation, for the callers whose own operation reopens it on replay.
    fn open_scope(&mut self, bounds: Bounds<ScaledPixels>) {
        let prefix = self.key_for(bounds);
        let tree = self.scopes.len();
        if self.bounds_trees.len() <= tree {
            self.bounds_trees.push(BoundsTree::default());
        } else {
            self.bounds_trees[tree].clear();
        }
        self.scopes.push(Scope {
            prefix,
            z_index: 0,
            tree,
        });
    }

    fn close_scope(&mut self) {
        // The root scope is never closed: a scope is only ever closed by the
        // caller that opened it, so a second scope exists here.
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Sorts primitives inserted until the matching `pop_z_index` at `z_index`
    /// within the current scope: above every primitive of the scope with a
    /// lower z-index and below every one with a higher, whatever their kind or
    /// paint sequence.
    pub fn push_z_index(&mut self, z_index: i32) {
        let scope = self.scope();
        let previous = std::mem::replace(&mut scope.z_index, z_index);
        self.z_index_stack.push(previous);
        self.paint_operations
            .push(PaintOperation::PushZIndex(z_index));
    }

    pub fn pop_z_index(&mut self) {
        if let Some(previous) = self.z_index_stack.pop() {
            self.scope().z_index = previous;
        }
        self.paint_operations.push(PaintOperation::PopZIndex);
    }

    /// Start a subtree that draws into a layer composited through `mask`.
    /// See [`StartLayerMask`].
    pub fn push_layer_mask(&mut self, mask: LayerMask) {
        self.open_scope(mask.bounds());
        let key = self.scope_edge_key(i32::MIN, DrawOrder::MIN);
        let index = self.start_layer_masks.len();
        self.start_layer_masks
            .push(StartLayerMask { order: 0, mask });
        self.record(key, PrimitiveKind::StartLayerMask, 0, index);
    }

    /// End the innermost masked subtree.
    pub fn pop_layer_mask(&mut self) {
        let key = self.scope_edge_key(i32::MAX, DrawOrder::MAX);
        let index = self.end_layer_masks.len();
        self.end_layer_masks.push(EndLayerMask { order: 0 });
        self.record(key, PrimitiveKind::EndLayerMask, 0, index);
        self.close_scope();
    }

    /// A key in the current scope that sorts before every child of the
    /// scope when `(z_index, order)` is the minimum and after every child
    /// when it is the maximum, whatever the children's z-indices and bounds.
    fn scope_edge_key(&mut self, z_index: i32, order: DrawOrder) -> SortKey {
        let mut key = self.scope().prefix.clone();
        key.push(sort_step(z_index, order));
        key
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        match primitive.into() {
            Primitive::Shadow(shadow) => self.store(shadow),
            Primitive::Quad(quad) => self.store(quad),
            Primitive::Path(path) => self.store(path),
            Primitive::Underline(underline) => self.store(underline),
            Primitive::MonochromeSprite(sprite) => self.store(sprite),
            Primitive::SubpixelSprite(sprite) => self.store(sprite),
            Primitive::PolychromeSprite(sprite) => self.store(sprite),
            Primitive::Surface(surface) => self.store(surface),
            Primitive::BackdropBlur(blur) => self.store(blur),
        }
    }

    /// Moves `primitive` into the vector of its kind and logs a reference to
    /// it, unless its content mask clips it away.
    fn store<P: Stored>(&mut self, primitive: P) {
        let clipped_bounds = primitive.clipped_bounds();
        if !clipped_bounds.is_empty() {
            self.store_visible(primitive, clipped_bounds);
        }
    }

    fn store_visible<P: Stored>(&mut self, mut primitive: P, clipped_bounds: Bounds<ScaledPixels>) {
        let key = self.key_for(clipped_bounds);
        let tile = primitive.tile();
        let stored = P::stored_mut(self);
        let index = stored.len();
        primitive.stored_at(index);
        stored.push(primitive);
        self.record(key, P::KIND, tile, index);
    }

    /// Ranks the primitive of `kind` at `index` by `key`, and logs a
    /// reference to it.
    ///
    /// Each primitive is stored once, in the vector of its kind. The log holds
    /// eight bytes per primitive rather than a second copy, and a range of it
    /// replays from the vectors of the scene that recorded it.
    fn record(&mut self, key: SortKey, kind: PrimitiveKind, tile: u32, index: usize) {
        let index = index as u32;
        self.ranks.push(Rank {
            key,
            kind,
            tile,
            index,
        });
        self.paint_operations
            .push(PaintOperation::Primitive(PrimitiveRef { kind, index }));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match *operation {
                PaintOperation::Primitive(PrimitiveRef { kind, index }) => match kind {
                    PrimitiveKind::Shadow => self.replay_stored::<Shadow>(index, prev_scene),
                    PrimitiveKind::Quad => self.replay_stored::<Quad>(index, prev_scene),
                    PrimitiveKind::Path => {
                        self.replay_stored::<Path<ScaledPixels>>(index, prev_scene)
                    }
                    PrimitiveKind::Underline => self.replay_stored::<Underline>(index, prev_scene),
                    PrimitiveKind::MonochromeSprite => {
                        self.replay_stored::<MonochromeSprite>(index, prev_scene)
                    }
                    PrimitiveKind::SubpixelSprite => {
                        self.replay_stored::<SubpixelSprite>(index, prev_scene)
                    }
                    PrimitiveKind::PolychromeSprite => {
                        self.replay_stored::<PolychromeSprite>(index, prev_scene)
                    }
                    PrimitiveKind::Surface => self.replay_stored::<PaintSurface>(index, prev_scene),
                    PrimitiveKind::BackdropBlur => {
                        self.replay_stored::<BackdropBlur>(index, prev_scene)
                    }
                    PrimitiveKind::StartLayerMask => self
                        .push_layer_mask(prev_scene.start_layer_masks[index as usize].mask.clone()),
                    PrimitiveKind::EndLayerMask => self.pop_layer_mask(),
                },
                PaintOperation::StartLayer(bounds) => self.push_layer(bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::PushZIndex(z_index) => self.push_z_index(z_index),
                PaintOperation::PopZIndex => self.pop_z_index(),
            }
        }
    }

    /// Clones the primitive at `index` of `prev_scene` straight into the
    /// vector of its kind. It reached the log of `prev_scene`, so its content
    /// mask does not clip it away.
    fn replay_stored<P: Stored>(&mut self, index: u32, prev_scene: &Scene) {
        let source = &P::stored(prev_scene)[index as usize];
        self.store_visible(source.clone(), source.clipped_bounds());
    }

    /// The union of the window regions that the primitives logged at
    /// `start..` of the paint log draw into, or `None` when none was logged.
    /// Read during paint, before [`Self::finish`] reorders the primitives.
    pub(crate) fn painted_bounds_since(&self, start: usize) -> Option<Bounds<ScaledPixels>> {
        self.paint_operations
            .get(start..)?
            .iter()
            .filter_map(|operation| match *operation {
                PaintOperation::Primitive(PrimitiveRef { kind, index }) => {
                    self.clipped_bounds_of(kind, index as usize)
                }
                PaintOperation::StartLayer(_)
                | PaintOperation::EndLayer
                | PaintOperation::PushZIndex(_)
                | PaintOperation::PopZIndex => None,
            })
            .reduce(|union, bounds| union.union(&bounds))
    }

    /// The clipped window region of the primitive of `kind` at `index`, or
    /// `None` for a layer mask marker, which draws nothing itself.
    fn clipped_bounds_of(&self, kind: PrimitiveKind, index: usize) -> Option<Bounds<ScaledPixels>> {
        fn stored<P: Stored>(scene: &Scene, index: usize) -> Bounds<ScaledPixels> {
            Stored::clipped_bounds(&P::stored(scene)[index])
        }
        Some(match kind {
            PrimitiveKind::Shadow => stored::<Shadow>(self, index),
            PrimitiveKind::Quad => stored::<Quad>(self, index),
            PrimitiveKind::Path => stored::<Path<ScaledPixels>>(self, index),
            PrimitiveKind::Underline => stored::<Underline>(self, index),
            PrimitiveKind::MonochromeSprite => stored::<MonochromeSprite>(self, index),
            PrimitiveKind::SubpixelSprite => stored::<SubpixelSprite>(self, index),
            PrimitiveKind::PolychromeSprite => stored::<PolychromeSprite>(self, index),
            PrimitiveKind::Surface => stored::<PaintSurface>(self, index),
            PrimitiveKind::BackdropBlur => stored::<BackdropBlur>(self, index),
            PrimitiveKind::StartLayerMask | PrimitiveKind::EndLayerMask => return None,
        })
    }

    pub fn finish(&mut self) {
        // The key fixes the draw order. The sort is stable, so primitives of
        // one kind keep paint order within a key, except that sprites order
        // by atlas tile first. The position of every primitive is then its
        // rank among the primitives of its kind.
        self.ranks
            .sort_by(|a, b| a.key.cmp(&b.key).then(a.tile.cmp(&b.tile)));
        let mut destinations = std::mem::take(&mut self.destinations);
        for kind in PrimitiveKind::ALL {
            let moves = &mut destinations[kind as usize];
            moves.clear();
            moves.resize(self.stored_len(kind), 0);
        }
        let mut next = [0u32; PrimitiveKind::ALL.len()];
        let mut order: DrawOrder = 0;
        for position in 0..self.ranks.len() {
            if position > 0 && self.ranks[position - 1].key != self.ranks[position].key {
                order += 1;
            }
            let rank = &mut self.ranks[position];
            let index = rank.index as usize;
            let destination = &mut next[rank.kind as usize];
            destinations[rank.kind as usize][index] = *destination;
            rank.index = *destination;
            *destination += 1;
            match rank.kind {
                PrimitiveKind::Shadow => self.shadows[index].order = order,
                PrimitiveKind::Quad => self.quads[index].order = order,
                PrimitiveKind::Path => self.paths[index].order = order,
                PrimitiveKind::Underline => self.underlines[index].order = order,
                PrimitiveKind::MonochromeSprite => self.monochrome_sprites[index].order = order,
                PrimitiveKind::SubpixelSprite => self.subpixel_sprites[index].order = order,
                PrimitiveKind::PolychromeSprite => self.polychrome_sprites[index].order = order,
                PrimitiveKind::Surface => self.surfaces[index].order = order,
                PrimitiveKind::BackdropBlur => self.backdrop_blurs[index].order = order,
                PrimitiveKind::StartLayerMask => self.start_layer_masks[index].order = order,
                PrimitiveKind::EndLayerMask => self.end_layer_masks[index].order = order,
            }
        }

        // A reference moves with its primitive, so a finished scene, which is
        // the one a later frame replays from, resolves every reference to the
        // primitive it was logged for.
        for operation in &mut self.paint_operations {
            if let PaintOperation::Primitive(reference) = operation {
                reference.index = destinations[reference.kind as usize][reference.index as usize];
            }
        }
        for kind in PrimitiveKind::ALL {
            self.move_to_destinations(kind, &mut destinations[kind as usize]);
        }
        self.destinations = destinations;
    }

    fn stored_len(&self, kind: PrimitiveKind) -> usize {
        match kind {
            PrimitiveKind::Shadow => self.shadows.len(),
            PrimitiveKind::Quad => self.quads.len(),
            PrimitiveKind::Path => self.paths.len(),
            PrimitiveKind::Underline => self.underlines.len(),
            PrimitiveKind::MonochromeSprite => self.monochrome_sprites.len(),
            PrimitiveKind::SubpixelSprite => self.subpixel_sprites.len(),
            PrimitiveKind::PolychromeSprite => self.polychrome_sprites.len(),
            PrimitiveKind::Surface => self.surfaces.len(),
            PrimitiveKind::BackdropBlur => self.backdrop_blurs.len(),
            PrimitiveKind::StartLayerMask => self.start_layer_masks.len(),
            PrimitiveKind::EndLayerMask => self.end_layer_masks.len(),
        }
    }

    fn move_to_destinations(&mut self, kind: PrimitiveKind, destinations: &mut [u32]) {
        match kind {
            PrimitiveKind::Shadow => permute(&mut self.shadows, destinations),
            PrimitiveKind::Quad => permute(&mut self.quads, destinations),
            PrimitiveKind::Path => permute(&mut self.paths, destinations),
            PrimitiveKind::Underline => permute(&mut self.underlines, destinations),
            PrimitiveKind::MonochromeSprite => permute(&mut self.monochrome_sprites, destinations),
            PrimitiveKind::SubpixelSprite => permute(&mut self.subpixel_sprites, destinations),
            PrimitiveKind::PolychromeSprite => permute(&mut self.polychrome_sprites, destinations),
            PrimitiveKind::Surface => permute(&mut self.surfaces, destinations),
            PrimitiveKind::BackdropBlur => permute(&mut self.backdrop_blurs, destinations),
            PrimitiveKind::StartLayerMask => permute(&mut self.start_layer_masks, destinations),
            PrimitiveKind::EndLayerMask => permute(&mut self.end_layer_masks, destinations),
        }
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            backdrop_blurs_start: 0,
            backdrop_blurs_iter: self.backdrop_blurs.iter().peekable(),
            start_layer_masks_iter: self.start_layer_masks.iter().peekable(),
            end_layer_masks_iter: self.end_layer_masks.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    StartLayerMask,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    BackdropBlur,
    EndLayerMask,
}

impl PrimitiveKind {
    /// Every kind, with `ALL[kind as usize] == kind`.
    const ALL: [PrimitiveKind; 11] = [
        PrimitiveKind::StartLayerMask,
        PrimitiveKind::Shadow,
        PrimitiveKind::Quad,
        PrimitiveKind::Path,
        PrimitiveKind::Underline,
        PrimitiveKind::MonochromeSprite,
        PrimitiveKind::SubpixelSprite,
        PrimitiveKind::PolychromeSprite,
        PrimitiveKind::Surface,
        PrimitiveKind::BackdropBlur,
        PrimitiveKind::EndLayerMask,
    ];
}

const _: () = {
    let mut index = 0;
    while index < PrimitiveKind::ALL.len() {
        assert!(PrimitiveKind::ALL[index] as usize == index);
        index += 1;
    }
};

/// Moves `items[i]` to `items[destinations[i]]` for every `i`, in place.
///
/// `destinations` is a permutation of `0..items.len()` and is left as the
/// identity. Every swap puts one item where it belongs, and an item that is
/// where it belongs never moves again, so this makes fewer than
/// `items.len()` swaps even when `destinations` is not a permutation.
fn permute<T>(items: &mut [T], destinations: &mut [u32]) {
    debug_assert_eq!(items.len(), destinations.len());
    for start in 0..items.len() {
        loop {
            let destination = destinations[start] as usize;
            if destination == start || destinations[destination] as usize == destination {
                debug_assert_eq!(destination, start, "destinations is not a permutation");
                break;
            }
            items.swap(start, destination);
            destinations.swap(start, destination);
        }
    }
}

/// A primitive or layer mask marker recorded in the paint log: the kind of
/// vector it is stored in, and its position there.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrimitiveRef {
    kind: PrimitiveKind,
    index: u32,
}

pub(crate) enum PaintOperation {
    Primitive(PrimitiveRef),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    PushZIndex(i32),
    PopZIndex,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BackdropBlur(BackdropBlur),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BackdropBlur(blur) => &blur.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BackdropBlur(blur) => &blur.content_mask,
        }
    }

    /// The transformation from the primitive's `bounds` to window space.
    pub fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::Shadow(shadow) => shadow.transformation,
            Primitive::Quad(quad) => quad.transformation,
            Primitive::Path(path) => path.transformation,
            Primitive::Underline(underline) => underline.transformation,
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            Primitive::PolychromeSprite(sprite) => sprite.transformation,
            Primitive::Surface(_) => TransformationMatrix::unit(),
            Primitive::BackdropBlur(blur) => blur.transformation,
        }
    }
}

/// A primitive a scene stores in a vector of its kind.
trait Stored: Clone {
    const KIND: PrimitiveKind;
    fn stored(scene: &Scene) -> &Vec<Self>;
    fn stored_mut(scene: &mut Scene) -> &mut Vec<Self>;
    /// The region of the window the primitive can draw into: its bounds
    /// after its transformation, within its content mask.
    fn clipped_bounds(&self) -> Bounds<ScaledPixels>;
    /// The atlas tile that orders sprites of one draw order; zero for every
    /// other kind.
    fn tile(&self) -> u32 {
        0
    }
    /// Records that the primitive is stored at `index` of its vector.
    fn stored_at(&mut self, _index: usize) {}
}

macro_rules! impl_stored {
    ($primitive:ty, $kind:ident, $field:ident { $($extra:tt)* }) => {
        impl Stored for $primitive {
            const KIND: PrimitiveKind = PrimitiveKind::$kind;
            fn stored(scene: &Scene) -> &Vec<Self> {
                &scene.$field
            }
            fn stored_mut(scene: &mut Scene) -> &mut Vec<Self> {
                &mut scene.$field
            }
            fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
                self.transformation
                    .apply_to_bounds(self.bounds)
                    .intersect(&self.content_mask.bounds)
            }
            $($extra)*
        }
    };
}

impl_stored!(Shadow, Shadow, shadows {});
impl_stored!(Quad, Quad, quads {});
impl_stored!(Path<ScaledPixels>, Path, paths {
    fn stored_at(&mut self, index: usize) {
        self.id = PathId(index);
    }
});
impl_stored!(Underline, Underline, underlines {});
impl_stored!(MonochromeSprite, MonochromeSprite, monochrome_sprites {
    fn tile(&self) -> u32 {
        self.tile.tile_id.0
    }
});
impl_stored!(SubpixelSprite, SubpixelSprite, subpixel_sprites {
    fn tile(&self) -> u32 {
        self.tile.tile_id.0
    }
});
impl_stored!(PolychromeSprite, PolychromeSprite, polychrome_sprites {
    fn tile(&self) -> u32 {
        self.tile.tile_id.0
    }
});
impl_stored!(BackdropBlur, BackdropBlur, backdrop_blurs {});

impl Stored for PaintSurface {
    const KIND: PrimitiveKind = PrimitiveKind::Surface;
    fn stored(scene: &Scene) -> &Vec<Self> {
        &scene.surfaces
    }
    fn stored_mut(scene: &mut Scene) -> &mut Vec<Self> {
        &mut scene.surfaces
    }
    fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    backdrop_blurs_start: usize,
    backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
    start_layer_masks_iter: Peekable<slice::Iter<'a, StartLayerMask>>,
    end_layer_masks_iter: Peekable<slice::Iter<'a, EndLayerMask>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.backdrop_blurs_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (
                self.start_layer_masks_iter.peek().map(|m| m.order),
                PrimitiveKind::StartLayerMask,
            ),
            (
                self.end_layer_masks_iter.peek().map(|m| m.order),
                PrimitiveKind::EndLayerMask,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::BackdropBlur => {
                let backdrop_blurs_start = self.backdrop_blurs_start;
                let mut backdrop_blurs_end = backdrop_blurs_start + 1;
                self.backdrop_blurs_iter.next();
                while self
                    .backdrop_blurs_iter
                    .next_if(|blur| (blur.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    backdrop_blurs_end += 1;
                }
                self.backdrop_blurs_start = backdrop_blurs_end;
                Some(PrimitiveBatch::BackdropBlurs(
                    backdrop_blurs_start..backdrop_blurs_end,
                ))
            }
            PrimitiveKind::StartLayerMask => {
                let start = self.start_layer_masks_iter.next().unwrap();
                Some(PrimitiveBatch::StartLayerMask(start.mask.clone()))
            }
            PrimitiveKind::EndLayerMask => {
                self.end_layer_masks_iter.next().unwrap();
                Some(PrimitiveBatch::EndLayerMask)
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    BackdropBlurs(Range<usize>),
    StartLayerMask(LayerMask),
    EndLayerMask,
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::BackdropBlurs(range) => format!("backdrop blurs ({})", range.len()),
            Self::StartLayerMask(LayerMask::Path(_)) => "start path clip".to_string(),
            Self::StartLayerMask(LayerMask::EdgeFade(_)) => "start edge fade".to_string(),
            Self::EndLayerMask => "end layer mask".to_string(),
        }
    }
}

/// Marker primitive opening a masked subtree.
///
/// Primitives between a `StartLayerMask` and its [`EndLayerMask`] draw into a
/// layer cleared to transparent black. At the `EndLayerMask` the layer, which
/// holds premultiplied color, is composited onto the target beneath it inside
/// [`LayerMask::bounds`]: `dst.rgb = layer.rgb * m + dst.rgb * (1 - layer.a * m)`,
/// where `m` is the value of `mask` at the pixel; destination alpha combines
/// as it does for path sprites. Masks nest: each open mask draws into its own
/// layer. A [`BackdropBlur`] inside a mask samples the frame beneath every
/// open layer.
#[derive(Clone, Debug)]
pub struct StartLayerMask {
    /// The draw order
    pub order: DrawOrder,
    /// The mask the layer is composited through
    pub mask: LayerMask,
}

/// Marker primitive closing a masked subtree. See [`StartLayerMask`].
#[derive(Clone, Copy, Debug)]
pub struct EndLayerMask {
    /// The draw order
    pub order: DrawOrder,
}

/// The mask value `m` a layer is composited through. See [`StartLayerMask`].
#[derive(Clone, Debug)]
pub enum LayerMask {
    /// `m` is the alpha of the path rasterized with path antialiasing. The
    /// path is rasterized when the mask closes, so paths drawn inside the
    /// subtree do not change `m`.
    Path(Path<ScaledPixels>),
    /// `m = r * r`, with `r` the fade ratio documented on [`EdgeFadeMask`].
    EdgeFade(EdgeFadeMask),
}

impl LayerMask {
    /// The region of the target the layer is composited into. `m` is zero
    /// outside it.
    pub fn bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Self::Path(path) => path.transformation.apply_to_bounds(path.clipped_bounds()),
            Self::EdgeFade(fade) => fade.bounds,
        }
    }
}

/// An edge fade in device pixels, in the layout the GPU reads it.
///
/// At a pixel center `p`, the fade ratio `r` is the minimum, over each edge
/// of `fade_bounds` with a nonzero band, of `clamp(d / band, 0, 1)`, where
/// `d` is the distance from `p` to that edge measured toward the inside of
/// `fade_bounds`. `r` is 1 when every band is zero, and 0 beyond an edge with
/// a nonzero band.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct EdgeFadeMask {
    /// The region the layer is composited into: the `clip` passed to
    /// [`EdgeFadeMask::new`], limited at each edge of `fade_bounds` with a
    /// nonzero band.
    pub bounds: Bounds<ScaledPixels>,
    /// The faded region.
    pub fade_bounds: Bounds<ScaledPixels>,
    /// The width of the ramp inside each edge of `fade_bounds`; zero leaves
    /// that edge unfaded.
    pub bands: Edges<ScaledPixels>,
}

impl EdgeFadeMask {
    /// Fades `fade_bounds` across `bands`, composited inside `clip`. A
    /// negative or NaN band is zero.
    pub fn new(
        fade_bounds: Bounds<ScaledPixels>,
        bands: Edges<ScaledPixels>,
        clip: Bounds<ScaledPixels>,
    ) -> Self {
        let bands = bands.map(|band| ScaledPixels(band.0.max(0.0)));
        // A faded edge limits the clip; an unfaded edge leaves it open.
        let (fade, clip) = (fade_bounds, clip);
        let faded = |band: ScaledPixels, limited: f32, open: f32| {
            if band.0 > 0.0 { limited } else { open }
        };
        let left = faded(bands.left, fade.left().0.max(clip.left().0), clip.left().0);
        let top = faded(bands.top, fade.top().0.max(clip.top().0), clip.top().0);
        let right = faded(
            bands.right,
            fade.right().0.min(clip.right().0),
            clip.right().0,
        );
        let bottom = faded(
            bands.bottom,
            fade.bottom().0.min(clip.bottom().0),
            clip.bottom().0,
        );
        let (right, bottom) = (right.max(left), bottom.max(top));
        Self {
            bounds: Bounds::from_corners(
                point(ScaledPixels(left), ScaledPixels(top)),
                point(ScaledPixels(right), ScaledPixels(bottom)),
            ),
            fade_bounds,
            bands,
        }
    }
}

/// Consecutive frames without a backdrop blur (or without a layer mask) after
/// which a renderer releases the offscreen textures that feature uses.
pub const LAYER_IDLE_RELEASE_FRAMES: u32 = 30;

/// Counts consecutive frames in which a renderer feature went unused.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerIdleCounter(u32);

impl LayerIdleCounter {
    /// Records one finished frame. Returns true when the feature has gone
    /// unused for [`LAYER_IDLE_RELEASE_FRAMES`] consecutive frames, and on
    /// every later unused frame. A used frame restarts the count.
    pub fn tick(&mut self, used: bool) -> bool {
        if used {
            self.0 = 0;
            return false;
        }
        self.0 = self.0.saturating_add(1);
        self.0 >= LAYER_IDLE_RELEASE_FRAMES
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub transformation: TransformationMatrix,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
    pub transformation: TransformationMatrix,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
    pub transformation: TransformationMatrix,
}
impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A region that shows the frame drawn before it, blurred, saturated, and
/// tinted.
///
/// `bounds` and `corner_radii` give the body shape before `transformation`,
/// `content_mask` the rounded clip in window space, `blur_radius` the sampling
/// radius in device pixels, `saturation` the saturation factor (1 leaves
/// colors unchanged), and `tint` the color mixed over the result by its alpha.
///
/// For a pixel at window position `p`, `B(x)` is the frame drawn before this
/// primitive (beneath every open path clip layer), sampled with bilinear
/// filtering and clamped to the frame edge. Every backend computes:
///
/// - Blur, with `r = blur_radius`: the weighted mean of `B(p)` with weight 1
///   and 24 taps `B(p + r * k * (cos t, sin t))` for `i` in `0..8`: ring 1
///   `k = 0.38`, weight 0.637, `t = i * pi / 4`; ring 2 `k = 0.70`, weight
///   0.216, `t = (i + 0.5) * pi / 4`; ring 3 `k = 1.00`, weight 0.044,
///   `t = i * pi / 4`. When `r <= 0` the result is `B(p)`.
/// - Saturation: `L = dot(rgb, (0.2126, 0.7152, 0.0722))` and
///   `rgb' = clamp(L + saturation * (rgb - L), 0, 1)`.
/// - Tint: `rgb'' = mix(rgb', tint.rgb, tint.a)`, with `tint` in RGBA.
/// - Coverage: `saturate(0.5 - sdf(q, bounds, corner_radii)) *
///   saturate(0.5 - sdf(p, content_mask.bounds, content_mask.corner_radii))`,
///   where `q` is `p` before `transformation` and `sdf` is the rounded
///   rectangle signed distance that quads use.
/// - Output: color `rgb''` with alpha `a * coverage`, where `a` is the alpha
///   of the blurred color, blended source-over onto the target.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BackdropBlur {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub blur_radius: ScaledPixels,
    pub saturation: f32,
    pub tint: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<BackdropBlur> for Primitive {
    fn from(blur: BackdropBlur) -> Self {
        Primitive::BackdropBlur(blur)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A transformation to apply to an element.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Transformation {
    /// Scaling factor along x and y axes.
    pub scale: Size<f32>,
    /// Translation offset.
    pub translate: Point<Pixels>,
    /// Rotation angle in radians.
    pub rotate: Radians,
    /// Transform origin. If None, the center of the element bounds is used.
    pub origin: Option<Point<Pixels>>,
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
            origin: None,
        }
    }
}

impl Transformation {
    /// Create an identity transformation.
    pub fn unit() -> Self {
        Self::default()
    }

    /// Create a transformation with the specified scale along each axis.
    pub fn scale(scale: Size<f32>) -> Self {
        Self {
            scale,
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
            origin: None,
        }
    }

    /// Create a transformation with uniform scaling.
    pub fn uniform_scale(factor: f32) -> Self {
        Self::scale(size(factor, factor))
    }

    /// Create a transformation with the specified translation.
    pub fn translate(translate: Point<Pixels>) -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate,
            rotate: radians(0.0),
            origin: None,
        }
    }

    /// Create a transformation with the specified rotation in radians.
    pub fn rotate(rotate: impl Into<Radians>) -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate: rotate.into(),
            origin: None,
        }
    }

    /// Set the transform origin.
    pub fn with_origin(mut self, origin: Point<Pixels>) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Update the scaling factor of this transformation.
    pub fn with_scaling(mut self, scale: Size<f32>) -> Self {
        self.scale = scale;
        self
    }

    /// Update the translation value of this transformation.
    pub fn with_translation(mut self, translate: Point<Pixels>) -> Self {
        self.translate = translate;
        self
    }

    /// Update the rotation angle of this transformation.
    pub fn with_rotation(mut self, rotate: impl Into<Radians>) -> Self {
        self.rotate = rotate.into();
        self
    }

    /// Convert this transformation into a 2x3 affine matrix for rendering.
    pub fn into_matrix(self, center: Point<Pixels>, scale_factor: f32) -> TransformationMatrix {
        let center = self.origin.unwrap_or(center);
        TransformationMatrix::unit()
            .translate(center.scale(scale_factor) + self.translate.scale(scale_factor))
            .rotate(self.rotate)
            .scale(self.scale)
            .translate(center.scale(-scale_factor))
    }

    /// Linearly interpolate between this and another transformation.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            scale: crate::size(
                self.scale.width + (other.scale.width - self.scale.width) * t,
                self.scale.height + (other.scale.height - self.scale.height) * t,
            ),
            translate: crate::point(
                crate::px(self.translate.x.0 + (other.translate.x.0 - self.translate.x.0) * t),
                crate::px(self.translate.y.0 + (other.translate.y.0 - self.translate.y.0) * t),
            ),
            rotate: crate::radians(self.rotate.0 + (other.rotate.0 - self.rotate.0) * t),
            origin: other.origin.or(self.origin),
        }
    }
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }

    /// Apply transformation to a point in scaled pixels.
    pub fn apply_scaled(&self, point: Point<ScaledPixels>) -> Point<ScaledPixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(ScaledPixels(output[0]), ScaledPixels(output[1]))
    }

    /// Apply this transformation to the 4 corners of a bounding box and return the
    /// axis-aligned bounding box that encloses the transformed corners.
    pub fn apply_to_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let tl = self.apply_scaled(bounds.origin);
        let tr = self.apply_scaled(point(bounds.right(), bounds.top()));
        let br = self.apply_scaled(point(bounds.right(), bounds.bottom()));
        let bl = self.apply_scaled(point(bounds.left(), bounds.bottom()));

        let min_x = tl.x.min(tr.x).min(br.x).min(bl.x);
        let max_x = tl.x.max(tr.x).max(br.x).max(bl.x);
        let min_y = tl.y.min(tr.y).min(br.y).min(bl.y);
        let max_y = tl.y.max(tr.y).max(br.y).max(bl.y);

        Bounds {
            origin: point(min_x, min_y),
            size: size(max_x - min_x, max_y - min_y),
        }
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    pub transformation: TransformationMatrix,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            transformation: TransformationMatrix::unit(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
            transformation: self.transformation,
        }
    }
    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

/// A 1x1 BGRA pixel buffer for tests that insert a [`PaintSurface`].
/// `CVPixelBuffer` has no `Default`.
#[cfg(all(test, target_os = "macos"))]
fn test_image_buffer() -> core_video::pixel_buffer::CVPixelBuffer {
    use core_video::pixel_buffer::{CVPixelBuffer, kCVPixelFormatType_32BGRA};

    CVPixelBuffer::new(kCVPixelFormatType_32BGRA, 1, 1, None)
        .expect("CoreVideo creates a 1x1 BGRA pixel buffer")
}

#[cfg(test)]
mod transformed_bounds_tests;

#[cfg(test)]
mod layer_mask_tests;

#[cfg(test)]
mod paint_log_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BorderStyle, ScaledPixels, TransformationMatrix, point, size};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: rect(0.0, 0.0, 1000.0, 1000.0),
            corner_radii: Corners::default(),
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>) -> Quad {
        Quad {
            order: 0,
            border_style: BorderStyle::default(),
            bounds,
            content_mask: mask(),
            background: Background::default(),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transformation: TransformationMatrix::unit(),
        }
    }

    fn surface(bounds: Bounds<ScaledPixels>) -> PaintSurface {
        PaintSurface {
            order: 0,
            bounds,
            content_mask: mask(),
            #[cfg(target_os = "macos")]
            image_buffer: test_image_buffer(),
        }
    }

    /// The kinds in draw sequence, one entry per batch, with the batch length.
    fn batch_kinds(scene: &Scene) -> Vec<(PrimitiveKind, usize)> {
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Shadows(r) => (PrimitiveKind::Shadow, r.len()),
                PrimitiveBatch::Quads(r) => (PrimitiveKind::Quad, r.len()),
                PrimitiveBatch::Paths(r) => (PrimitiveKind::Path, r.len()),
                PrimitiveBatch::Underlines(r) => (PrimitiveKind::Underline, r.len()),
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    (PrimitiveKind::MonochromeSprite, range.len())
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    (PrimitiveKind::SubpixelSprite, range.len())
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    (PrimitiveKind::PolychromeSprite, range.len())
                }
                PrimitiveBatch::Surfaces(r) => (PrimitiveKind::Surface, r.len()),
                PrimitiveBatch::BackdropBlurs(r) => (PrimitiveKind::BackdropBlur, r.len()),
                PrimitiveBatch::StartLayerMask(_) => (PrimitiveKind::StartLayerMask, 1),
                PrimitiveBatch::EndLayerMask => (PrimitiveKind::EndLayerMask, 1),
            })
            .collect()
    }

    /// The wall this patch removes: inside a layer every primitive shared the
    /// layer's order, so a quad painted after a surface drew under it because
    /// equal orders batch by kind and surfaces batch after quads.
    #[test]
    fn a_quad_painted_over_a_surface_inside_a_layer_draws_above_it() {
        let mut scene = Scene::default();
        let layer = rect(0.0, 0.0, 200.0, 200.0);
        scene.push_layer(layer);
        scene.insert_primitive(surface(layer));
        scene.insert_primitive(quad(rect(10.0, 10.0, 50.0, 50.0)));
        scene.pop_layer();
        scene.finish();

        assert_eq!(
            batch_kinds(&scene),
            vec![(PrimitiveKind::Surface, 1), (PrimitiveKind::Quad, 1)]
        );
    }

    /// Non-overlapping primitives inside a layer still share an order, so a
    /// list of rows costs one batch per kind, not one per row.
    #[test]
    fn non_overlapping_rows_in_a_layer_batch_by_kind() {
        let mut scene = Scene::default();
        scene.push_layer(rect(0.0, 0.0, 200.0, 400.0));
        for row in 0..10 {
            let y = row as f32 * 40.0;
            scene.insert_primitive(quad(rect(0.0, y, 200.0, 40.0)));
            scene.insert_primitive(surface(rect(4.0, y + 4.0, 32.0, 32.0)));
        }
        scene.pop_layer();
        scene.finish();

        assert_eq!(
            batch_kinds(&scene),
            vec![(PrimitiveKind::Quad, 10), (PrimitiveKind::Surface, 10)]
        );
    }

    /// A sibling painted after a layer and overlapping it draws above the
    /// layer's contents. Shifting in-layer orders by a global sequence broke
    /// this: a layer's children outran the orders of everything painted later.
    #[test]
    fn a_sibling_painted_after_a_layer_draws_above_the_layer_contents() {
        let mut scene = Scene::default();
        for i in 0..8 {
            scene.insert_primitive(quad(rect(0.0, 0.0, 100.0 + i as f32, 100.0)));
        }
        scene.push_layer(rect(0.0, 0.0, 200.0, 200.0));
        scene.insert_primitive(quad(rect(0.0, 0.0, 200.0, 200.0)));
        scene.insert_primitive(quad(rect(0.0, 0.0, 150.0, 150.0)));
        scene.insert_primitive(quad(rect(0.0, 0.0, 120.0, 120.0)));
        scene.pop_layer();
        scene.insert_primitive(surface(rect(20.0, 20.0, 60.0, 60.0)));
        scene.finish();

        let layer_max = scene.quads.iter().map(|q| q.order).max().unwrap();
        let tooltip = scene.surfaces[0].order;
        assert!(
            tooltip > layer_max,
            "tooltip order {tooltip} must exceed every layer quad order {layer_max}"
        );
    }

    /// An explicit z-index reorders within its scope regardless of kind or
    /// paint sequence: the surface at z 2 draws above the quad at z 1 that was
    /// painted after it.
    #[test]
    fn a_higher_z_index_draws_above_a_lower_one_whatever_the_kind_or_sequence() {
        let mut scene = Scene::default();
        let bounds = rect(0.0, 0.0, 100.0, 100.0);
        scene.push_z_index(2);
        scene.insert_primitive(surface(bounds));
        scene.pop_z_index();
        scene.push_z_index(1);
        scene.insert_primitive(quad(bounds));
        scene.pop_z_index();
        scene.insert_primitive(quad(rect(0.0, 0.0, 10.0, 10.0)));
        scene.finish();

        assert_eq!(
            batch_kinds(&scene),
            vec![(PrimitiveKind::Quad, 2), (PrimitiveKind::Surface, 1)]
        );
        assert!(
            scene
                .quads
                .iter()
                .all(|q| q.order < scene.surfaces[0].order)
        );
    }

    /// A negative z-index sinks below the scope's unindexed primitives, even
    /// when painted last.
    #[test]
    fn a_negative_z_index_sinks_below_the_unindexed_primitives_of_its_scope() {
        let mut scene = Scene::default();
        let bounds = rect(0.0, 0.0, 100.0, 100.0);
        scene.insert_primitive(surface(bounds));
        scene.push_z_index(-1);
        scene.insert_primitive(quad(bounds));
        scene.pop_z_index();
        scene.finish();

        assert_eq!(
            batch_kinds(&scene),
            vec![(PrimitiveKind::Quad, 1), (PrimitiveKind::Surface, 1)]
        );
    }

    /// A z-index inside a layer is scoped to that layer: it cannot lift the
    /// layer's contents above a sibling painted after the layer.
    #[test]
    fn a_z_index_inside_a_layer_does_not_escape_the_layer() {
        let mut scene = Scene::default();
        let bounds = rect(0.0, 0.0, 100.0, 100.0);
        scene.push_layer(bounds);
        scene.push_z_index(1_000);
        scene.insert_primitive(surface(bounds));
        scene.pop_z_index();
        scene.pop_layer();
        scene.insert_primitive(quad(bounds));
        scene.finish();

        assert_eq!(
            batch_kinds(&scene),
            vec![(PrimitiveKind::Surface, 1), (PrimitiveKind::Quad, 1)]
        );
    }

    /// Replaying a cached subtree derives the same orders as painting it, so a
    /// reused view and a re-rendered view sort identically.
    #[test]
    fn replay_reproduces_the_orders_of_the_original_paint() {
        let mut first = Scene::default();
        first.push_layer(rect(0.0, 0.0, 200.0, 200.0));
        first.insert_primitive(surface(rect(0.0, 0.0, 200.0, 200.0)));
        first.push_z_index(3);
        first.insert_primitive(quad(rect(10.0, 10.0, 50.0, 50.0)));
        first.pop_z_index();
        first.pop_layer();
        first.insert_primitive(quad(rect(0.0, 0.0, 20.0, 20.0)));
        let operations = first.len();

        let mut second = Scene::default();
        second.replay(0..operations, &first);

        first.finish();
        second.finish();
        let orders = |scene: &Scene| {
            (
                scene.quads.iter().map(|q| q.order).collect::<Vec<_>>(),
                scene.surfaces.iter().map(|s| s.order).collect::<Vec<_>>(),
            )
        };
        assert_eq!(orders(&first), orders(&second));
        assert_eq!(
            batch_kinds(&second),
            vec![(PrimitiveKind::Surface, 1), (PrimitiveKind::Quad, 2)]
        );
        assert_eq!(orders(&second).0, vec![1, 2]);
    }

    #[test]
    fn content_mask_intersect_preserves_and_clamps_corner_radii() {
        let outer = ContentMask {
            bounds: rect(0.0, 0.0, 100.0, 100.0),
            corner_radii: Corners::default(),
        };
        let inner = ContentMask {
            bounds: rect(10.0, 10.0, 20.0, 20.0),
            corner_radii: Corners {
                top_left: ScaledPixels(15.0),
                top_right: ScaledPixels(15.0),
                bottom_right: ScaledPixels(15.0),
                bottom_left: ScaledPixels(15.0),
            },
        };
        let intersected = outer.intersect(&inner);
        assert_eq!(intersected.bounds, rect(10.0, 10.0, 20.0, 20.0));
        // Max radius for a 20x20 rect is 10.0
        assert_eq!(intersected.corner_radii.top_left, ScaledPixels(10.0));
        assert_eq!(intersected.corner_radii.top_right, ScaledPixels(10.0));
        assert_eq!(intersected.corner_radii.bottom_right, ScaledPixels(10.0));
        assert_eq!(intersected.corner_radii.bottom_left, ScaledPixels(10.0));
    }

    #[test]
    fn path_clip_batches_produce_start_and_end_layer_mask_markers() {
        let mut scene = Scene::default();
        let path = Path::new(Point::default()).scale(1.0);
        scene.push_layer_mask(LayerMask::Path(path));
        scene.insert_primitive(quad(rect(10.0, 10.0, 50.0, 50.0)));
        scene.pop_layer_mask();
        scene.finish();

        let batches = scene.batches().collect::<Vec<_>>();
        assert_eq!(batches.len(), 3, "batches: {:?}", batches);
        match (&batches[0], &batches[1], &batches[2]) {
            (
                PrimitiveBatch::StartLayerMask(LayerMask::Path(_)),
                PrimitiveBatch::Quads(range),
                PrimitiveBatch::EndLayerMask,
            ) => {
                assert_eq!(range.len(), 1);
            }
            other => panic!("unexpected batches: {:?}", other),
        }
    }

    /// A renderer releases a feature's textures once the feature has been
    /// unused for `LAYER_IDLE_RELEASE_FRAMES` consecutive frames, and any use
    /// restarts the count. Catches an off-by-one threshold, a count that
    /// ignores use, and a counter that stops reporting after the first release.
    #[test]
    fn layer_idle_counter_releases_after_consecutive_unused_frames() {
        let mut counter = LayerIdleCounter::default();
        for _ in 1..LAYER_IDLE_RELEASE_FRAMES {
            assert!(!counter.tick(false));
        }
        assert!(!counter.tick(true));
        for _ in 1..LAYER_IDLE_RELEASE_FRAMES {
            assert!(!counter.tick(false));
        }
        assert!(counter.tick(false));
        assert!(counter.tick(false));
    }
}
