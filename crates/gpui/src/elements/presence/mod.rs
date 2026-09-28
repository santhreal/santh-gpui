//! Presence: keyed children that fade in when they appear and keep painting
//! while they fade out after they disappear.

#[cfg(test)]
mod tests;

use std::{mem, rc::Rc};

use collections::FxHashMap;

use crate::{
    AnyElement, App, Bounds, Div, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LayoutId, ParentElement as _, Pixels, Point, StyleRefinement, Styled,
    Transformation, TransformationMatrix, Window, div,
    motion::{
        ElementSpring, FrameInstant, MotionModel, MotionPolicy, SpringConfig, UNIT_REST_DISTANCE,
        presets,
    },
    point, px,
};

/// The motion of a [`Presence`] unless [`Presence::motion`] sets another:
/// [`presets::SETTLE_SPRING`], stiffness 280, damping 30, unit mass.
pub const PRESENCE_SPRING: SpringConfig = presets::SETTLE_SPRING;

/// How a child of a [`Presence`] differs from its resting appearance while
/// it is absent. An entering child moves from this appearance to rest, and an
/// exiting child moves from rest to this appearance, both at opacity 0 when
/// absent and 1 at rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresenceEffect {
    /// Offset of an absent child from its layout position. Hitboxes move
    /// with the offset.
    pub offset: Point<Pixels>,
    /// Scale of an absent child about its center, painted only: layout and
    /// hitboxes keep the unscaled size.
    pub scale: f32,
}

impl PresenceEffect {
    /// Opacity only.
    pub const FADE: Self = Self {
        offset: Point {
            x: px(0.0),
            y: px(0.0),
        },
        scale: 1.0,
    };

    /// Opacity and a rise of 6 px: an entering child starts 6 px below its
    /// place. The default.
    pub const RISE: Self = Self {
        offset: Point {
            x: px(0.0),
            y: px(6.0),
        },
        scale: 1.0,
    };
}

impl Default for PresenceEffect {
    fn default() -> Self {
        Self::RISE
    }
}

/// Builds the element of one child of a [`Presence`], each frame the child
/// is painted.
type ChildBuilder = Rc<dyn Fn(&mut Window, &mut App) -> AnyElement>;

/// Creates an empty [`Presence`] with the id `id`.
pub fn presence(id: impl Into<ElementId>) -> Presence {
    Presence {
        id: id.into(),
        container: Some(div()),
        children: Vec::new(),
        effect: PresenceEffect::RISE,
        model: MotionModel::Spring(PRESENCE_SPRING),
    }
}

/// A container of keyed children that animates each child in when its key
/// appears and out when its key disappears.
///
/// Each frame the caller supplies the current children as pairs of a key
/// and a builder. A key absent from the previous frame enters: its child
/// moves from the [`PresenceEffect`] to rest, opacity 0 to 1, starting 6 px
/// below its place by default. A key the caller stops supplying exits: the
/// container keeps the last builder supplied for that key and builds the
/// child with it each frame while the child moves back to the effect,
/// opacity 1 to 0, then drops it. An exiting child keeps its place in the
/// layout, after the child that preceded it, and its hitboxes until it is
/// dropped. A key supplied again before its exit ends reverses from the
/// value and velocity of that instant.
///
/// Siblings move to their new places in the frame an exiting child is
/// dropped. A sibling wrapped in a [`layout_transition`](crate::layout_transition)
/// moves on a spring from where it was painted instead.
///
/// Children present when the container first paints appear at rest. Keys
/// are unique among the children of one frame; a child's element state is
/// keyed by its key, so it survives reordering.
///
/// While a child moves, the container requests animation frames that
/// repaint the region the child painted in the previous frame and in this
/// one, and the whole window in a frame whose set or order of children
/// changes. At rest it requests nothing. Under reduced motion
/// ([`App::motion_policy`]) a child enters and is dropped in the frame of
/// the change.
///
/// The container is a [`Div`] and takes its styles through [`Styled`].
///
/// ```ignore
/// presence("notes")
///     .flex_col()
///     .children(notes.iter().map(|note| {
///         let text = note.text.clone();
///         (note.id, move |_: &mut Window, _: &mut App| div().child(text.clone()))
///     }))
/// ```
pub struct Presence {
    id: ElementId,
    container: Option<Div>,
    children: Vec<(ElementId, ChildBuilder)>,
    effect: PresenceEffect,
    model: MotionModel,
}

impl Presence {
    /// Adds the child keyed `key`, built by `build` each frame it is painted.
    /// After the caller stops supplying `key`, the last `build` supplied
    /// builds the child until its exit ends.
    pub fn child<E: IntoElement>(
        mut self,
        key: impl Into<ElementId>,
        build: impl Fn(&mut Window, &mut App) -> E + 'static,
    ) -> Self {
        let build: ChildBuilder = Rc::new(move |window: &mut Window, cx: &mut App| {
            build(window, cx).into_any_element()
        });
        self.children.push((key.into(), build));
        self
    }

    /// Adds each `(key, build)` pair as [`Self::child`] does.
    pub fn children<K, F, E>(mut self, children: impl IntoIterator<Item = (K, F)>) -> Self
    where
        K: Into<ElementId>,
        F: Fn(&mut Window, &mut App) -> E + 'static,
        E: IntoElement,
    {
        for (key, build) in children {
            self = self.child(key, build);
        }
        self
    }

    /// Sets how an absent child differs from its resting appearance.
    pub fn effect(mut self, effect: PresenceEffect) -> Self {
        self.effect = effect;
        self
    }

    /// Sets the model that moves children in and out: a spring, which carries
    /// the velocity of a reversal, or a duration, which restarts its curve
    /// from the value of the reversal. Takes effect at the next change.
    pub fn motion(mut self, model: MotionModel) -> Self {
        self.model = model;
        self
    }
}

impl Styled for Presence {
    fn style(&mut self) -> &mut StyleRefinement {
        self.container
            .as_mut()
            .expect("a presence is styled before it is laid out")
            .style()
    }
}

impl IntoElement for Presence {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The children a [`Presence`] keeps between frames, in paint order.
#[derive(Default)]
struct PresenceState {
    entries: Vec<PresenceEntry>,
}

struct PresenceEntry {
    key: ElementId,
    /// The builder supplied for the key in the last frame it was supplied.
    build: ChildBuilder,
    /// 1 at rest and present, 0 absent.
    progress: ElementSpring,
    /// Whether the key was supplied in the current frame.
    present: bool,
    /// Whether the progress moved this frame or starts moving: the child
    /// declares its painted region as damage.
    damage: bool,
}

impl PresenceEntry {
    /// Sets whether the key is supplied; a change moves the progress toward
    /// 1 or 0 from its value and velocity at `now`.
    fn set_present(
        &mut self,
        present: bool,
        model: MotionModel,
        policy: MotionPolicy,
        now: FrameInstant,
    ) {
        self.progress.set_model(model);
        if self.present != present {
            self.present = present;
            let target = if present { 1.0 } else { 0.0 };
            self.progress.set_target(target, policy, now);
        }
    }
}

impl PresenceState {
    /// Brings the entries to the keys of `children`, in their order, and
    /// returns whether the set or order of entries changed. An entry whose
    /// key is gone stays after the entry that preceded it. On the first
    /// frame, entries start at rest.
    fn reconcile(
        &mut self,
        children: Vec<(ElementId, ChildBuilder)>,
        first: bool,
        model: MotionModel,
        policy: MotionPolicy,
        now: FrameInstant,
    ) -> bool {
        if self.entries.len() == children.len()
            && self
                .entries
                .iter()
                .zip(&children)
                .all(|(entry, (key, _))| entry.present && entry.key == *key)
        {
            for (entry, (_, build)) in self.entries.iter_mut().zip(children) {
                entry.build = build;
                entry.progress.set_model(model);
            }
            return false;
        }

        let old = mem::take(&mut self.entries);
        let mut kept: Vec<Option<(usize, PresenceEntry)>> = Vec::new();
        kept.resize_with(children.len(), || None);
        // Each gone entry, the position among `children` it precedes, and
        // its old position.
        let mut gone = Vec::new();
        {
            let index: FxHashMap<&ElementId, usize> = children
                .iter()
                .enumerate()
                .map(|(position, (key, _))| (key, position))
                .collect();
            let mut slot = 0;
            for (old_position, entry) in old.into_iter().enumerate() {
                match index.get(&entry.key) {
                    Some(&position) if kept[position].is_none() => {
                        slot = position + 1;
                        kept[position] = Some((old_position, entry));
                    }
                    _ => gone.push((slot, old_position, entry)),
                }
            }
        }
        gone.sort_by_key(|(slot, _, _)| *slot);

        let mut changed = false;
        let mut entries = Vec::with_capacity(children.len() + gone.len());
        let mut place = |entries: &mut Vec<PresenceEntry>,
                         old_position: Option<usize>,
                         mut entry: PresenceEntry,
                         present: bool| {
            entry.set_present(present, model, policy, now);
            changed |= old_position != Some(entries.len());
            entries.push(entry);
        };
        let mut gone = gone.into_iter().peekable();
        for (position, ((key, build), kept)) in children.into_iter().zip(kept).enumerate() {
            while let Some((_, old_position, entry)) =
                gone.next_if(|(slot, _, _)| *slot <= position)
            {
                place(&mut entries, Some(old_position), entry, false);
            }
            let (old_position, entry) = match kept {
                Some((old_position, mut entry)) => {
                    entry.build = build;
                    (Some(old_position), entry)
                }
                None => {
                    let progress = if first { 1.0 } else { 0.0 };
                    let entry = PresenceEntry {
                        key,
                        build,
                        progress: ElementSpring::at_rest(
                            progress,
                            PRESENCE_SPRING,
                            UNIT_REST_DISTANCE,
                        ),
                        present: first,
                        damage: false,
                    };
                    (None, entry)
                }
            };
            place(&mut entries, old_position, entry, true);
        }
        for (_, old_position, entry) in gone {
            place(&mut entries, Some(old_position), entry, false);
        }
        self.entries = entries;
        changed
    }
}

impl Element for Presence {
    /// The container holding one item per child, and whether the set or
    /// order of children changed this frame.
    type RequestLayoutState = (AnyElement, bool);
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let global_id = global_id.expect("a presence has an id");
        let now = cx.frame_instant();
        let policy = cx.motion_policy();
        let children = mem::take(&mut self.children);
        let (effect, model) = (self.effect, self.model);

        let (items, moving, relayout) =
            window.with_element_state(global_id, |state: Option<PresenceState>, window| {
                let first = state.is_none();
                let mut state = state.unwrap_or_default();
                let mut relayout = state.reconcile(children, first, model, policy, now);
                let mut moving = false;
                state.entries.retain_mut(|entry| {
                    let was_moving = entry.progress.is_moving();
                    let entry_moving = entry.progress.update(policy, now);
                    entry.damage = was_moving || entry_moving;
                    moving |= entry_moving;
                    let keep = entry.present || entry_moving;
                    relayout |= !keep;
                    keep
                });
                let items: Vec<AnyElement> = state
                    .entries
                    .iter()
                    .map(|entry| {
                        PresenceItem {
                            key: entry.key.clone(),
                            element: (entry.build)(window, cx),
                            progress: entry.progress.value(),
                            effect,
                            damage: entry.damage,
                        }
                        .into_any_element()
                    })
                    .collect();
                ((items, moving, relayout), state)
            });

        if moving {
            window.request_animation_frame_at_paint();
        }
        let mut container = self
            .container
            .take()
            .expect("request_layout is called once per frame");
        container.extend(items);
        let mut container = container.into_any_element();
        let layout_id = container.request_layout(window, cx);
        (layout_id, (container, relayout))
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (container, _): &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        container.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (container, relayout): &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        container.paint(window, cx);
        if *relayout {
            window.declare_damage(Bounds::new(Point::default(), window.viewport_size()));
        }
    }
}

/// One child of a [`Presence`], painted at the appearance of its progress.
struct PresenceItem {
    key: ElementId,
    element: AnyElement,
    progress: f32,
    effect: PresenceEffect,
    damage: bool,
}

impl IntoElement for PresenceItem {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for PresenceItem {
    type RequestLayoutState = ();
    /// The offset the child is painted at.
    type PrepaintState = Point<Pixels>;

    fn id(&self) -> Option<ElementId> {
        Some(self.key.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (self.element.request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let absent = 1.0 - self.progress;
        let offset = point(
            self.effect.offset.x * absent,
            self.effect.offset.y * absent,
        );
        let element = &mut self.element;
        window.with_element_offset(offset, |window| element.prepaint(window, cx));
        offset
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        offset: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let opacity = (self.progress < 1.0).then_some(self.progress.max(0.0));
        let scale = 1.0 + (self.effect.scale - 1.0) * (1.0 - self.progress);
        let transformation = if (scale - 1.0).abs() <= f32::EPSILON {
            TransformationMatrix::unit()
        } else {
            Transformation::uniform_scale(scale)
                .into_matrix(bounds.center() + *offset, window.scale_factor())
        };
        let scene_index = window.scene_index();
        let element = &mut self.element;
        window.with_element_opacity(opacity, |window| {
            window.with_transformation(transformation, |window| element.paint(window, cx))
        });
        if self.damage {
            window.declare_painted_damage(scene_index);
        }
    }
}
