//! Layout transitions: an element whose layout position changes moves from
//! where it was painted to its new position on a spring.

use std::cell::RefCell;

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement,
    LayoutId, ParentElement, Pixels, Point, Size, Window,
    motion::{ElementSpring, PIXEL_REST_DISTANCE, SpringConfig, presets},
    point, px,
};

#[cfg(test)]
mod tests;

/// The spring of a [`LayoutTransition`] unless [`LayoutTransition::spring`]
/// sets another: [`presets::SETTLE_SPRING`], stiffness 280, damping 30, unit
/// mass.
pub const LAYOUT_SPRING: SpringConfig = presets::SETTLE_SPRING;

/// Wraps `element` in a [`LayoutTransition`] with the id `id`.
pub fn layout_transition<E: IntoElement>(
    id: impl Into<ElementId>,
    element: E,
) -> LayoutTransition<E> {
    LayoutTransition {
        id: id.into(),
        element: Some(element),
        spring: LAYOUT_SPRING,
    }
}

/// An element that moves to a new layout position on a spring instead of
/// jumping to it.
///
/// Each frame compares the layout position of the wrapped element with its
/// position in the previous frame. When the position changed, the element is
/// painted where it was, displaced from its new position, and one spring per
/// axis carries the displacement to zero. A change during the motion starts
/// from the displacement and velocity of that instant, so the element never
/// jumps and keeps its speed.
///
/// The element moves by its element offset: its hitboxes move with it, and
/// every descendant, text included, is laid out once at its final position and
/// size and translated whole. The size changes at once; text is never scaled
/// or reflowed by the transition. Offsets snap to device pixels.
///
/// The position is measured in the layout tree the element is laid out in,
/// so scrolling an ancestor does not move it. Nested in another layout
/// transition of the same tree, it measures its position from that
/// transition, so an element that moves with its parent does not move twice.
/// A resize of the window moves every element to its new position at once.
///
/// While moving, the element requests animation frames that repaint the
/// region it painted in the previous frame and in this one, and no other part
/// of the window. At rest it requests nothing. Under reduced motion
/// ([`App::motion_policy`]) the element is at its new position in the frame
/// its layout changes.
pub struct LayoutTransition<E> {
    id: ElementId,
    element: Option<E>,
    spring: SpringConfig,
}

impl<E> LayoutTransition<E> {
    /// Sets the spring that carries each axis. Takes effect at the next
    /// change of position.
    pub fn spring(mut self, spring: SpringConfig) -> Self {
        self.spring = spring;
        self
    }

    /// Applies `f` to the wrapped element.
    pub fn map_element(mut self, f: impl FnOnce(E) -> E) -> Self {
        self.element = self.element.map(f);
        self
    }
}

impl<E: ParentElement> ParentElement for LayoutTransition<E> {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        if let Some(element) = &mut self.element {
            element.extend(elements);
        }
    }
}

impl<E: IntoElement + 'static> IntoElement for LayoutTransition<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The state a [`LayoutTransition`] keeps between frames.
struct LayoutTransitionState {
    /// The layout position of the last frame, measured as
    /// [`measure_position`] describes.
    position: Point<Pixels>,
    /// The window viewport of the last frame.
    viewport: Size<Pixels>,
    /// The displacement from the layout position, per axis, in pixels.
    x: ElementSpring,
    y: ElementSpring,
}

thread_local! {
    /// The layout transitions whose descendants are being prepainted,
    /// innermost last: the layout node of each and its position in its
    /// layout tree.
    static ENCLOSING: RefCell<Vec<(LayoutId, Point<Pixels>)>> = const { RefCell::new(Vec::new()) };
}

/// Pops the innermost entry of [`ENCLOSING`] when dropped.
struct EnclosingEntry;

impl EnclosingEntry {
    fn push(layout_id: LayoutId, origin: Point<Pixels>) -> Self {
        ENCLOSING.with(|enclosing| enclosing.borrow_mut().push((layout_id, origin)));
        Self
    }
}

impl Drop for EnclosingEntry {
    fn drop(&mut self) {
        ENCLOSING.with(|enclosing| enclosing.borrow_mut().pop());
    }
}

impl<E: IntoElement + 'static> Element for LayoutTransition<E> {
    type RequestLayoutState = (AnyElement, LayoutId);
    /// Whether the painted region changes between this frame and the next,
    /// or changed since the last.
    type PrepaintState = bool;

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
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
        let mut element = self
            .element
            .take()
            .expect("request_layout is called once per frame")
            .into_any_element();
        let layout_id = element.request_layout(window, cx);
        (layout_id, (element, layout_id))
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        (element, layout_id): &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let now = cx.frame_instant();
        let policy = cx.motion_policy();
        let viewport = window.viewport_size();
        let origin_in_tree = bounds.origin - window.pixel_snap_point(window.element_offset());
        let position = measure_position(*layout_id, origin_in_tree, window);
        let spring = self.spring;
        let global_id = global_id.expect("a layout transition has an id");

        let (offset, moving, declare_damage) =
            window.with_element_state(global_id, |state: Option<LayoutTransitionState>, _| {
                let mut state = state.unwrap_or_else(|| LayoutTransitionState {
                    position,
                    viewport,
                    x: ElementSpring::at_rest(0.0, spring, PIXEL_REST_DISTANCE),
                    y: ElementSpring::at_rest(0.0, spring, PIXEL_REST_DISTANCE),
                });
                state.x.set_config(spring);
                state.y.set_config(spring);
                if state.viewport != viewport {
                    state.x.snap(0.0);
                    state.y.snap(0.0);
                } else if state.position != position {
                    let displacement = state.position - position;
                    state.x.displace(displacement.x.0, policy, now);
                    state.y.displace(displacement.y.0, policy, now);
                }
                state.position = position;
                state.viewport = viewport;

                let was_moving = state.x.is_moving() || state.y.is_moving();
                let moving_x = state.x.update(policy, now);
                let moving_y = state.y.update(policy, now);
                let moving = moving_x || moving_y;
                let offset = point(px(state.x.value()), px(state.y.value()));
                ((offset, moving, was_moving || moving), state)
            });

        if moving {
            window.request_animation_frame_at_paint();
        }
        let _enclosing = EnclosingEntry::push(*layout_id, origin_in_tree);
        window.with_element_offset(offset, |window| element.prepaint(window, cx));
        declare_damage
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (element, _): &mut Self::RequestLayoutState,
        declare_damage: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let scene_index = window.scene_index();
        element.paint(window, cx);
        if *declare_damage {
            window.declare_painted_damage(scene_index);
        }
    }
}

/// The position of the element laid out as `layout_id` at `origin_in_tree`
/// in its layout tree: relative to the innermost enclosing layout transition
/// of the same tree, or to the root of the tree when there is none.
fn measure_position(
    layout_id: LayoutId,
    origin_in_tree: Point<Pixels>,
    window: &Window,
) -> Point<Pixels> {
    let reference = ENCLOSING.with(|enclosing| {
        enclosing
            .borrow()
            .last()
            .filter(|(ancestor, _)| window.layout_contains(*ancestor, layout_id))
            .map(|(_, origin)| *origin)
    });
    origin_in_tree - reference.unwrap_or_default()
}
