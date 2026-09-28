//! Height reveals: an element that grows from no height to the natural height
//! of its child, and shrinks back, on a spring.

use std::mem;

use crate::{
    AnyElement, App, Bounds, ContentMask, Element, ElementId, GlobalElementId, InspectorElementId,
    IntoElement, LAYOUT_SPRING, LayoutId, Pixels, Size, Style, Window,
    motion::{ElementSpring, PIXEL_REST_DISTANCE, SpringConfig},
    px,
};

#[cfg(test)]
mod tests;

/// Wraps `child` in a [`HeightReveal`] with the id `id`: at the natural
/// height of `child` when `expanded`, at no height otherwise.
pub fn height_reveal<E: IntoElement>(
    id: impl Into<ElementId>,
    expanded: bool,
    child: E,
) -> HeightReveal<E> {
    HeightReveal {
        id: id.into(),
        expanded,
        child: Some(child),
        spring: LAYOUT_SPRING,
    }
}

/// An element whose height moves on a spring between zero and the natural
/// height of its child.
///
/// The child is laid out at its natural height in every frame, whatever the
/// height of the reveal, so its content is never scaled or reflowed; the
/// reveal clips it while the height moves. Everything laid out after the
/// reveal moves with its height. Collapsed and at rest, the reveal has no
/// height, and the child is laid out but not prepainted or painted: it paints
/// nothing and has no hitbox, and the element state of its descendants is
/// dropped.
///
/// A change of `expanded`, or of the natural height of the child while
/// expanded, carries the height from its value and velocity at that instant
/// to the new target, so an interrupted motion reverses without a jump in
/// position or speed. The first frame of a reveal, and a resize of the
/// window, place it at its target at once.
///
/// A moving height moves the layout of what follows it, so each frame of the
/// motion repaints the whole window. At rest the reveal requests no frame.
/// Under reduced motion ([`App::motion_policy`]) the height is at its target
/// in the frame of the change.
///
/// ```ignore
/// height_reveal(("details", item.id), item.open, details(item))
/// ```
pub struct HeightReveal<E> {
    id: ElementId,
    expanded: bool,
    child: Option<E>,
    spring: SpringConfig,
}

impl<E> HeightReveal<E> {
    /// Sets the spring that carries the height, [`LAYOUT_SPRING`] unless set.
    /// Takes effect at the next change of target.
    pub fn spring(mut self, spring: SpringConfig) -> Self {
        self.spring = spring;
        self
    }
}

impl<E: IntoElement + 'static> IntoElement for HeightReveal<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The state a [`HeightReveal`] keeps between frames.
struct HeightRevealState {
    /// The natural height of the child in the last frame.
    natural: f32,
    /// The height the spring settles on: the natural height of the child when
    /// expanded, zero when collapsed.
    target: f32,
    /// The height minus [`Self::target`]. Its target is zero.
    offset: ElementSpring,
    /// The window viewport of the last frame.
    viewport: Size<Pixels>,
}

impl<E: IntoElement + 'static> Element for HeightReveal<E> {
    /// The child, the layout node that holds it at its natural height, and
    /// the height of the reveal: `None` when it is laid out at the natural
    /// height of the child.
    type RequestLayoutState = (AnyElement, LayoutId, Option<Pixels>);
    /// `None` when the child is not painted, otherwise the mask it is
    /// clipped to while the height moves.
    type PrepaintState = Option<Option<ContentMask<Pixels>>>;

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
        let now = cx.frame_instant();
        let policy = cx.motion_policy();
        let viewport = window.viewport_size();
        let (expanded, spring) = (self.expanded, self.spring);
        let global_id = global_id.expect("a height reveal has an id");

        let height = window.with_element_state(global_id, |state: Option<HeightRevealState>, _| {
            let fresh = state.is_none();
            let mut state = state.unwrap_or_else(|| HeightRevealState {
                natural: 0.0,
                target: 0.0,
                offset: ElementSpring::at_rest(0.0, spring, PIXEL_REST_DISTANCE),
                viewport,
            });
            state.offset.set_config(spring);
            let resized = mem::replace(&mut state.viewport, viewport) != viewport;
            if expanded && (fresh || resized || policy.reduced()) {
                // Laid out at the natural height of this frame; prepaint
                // records it as the target.
                state.offset.snap(0.0);
                return (None, state);
            }
            let target = if expanded { state.natural } else { 0.0 };
            if resized {
                state.offset.snap(0.0);
            } else if target != state.target {
                state.offset.displace(state.target - target, policy, now);
            }
            state.target = target;
            state.offset.update(policy, now);
            let height = (target + state.offset.value()).max(0.0);
            (Some(px(height)), state)
        });

        let mut child = self
            .child
            .take()
            .expect("request_layout is called once per frame")
            .into_any_element();
        let child_id = child.request_layout(window, cx);
        // Auto height, so the child's height never resolves against the
        // height of the reveal.
        let content_id = window.request_layout(Style::default(), [child_id], cx);
        let mut style = Style::default();
        if let Some(height) = height {
            style.size.height = height.into();
        }
        let layout_id = window.request_layout(style, [content_id], cx);
        (layout_id, (child, content_id, height))
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        (child, content_id, height): &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let now = cx.frame_instant();
        let policy = cx.motion_policy();
        let natural = window.layout_bounds(*content_id).size.height.0;
        let expanded = self.expanded;
        let global_id = global_id.expect("a height reveal has an id");

        let moving = window.with_element_state(global_id, |state: Option<HeightRevealState>, _| {
            let mut state = state.expect("request_layout stores the state of a height reveal");
            if height.is_none() {
                state.target = natural;
            } else if expanded && natural != state.target {
                // The height laid out this frame stays; the spring carries it
                // to the new natural height.
                state.offset.displace(state.target - natural, policy, now);
                state.target = natural;
            }
            state.natural = natural;
            (state.offset.is_moving(), state)
        });

        if moving {
            window.request_animation_frame();
        }
        if height.is_some_and(|height| height <= px(0.0)) {
            return None;
        }
        let mask = moving.then(|| ContentMask::from(bounds));
        window.with_content_mask(mask, |window| child.prepaint(window, cx));
        Some(mask)
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (child, _, _): &mut Self::RequestLayoutState,
        mask: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if let Some(mask) = *mask {
            window.with_content_mask(mask, |window| child.paint(window, cx));
        }
    }
}
