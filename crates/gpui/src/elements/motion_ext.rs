use crate::{ElementId, IntoElement, LayoutTransition, layout_transition};

/// Motion of an element's layout, added to every element.
///
/// See [`LayoutTransition`].
pub trait MotionExt: IntoElement + Sized {
    /// Moves the element to each new layout position on a spring instead of
    /// jumping to it. See [`LayoutTransition`]; `id` keeps the motion across
    /// frames and must be unique among its siblings.
    fn animate_layout(self, id: impl Into<ElementId>) -> LayoutTransition<Self> {
        layout_transition(id, self)
    }
}

impl<E: IntoElement> MotionExt for E {}
