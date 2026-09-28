//! Crossfade: a keyed child that fades into the child of the next key.

use super::{Presence, PresenceEffect, presence};
use crate::{
    App, ElementId, IntoElement, StyleRefinement, Styled, Window,
    motion::{MotionModel, SpringConfig},
};

/// Creates a [`Crossfade`] with the id `id` showing the child keyed `key`,
/// built by `child`.
pub fn crossfade<E: IntoElement>(
    id: impl Into<ElementId>,
    key: impl Into<ElementId>,
    child: impl Fn(&mut Window, &mut App) -> E + 'static,
) -> Crossfade {
    let mut presence = presence(id).effect(PresenceEffect::FADE).child(key, child);
    presence.stacked = true;
    Crossfade(presence.relative())
}

/// A container that shows one keyed child and fades between children when
/// the key changes.
///
/// Each frame the caller supplies the key of the current child and a builder
/// for it. When the key changes, the child of the new key fades in, opacity
/// 0 to 1, on [`PRESENCE_SPRING`](super::PRESENCE_SPRING), while the child
/// of the previous key fades
/// out on the same spring, painted beneath it at the top left corner of the
/// container and built with the last builder supplied for its key. The
/// container takes the size of the current child. The previous child drops
/// once its fade rests. A key that returns before its fade-out ends fades
/// back in from the opacity and velocity of that instant.
///
/// While fading, the container requests animation frames that repaint the
/// regions the children paint. At rest it requests nothing and paints only
/// the current child. Under reduced motion ([`App::motion_policy`]) the
/// children swap in the frame the key changes.
///
/// The container is a relatively positioned [`Div`](crate::Div) and takes
/// its styles through [`Styled`].
///
/// ```ignore
/// crossfade("status", status.key(), move |_: &mut Window, _: &mut App| {
///     div().child(status.label())
/// })
/// ```
pub struct Crossfade(Presence);

impl Crossfade {
    /// Sets the spring of the fades. Takes effect at the next change of key.
    pub fn spring(self, spring: SpringConfig) -> Self {
        Self(self.0.motion(MotionModel::Spring(spring)))
    }
}

impl Styled for Crossfade {
    fn style(&mut self) -> &mut StyleRefinement {
        self.0.style()
    }
}

impl IntoElement for Crossfade {
    type Element = Presence;

    fn into_element(self) -> Self::Element {
        self.0
    }
}
