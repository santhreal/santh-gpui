//! The window manager states a window asks for before the window manager
//! has it: a window opened maximized or fullscreen.
//!
//! EWMH sets an unmapped window's states through its `_NET_WM_STATE`
//! property and a mapped window's through a `_NET_WM_STATE` client message,
//! and a window manager acts on no message about a window it has not taken.
//! Mutter replaces the property a client sets before the map with its own,
//! and acts on an add message sent once the window is mapped. A state asked
//! for before the map is therefore written to the property before the map
//! and sent as an add message on the window's first `MapNotify`. Adding a
//! state a window already has changes nothing, so the message leaves a
//! window manager that honoured the property where it was.

/// A state a window asks the window manager for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WmState {
    Maximized,
    Fullscreen,
}

/// The states a window asked for before its first `MapNotify`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct OpenState {
    mapped: bool,
    maximized: bool,
    fullscreen: bool,
}

impl OpenState {
    /// Toggles `state`. Returns true once the window has been mapped, the
    /// toggle then being the window manager's to apply; before that, the
    /// toggle is recorded for the map and nothing is sent.
    pub(super) fn toggle(&mut self, state: WmState) -> bool {
        if self.mapped {
            return true;
        }
        match state {
            WmState::Maximized => self.maximized = !self.maximized,
            WmState::Fullscreen => self.fullscreen = !self.fullscreen,
        }
        false
    }

    /// The states asked for, in the order the property lists them.
    pub(super) fn requested(self) -> impl Iterator<Item = WmState> {
        [
            (self.maximized, WmState::Maximized),
            (self.fullscreen, WmState::Fullscreen),
        ]
        .into_iter()
        .filter_map(|(asked, state)| asked.then_some(state))
    }

    /// The window's first `MapNotify`: returns the states asked for before
    /// it, to send as add messages. A later map returns `None`, and every
    /// later toggle is the window manager's to apply.
    pub(super) fn map(&mut self) -> Option<OpenState> {
        if self.mapped {
            return None;
        }
        let asked = *self;
        *self = OpenState {
            mapped: true,
            ..OpenState::default()
        };
        Some(asked)
    }
}

#[cfg(test)]
mod tests;
