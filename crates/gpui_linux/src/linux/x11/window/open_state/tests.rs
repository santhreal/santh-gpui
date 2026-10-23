//! WHY: a window opened with `WindowBounds::Maximized` or `Fullscreen` asked
//! for the state with a `_NET_WM_STATE` toggle message sent before the map,
//! which a window manager drops for a window it has not taken, so the
//! window opened plain at its saved size. These tests walk every state
//! through the record kept until the first `MapNotify`: each one asked for
//! before the map is listed for the property and returned once for the add
//! message, a pair of toggles cancels, and a toggle after the map is the
//! window manager's alone. Not covered: that the window writes the property
//! and sends the message, which needs an X server and a window manager.

use super::{OpenState, WmState};

/// Every state, in the order the property lists them. A new state fails to
/// compile here until it is listed.
fn every_state() -> [WmState; 2] {
    let listed = [WmState::Maximized, WmState::Fullscreen];
    for state in listed {
        match state {
            WmState::Maximized | WmState::Fullscreen => {}
        }
    }
    listed
}

/// No state.
const NONE: [WmState; 0] = [];

fn requested(open: OpenState) -> Vec<WmState> {
    open.requested().collect()
}

#[test]
fn a_state_asked_for_before_the_map_is_listed_and_sent_once_on_the_first_map() {
    for state in every_state() {
        let mut open = OpenState::default();
        assert!(!open.toggle(state), "{state:?} was sent before the map");
        assert_eq!(requested(open), [state], "{state:?} before the map");
        let asked = open.map().expect("the first map returns what was asked");
        assert_eq!(requested(asked), [state], "{state:?} on the first map");
        assert_eq!(requested(open), NONE, "{state:?} after the first map");
        assert_eq!(open.map(), None, "{state:?} on a second map");
    }
}

#[test]
fn a_state_toggled_twice_before_the_map_is_not_asked_for() {
    for state in every_state() {
        let mut open = OpenState::default();
        assert!(!open.toggle(state));
        assert!(!open.toggle(state));
        assert_eq!(requested(open), NONE, "{state:?}");
        assert_eq!(open.map().map(requested), Some(Vec::new()), "{state:?}");
    }
}

#[test]
fn every_state_asked_for_before_the_map_is_listed_in_property_order() {
    let mut open = OpenState::default();
    for state in every_state().into_iter().rev() {
        assert!(!open.toggle(state));
    }
    assert_eq!(requested(open), every_state());
    assert_eq!(open.map().map(requested), Some(every_state().to_vec()));
}

#[test]
fn a_toggle_after_the_first_map_is_the_window_managers_and_is_not_recorded() {
    for state in every_state() {
        let mut open = OpenState::default();
        assert_eq!(open.map().map(requested), Some(Vec::new()));
        assert!(open.toggle(state), "{state:?} after the map was not sent");
        assert_eq!(requested(open), NONE, "{state:?}");
        assert_eq!(open.map(), None, "{state:?}");
    }
}
