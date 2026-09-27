//! The display handle of a Wayland connection.

use std::ptr::NonNull;

use raw_window_handle as rwh;
use wayland_client::Connection;

/// The `wl_display` of a connection, as a display handle that keeps the
/// connection open.
#[derive(Debug)]
pub(super) struct ConnectionDisplay(pub(super) Connection);

impl rwh::HasDisplayHandle for ConnectionDisplay {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        let display = NonNull::new(self.0.backend().display_ptr().cast())
            .ok_or(rwh::HandleError::Unavailable)?;
        let handle = rwh::WaylandDisplayHandle::new(display);
        // Safety: the display is valid while the connection is open, and
        // `self` holds the connection.
        Ok(unsafe { rwh::DisplayHandle::borrow_raw(handle.into()) })
    }
}
