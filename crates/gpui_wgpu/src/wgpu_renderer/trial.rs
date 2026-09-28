//! A context whose adapter was selected without a surface can report a
//! window's surface as compatible and fail to configure it, as an adapter
//! of a host with more than one GPU can. A window renderer created on such
//! a context configures its surface on a worker and does not wait for it:
//! the first configure's result, read once it has returned, keeps the
//! context or rejects it, and `WgpuRenderer::recover` replaces a rejected
//! context with one whose adapter is selected by configuring the surface.

/// The trial of a window renderer's context by the renderer's first
/// configure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Trial {
    /// No configure result of the renderer rejects its context.
    Settled,
    /// The context had configured no surface as the renderer was created,
    /// and the renderer's first configure has not been read.
    Pending,
    /// The renderer's first configure failed on a context that has
    /// configured no surface.
    Rejected,
}

impl Trial {
    /// The trial of a renderer created on a context that has
    /// (`surface_tested`) or has not configured a window surface.
    /// `window_surface` is whether the renderer draws to one.
    pub(super) fn new(window_surface: bool, surface_tested: bool) -> Self {
        if window_surface && !surface_tested {
            Self::Pending
        } else {
            Self::Settled
        }
    }

    /// The trial once the first configure has returned, `failed` being
    /// whether it reported an error and `surface_tested` whether the
    /// renderer's context has configured a surface by now, another window's
    /// included. A failure on such a context is a failed frame: the
    /// context is replaced only if no configure on it has succeeded, so a
    /// renderer on a context selected by configuring its surface is never
    /// rejected.
    pub(super) fn settle(self, failed: bool, surface_tested: bool) -> Self {
        match self {
            Self::Pending if failed && !surface_tested => Self::Rejected,
            Self::Pending => Self::Settled,
            trial => trial,
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(all(test, target_os = "linux"))]
mod window_tests;
