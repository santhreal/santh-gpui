//! The GPU context the windows of one display share.

#[cfg(not(target_family = "wasm"))]
use std::cell::OnceCell;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

#[cfg(not(target_family = "wasm"))]
use crate::DisplayInstances;
use crate::WgpuContext;

/// The GPU context the windows of one display share, and the instances it
/// is created from. The first window to create a renderer creates the
/// context; device recovery replaces it, on the same instances.
#[derive(Clone, Default)]
pub struct GpuContext(Rc<Shared>);

#[derive(Default)]
struct Shared {
    #[cfg(not(target_family = "wasm"))]
    instances: OnceCell<DisplayInstances>,
    context: RefCell<Option<WgpuContext>>,
}

impl GpuContext {
    /// A GPU context whose instances are created with its first context.
    pub fn new() -> Self {
        Self::default()
    }

    /// A GPU context created from `instances`.
    #[cfg(not(target_family = "wasm"))]
    pub fn with_instances(instances: DisplayInstances) -> Self {
        Self(Rc::new(Shared {
            instances: OnceCell::from(instances),
            context: RefCell::default(),
        }))
    }

    pub fn borrow(&self) -> Ref<'_, Option<WgpuContext>> {
        self.0.context.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Option<WgpuContext>> {
        self.0.context.borrow_mut()
    }

    /// Sets the instances the context is created from. Fails if the GPU
    /// context has instances.
    #[cfg(not(target_family = "wasm"))]
    pub fn set_instances(&self, instances: DisplayInstances) -> anyhow::Result<()> {
        self.0
            .instances
            .set(instances)
            .map_err(|_| anyhow::anyhow!("The GPU context already has its instances"))
    }

    /// The instances the context is created from, created from the display
    /// of `window` at the first call on a GPU context that has none.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn instances<W>(&self, window: &W) -> &DisplayInstances
    where
        W: wgpu::wgt::WgpuHasDisplayHandle + Clone,
    {
        self.0
            .instances
            .get_or_init(|| DisplayInstances::new(window.clone()))
    }
}
