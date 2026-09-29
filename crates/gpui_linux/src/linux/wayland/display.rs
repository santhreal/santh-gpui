use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

use anyhow::Context as _;
use uuid::Uuid;

use gpui::{Bounds, DisplayId, Pixels, PlatformDisplay};

#[derive(Debug, Clone)]
pub(crate) struct WaylandDisplay {
    /// The wl_output's protocol-level id
    pub protocol_id: u32,
    pub name: Option<String>,
    pub bounds: Bounds<Pixels>,
}

impl Hash for WaylandDisplay {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.protocol_id.hash(state);
    }
}

impl PlatformDisplay for WaylandDisplay {
    fn id(&self) -> DisplayId {
        DisplayId::new(self.protocol_id as u64)
    }

    fn uuid(&self) -> anyhow::Result<Uuid> {
        let name = self
            .name
            .as_ref()
            .context("Wayland display does not have a name")?;
        Ok(Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes()))
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}
