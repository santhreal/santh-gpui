//! The Vulkan tier selects an adapter without creating a GL instance. That
//! is correct only when the Vulkan and GL tier would select the same
//! adapter: when the adapter ranks above every adapter the GL backend can
//! report. These tests fail if the Vulkan tier selects an adapter a GL
//! adapter outranks, which would select a Vulkan CPU or virtual adapter
//! over a GL GPU, or rejects one no GL adapter outranks, which would create
//! the GL instance where it cannot change the selection. They also fail if
//! the tier of a display whose GL instance is a last resort rejects a
//! Vulkan adapter, which would create the GL instance while a Vulkan CPU
//! adapter can present. They do not observe which instances a context
//! creates on a real host.

use super::{AdapterRank, BackendTier, parse_pci_id};
use crate::CompositorGpuHint;

const VENDOR: u32 = 0x1002;
const DEVICE: u32 = 0x744c;

/// Every device type. The match stops compiling when wgpu adds a type, so
/// a new type gets a rank and a tier decision before the tests build.
fn device_types() -> [wgpu::DeviceType; 5] {
    let all = [
        wgpu::DeviceType::Other,
        wgpu::DeviceType::IntegratedGpu,
        wgpu::DeviceType::DiscreteGpu,
        wgpu::DeviceType::VirtualGpu,
        wgpu::DeviceType::Cpu,
    ];
    for device_type in all {
        match device_type {
            wgpu::DeviceType::Other
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::VirtualGpu
            | wgpu::DeviceType::Cpu => {}
        }
    }
    all
}

/// The device types wgpu's GL backend infers from the renderer string.
const GL_DEVICE_TYPES: [wgpu::DeviceType; 3] = [
    wgpu::DeviceType::IntegratedGpu,
    wgpu::DeviceType::Other,
    wgpu::DeviceType::Cpu,
];

fn info(backend: wgpu::Backend, device_type: wgpu::DeviceType, device: u32) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: String::new(),
        vendor: VENDOR,
        device,
        device_type,
        device_pci_bus_id: String::new(),
        driver: String::new(),
        driver_info: String::new(),
        backend,
        subgroup_min_size: 32,
        subgroup_max_size: 64,
        transient_saves_memory: false,
    }
}

/// `ZED_DEVICE_ID` values: unset, the Vulkan device, another device, and
/// 0, the device id every GL adapter reports.
const FILTERS: [Option<u32>; 4] = [None, Some(DEVICE), Some(0x1234), Some(0)];

fn hints() -> [Option<CompositorGpuHint>; 4] {
    let hint = |vendor_id, device_id| {
        Some(CompositorGpuHint {
            vendor_id,
            device_id,
        })
    };
    [
        None,
        hint(VENDOR, DEVICE),
        hint(VENDOR, 0x1234),
        hint(VENDOR, 0),
    ]
}

#[test]
fn vulkan_tier_selects_exactly_the_adapters_above_every_gl_adapter() {
    for device_type in device_types() {
        for filter in FILTERS {
            for hint in hints() {
                let vulkan = AdapterRank::new(
                    &info(wgpu::Backend::Vulkan, device_type, DEVICE),
                    filter,
                    hint.as_ref(),
                );
                let above_every_gl = GL_DEVICE_TYPES.iter().all(|&gl_type| {
                    vulkan
                        < AdapterRank::new(
                            &info(wgpu::Backend::Gl, gl_type, 0),
                            filter,
                            hint.as_ref(),
                        )
                });
                assert_eq!(
                    BackendTier::Vulkan.selects(vulkan),
                    above_every_gl,
                    "{device_type:?} with filter {filter:?} and hint {:?}",
                    hint.map(|h| (h.vendor_id, h.device_id)),
                );
                assert!(BackendTier::VulkanAndGl.selects(vulkan));
                assert!(
                    BackendTier::EveryVulkan.selects(vulkan),
                    "{device_type:?} with filter {filter:?}"
                );
            }
        }
    }
}

#[test]
fn vulkan_tier_selects_hardware_adapters_and_the_named_device() {
    for device_type in device_types() {
        let hardware = match device_type {
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => true,
            wgpu::DeviceType::Other | wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Cpu => false,
        };
        let adapter = info(wgpu::Backend::Vulkan, device_type, DEVICE);
        let named = [
            (None, None, hardware),
            (Some(DEVICE), None, true),
            (None, hints()[1], true),
            (Some(0x1234), hints()[2], hardware),
        ];
        for (filter, hint, selected) in named {
            let rank = AdapterRank::new(&adapter, filter, hint.as_ref());
            assert_eq!(
                BackendTier::Vulkan.selects(rank),
                selected,
                "{device_type:?} with filter {filter:?}",
            );
        }
    }
}

#[test]
fn test_parse_device_id() {
    assert!(parse_pci_id("0xABCD").is_ok());
    assert!(parse_pci_id("ABCD").is_ok());
    assert!(parse_pci_id("abcd").is_ok());
    assert!(parse_pci_id("1234").is_ok());
    assert!(parse_pci_id("123").is_err());
    assert_eq!(
        parse_pci_id(&format!("{:x}", 0x1234)).unwrap(),
        parse_pci_id(&format!("{:X}", 0x1234)).unwrap(),
    );

    assert_eq!(
        parse_pci_id(&format!("{:#x}", 0x1234)).unwrap(),
        parse_pci_id(&format!("{:#X}", 0x1234)).unwrap(),
    );
}
