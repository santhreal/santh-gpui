use std::sync::Arc;

use crate::WgpuContext;

/// In a test build, the device of a dropped context stays alive. The NVIDIA
/// driver (580) deadlocked the tests of this crate in `vkDestroyDevice` in
/// about one run in twelve. Does not cover a device created outside
/// `WgpuContext`, which clippy rejects through `disallowed-methods`.
#[test]
fn a_dropped_context_leaves_its_device_alive() {
    let context = WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
        .expect("surfaceless context");
    let device = Arc::downgrade(&context.device);
    drop(context);
    assert!(
        device.upgrade().is_some(),
        "dropping the context released its device"
    );
}

/// In a test build, a shared object loaded while a context lives stays
/// mapped after its last `dlclose`. The devices the tests keep alive run
/// driver code until the process exits, and the libglvnd `libEGL.so.1`
/// destructor closed the NVIDIA driver under them: the test process
/// crashed after its last test in 14 runs of 40. The probe is a glibc
/// library that no part of the renderer loads. Does not cover an object
/// loaded after the last context is dropped.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn a_dropped_context_keeps_loaded_objects_mapped() {
    const PROBE: &std::ffi::CStr = c"libBrokenLocale.so.1";
    fn is_loaded() -> bool {
        // SAFETY: `PROBE` is NUL-terminated; `RTLD_NOLOAD` loads nothing,
        // and the handle is closed so the check leaves the count as found.
        unsafe {
            let handle = libc::dlopen(PROBE.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
            !handle.is_null() && libc::dlclose(handle) == 0
        }
    }

    assert!(!is_loaded(), "{PROBE:?} is loaded before the test loads it");
    // SAFETY: `PROBE` is NUL-terminated.
    let handle = unsafe { libc::dlopen(PROBE.as_ptr(), libc::RTLD_NOW) };
    assert!(!handle.is_null(), "dlopen {PROBE:?} failed");
    drop(
        WgpuContext::new_surfaceless(WgpuContext::surfaceless_instance(), None)
            .expect("surfaceless context"),
    );
    // SAFETY: `handle` came from the `dlopen` above and is closed once.
    assert_eq!(unsafe { libc::dlclose(handle) }, 0, "dlclose {PROBE:?}");
    assert!(
        is_loaded(),
        "the last dlclose of {PROBE:?} unmapped an object loaded while a context lived"
    );
}
