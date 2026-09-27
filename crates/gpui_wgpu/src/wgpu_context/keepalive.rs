//! In the tests of this crate, the device of a context lives to the end of
//! the process, and so does every shared object loaded while it lives.
//!
//! The NVIDIA driver (580) deadlocked in `vkDestroyDevice`: it joined a
//! worker thread of the driver whose thread-specific-data destructor
//! blocked on a driver mutex. The tests create and drop devices on many
//! threads at once and hit it in about one run in twelve, so a test build
//! never destroys a device.
//!
//! A device that is never destroyed keeps its worker threads until the
//! process exits. At exit, the libglvnd `libEGL.so.1` destructor closes
//! the NVIDIA EGL vendor library, which unmaps `libnvidia-eglcore`, the
//! code those threads run. The test process crashed after its last test in
//! 14 runs of 40. Pinning every loaded object keeps that code mapped.

use std::sync::Arc;

use super::WgpuContext;

impl Drop for WgpuContext {
    fn drop(&mut self) {
        std::mem::forget(Arc::clone(&self.device));
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        pin_loaded_objects();
    }
}

/// Marks every shared object loaded now `RTLD_NODELETE`, so no later
/// `dlclose` unmaps it before the process exits.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn pin_loaded_objects() {
    use std::ffi::{CStr, CString};

    unsafe extern "C" fn collect(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        names: *mut libc::c_void,
    ) -> libc::c_int {
        // SAFETY: `dl_iterate_phdr` passes a valid `info` for the duration
        // of the call and the `names` pointer given to it below.
        let (name, names) = unsafe { ((*info).dlpi_name, &mut *names.cast::<Vec<CString>>()) };
        if !name.is_null() {
            // SAFETY: a non-null `dlpi_name` is a NUL-terminated string.
            let name = unsafe { CStr::from_ptr(name) };
            // The main program has an empty name and cannot be unloaded.
            if !name.is_empty() {
                names.push(name.to_owned());
            }
        }
        0
    }

    // `dlopen` runs after the walk: `dl_iterate_phdr` holds the loader
    // write lock, and taking the load lock under it inverts the order
    // `dlopen` uses on other threads.
    let mut names = Vec::<CString>::new();
    // SAFETY: `collect` matches the callback contract and `names` outlives
    // the call.
    unsafe { libc::dl_iterate_phdr(Some(collect), (&raw mut names).cast()) };
    for name in names {
        // A null handle means the object was closed after the walk, and
        // nothing is left to pin. The handle is never closed.
        // SAFETY: `name` is NUL-terminated; `RTLD_NOLOAD` loads nothing new.
        unsafe {
            libc::dlopen(
                name.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
            )
        };
    }
}
