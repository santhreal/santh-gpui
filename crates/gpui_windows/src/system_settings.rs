use std::{
    cell::Cell,
    ffi::{c_uint, c_void},
};

use gpui_util::ResultExt;
use windows::{
    Win32::UI::WindowsAndMessaging::{
        SPI_GETCLIENTAREAANIMATION, SPI_GETWHEELSCROLLCHARS, SPI_GETWHEELSCROLLLINES,
        SPI_SETCLIENTAREAANIMATION, SYSTEM_PARAMETERS_INFO_ACTION,
        SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SystemParametersInfoW,
    },
    core::BOOL,
};

/// Reads the reduced-motion preference of the system: the "Show animations in
/// Windows" setting, `SPI_GETCLIENTAREAANIMATION`. A failed read returns
/// `false`.
pub(crate) fn system_reduce_motion() -> bool {
    let mut client_area_animation = BOOL::default();
    let result = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            Some((&mut client_area_animation) as *mut BOOL as *mut c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS::default(),
        )
    };
    result.log_err().is_some() && reduce_motion_from_client_area_animation(client_area_animation)
}

/// Whether a `SPI_GETCLIENTAREAANIMATION` value requests reduced motion:
/// `FALSE` turns client-area animations off.
pub(crate) fn reduce_motion_from_client_area_animation(client_area_animation: BOOL) -> bool {
    !client_area_animation.as_bool()
}

/// Whether the `wParam` of a `WM_SETTINGCHANGE` message reports a change of
/// the setting read by [`system_reduce_motion`].
pub(crate) fn is_reduce_motion_setting_change(wparam: usize) -> bool {
    u32::try_from(wparam)
        .is_ok_and(|action| SYSTEM_PARAMETERS_INFO_ACTION(action) == SPI_SETCLIENTAREAANIMATION)
}

/// Windows settings pulled from SystemParametersInfo
/// https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-systemparametersinfow
#[derive(Default, Debug, Clone)]
pub(crate) struct WindowsSystemSettings {
    pub(crate) mouse_wheel_settings: MouseWheelSettings,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct MouseWheelSettings {
    /// SEE: SPI_GETWHEELSCROLLCHARS
    pub(crate) wheel_scroll_chars: Cell<u32>,
    /// SEE: SPI_GETWHEELSCROLLLINES
    pub(crate) wheel_scroll_lines: Cell<u32>,
}

impl WindowsSystemSettings {
    pub(crate) fn new() -> Self {
        let mut settings = Self::default();
        settings.init();
        settings
    }

    fn init(&mut self) {
        self.mouse_wheel_settings.update();
    }

    pub(crate) fn update(&self, wparam: usize) {
        match SYSTEM_PARAMETERS_INFO_ACTION(wparam as u32) {
            SPI_GETWHEELSCROLLLINES | SPI_GETWHEELSCROLLCHARS => self.update_mouse_wheel_settings(),
            _ => {}
        }
    }

    fn update_mouse_wheel_settings(&self) {
        self.mouse_wheel_settings.update();
    }
}

impl MouseWheelSettings {
    fn update(&self) {
        self.update_wheel_scroll_chars();
        self.update_wheel_scroll_lines();
    }

    fn update_wheel_scroll_chars(&self) {
        let mut value = c_uint::default();
        let result = unsafe {
            SystemParametersInfoW(
                SPI_GETWHEELSCROLLCHARS,
                0,
                Some((&mut value) as *mut c_uint as *mut c_void),
                SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS::default(),
            )
        };

        if result.log_err() != None && self.wheel_scroll_chars.get() != value {
            self.wheel_scroll_chars.set(value);
        }
    }

    fn update_wheel_scroll_lines(&self) {
        let mut value = c_uint::default();
        let result = unsafe {
            SystemParametersInfoW(
                SPI_GETWHEELSCROLLLINES,
                0,
                Some((&mut value) as *mut c_uint as *mut c_void),
                SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS::default(),
            )
        };

        if result.log_err() != None && self.wheel_scroll_lines.get() != value {
            self.wheel_scroll_lines.set(value);
        }
    }
}

/// The mapping from the Windows animation setting to the reduced-motion
/// preference and the selection of the `WM_SETTINGCHANGE` messages that
/// report a change of it. Does not cover the `SystemParametersInfoW` call.
#[cfg(test)]
mod reduce_motion_tests {
    use windows::{
        Win32::UI::WindowsAndMessaging::{
            SPI_GETCLIENTAREAANIMATION, SPI_SETCLIENTAREAANIMATION, SPI_SETWHEELSCROLLCHARS,
            SPI_SETWHEELSCROLLLINES,
        },
        core::BOOL,
    };

    use super::{is_reduce_motion_setting_change, reduce_motion_from_client_area_animation};

    #[test]
    fn animations_off_requests_reduced_motion() {
        assert!(reduce_motion_from_client_area_animation(BOOL(0)));
    }

    #[test]
    fn every_nonzero_value_keeps_motion() {
        for value in [1, -1, 2, i32::MAX, i32::MIN] {
            assert!(
                !reduce_motion_from_client_area_animation(BOOL(value)),
                "BOOL({value}) enables animations"
            );
        }
    }

    #[test]
    fn only_the_animation_setting_reports_a_change() {
        assert!(is_reduce_motion_setting_change(
            SPI_SETCLIENTAREAANIMATION.0 as usize
        ));
        for action in [
            0,
            SPI_GETCLIENTAREAANIMATION.0 as usize,
            SPI_SETWHEELSCROLLLINES.0 as usize,
            SPI_SETWHEELSCROLLCHARS.0 as usize,
        ] {
            assert!(
                !is_reduce_motion_setting_change(action),
                "action {action:#x} is not the animation setting"
            );
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn wparam_above_u32_is_not_the_animation_setting() {
        let action = (1usize << 32) | SPI_SETCLIENTAREAANIMATION.0 as usize;
        assert!(!is_reduce_motion_setting_change(action));
    }
}
