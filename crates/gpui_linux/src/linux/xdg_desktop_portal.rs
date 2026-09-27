//! Provides a [calloop] event source from [XDG Desktop Portal] events
//!
//! This module uses the [ashpd] crate

use ashpd::desktop::settings::{ColorScheme, Settings};
use calloop::channel::Channel;
use calloop::{EventSource, Poll, PostAction, Readiness, Token, TokenFactory};
use smol::stream::StreamExt;

use gpui::{BackgroundExecutor, WindowAppearance};

pub enum Event {
    WindowAppearance(WindowAppearance),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorTheme(String),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorSize(u32),
    ButtonLayout(String),
}

pub struct XDPEventSource {
    channel: Channel<Event>,
}

impl XDPEventSource {
    pub fn new(executor: &BackgroundExecutor) -> Self {
        let (sender, channel) = calloop::channel::channel();

        let background = executor.clone();

        executor
            .spawn(async move {
                let settings = Settings::new().await?;

                if let Ok(initial_appearance) = settings.color_scheme().await {
                    sender.send(Event::WindowAppearance(
                        window_appearance_from_color_scheme(initial_appearance),
                    ))?;
                }
                if let Ok(initial_theme) = settings
                    .read::<String>("org.gnome.desktop.interface", "cursor-theme")
                    .await
                {
                    sender.send(Event::CursorTheme(initial_theme))?;
                }

                // If u32 is used here, it throws invalid type error
                if let Ok(initial_size) = settings
                    .read::<i32>("org.gnome.desktop.interface", "cursor-size")
                    .await
                {
                    sender.send(Event::CursorSize(initial_size as u32))?;
                }

                if let Ok(initial_layout) = settings
                    .read::<String>("org.gnome.desktop.wm.preferences", "button-layout")
                    .await
                {
                    sender.send(Event::ButtonLayout(initial_layout))?;
                }

                if let Ok(mut cursor_theme_changed) = settings
                    .receive_setting_changed_with_args(
                        "org.gnome.desktop.interface",
                        "cursor-theme",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(theme) = cursor_theme_changed.next().await {
                                let theme = theme?;
                                sender.send(Event::CursorTheme(theme))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut cursor_size_changed) = settings
                    .receive_setting_changed_with_args::<i32>(
                        "org.gnome.desktop.interface",
                        "cursor-size",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(size) = cursor_size_changed.next().await {
                                let size = size?;
                                sender.send(Event::CursorSize(size as u32))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut button_layout_changed) = settings
                    .receive_setting_changed_with_args(
                        "org.gnome.desktop.wm.preferences",
                        "button-layout",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(layout) = button_layout_changed.next().await {
                                let layout = layout?;
                                sender.send(Event::ButtonLayout(layout))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                let mut appearance_changed = settings.receive_color_scheme_changed().await?;
                while let Some(scheme) = appearance_changed.next().await {
                    sender.send(Event::WindowAppearance(
                        window_appearance_from_color_scheme(scheme),
                    ))?;
                }

                anyhow::Ok(())
            })
            .detach();

        Self { channel }
    }
}

impl EventSource for XDPEventSource {
    type Event = Event;
    type Metadata = ();
    type Ret = ();
    type Error = anyhow::Error;

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: F,
    ) -> Result<PostAction, Self::Error>
    where
        F: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        self.channel.process_events(readiness, token, |evt, _| {
            if let calloop::channel::Event::Msg(msg) = evt {
                (callback)(msg, &mut ())
            }
        })?;

        Ok(PostAction::Continue)
    }

    fn register(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.register(poll, token_factory)?;

        Ok(())
    }

    fn reregister(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.reregister(poll, token_factory)?;

        Ok(())
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.channel.unregister(poll)?;

        Ok(())
    }
}

fn window_appearance_from_color_scheme(cs: ColorScheme) -> WindowAppearance {
    match cs {
        ColorScheme::PreferDark => WindowAppearance::Dark,
        ColorScheme::PreferLight => WindowAppearance::Light,
        ColorScheme::NoPreference => WindowAppearance::Light,
    }
}

const APPEARANCE_NAMESPACE: &str = "org.freedesktop.appearance";
const REDUCED_MOTION_KEY: &str = "reduced-motion";
const GNOME_INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";
const ENABLE_ANIMATIONS_KEY: &str = "enable-animations";

/// The portal settings that select the reduced-motion preference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MotionSettings {
    /// `org.freedesktop.appearance` `reduced-motion`, when the portal has the
    /// key: `1` requests reduced motion, any other value requests none.
    pub(crate) reduced_motion: Option<u32>,
    /// `org.gnome.desktop.interface` `enable-animations`, when the portal has
    /// the key: `false` requests reduced motion.
    pub(crate) enable_animations: Option<bool>,
}

/// A new value of one of the [`MotionSettings`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum MotionSetting {
    ReducedMotion(u32),
    EnableAnimations(bool),
}

impl MotionSettings {
    /// Whether the settings request reduced motion. `reduced-motion` takes
    /// precedence over `enable-animations`; with neither key the result is
    /// `false`.
    pub(crate) fn reduce_motion(self) -> bool {
        match (self.reduced_motion, self.enable_animations) {
            (Some(reduced_motion), _) => reduced_motion == 1,
            (None, Some(enable_animations)) => !enable_animations,
            (None, None) => false,
        }
    }

    /// Records `setting` and returns the reduced-motion preference when the
    /// new value changed it.
    pub(crate) fn apply(&mut self, setting: MotionSetting) -> Option<bool> {
        let before = self.reduce_motion();
        match setting {
            MotionSetting::ReducedMotion(value) => self.reduced_motion = Some(value),
            MotionSetting::EnableAnimations(value) => self.enable_animations = Some(value),
        }
        let after = self.reduce_motion();
        (after != before).then_some(after)
    }
}

/// Sends the reduced-motion preference of the portal to `sender`: the
/// preference once the portal answers, then each change of it. Returns an
/// error when the portal is unavailable or `sender` has no receiver, and
/// returns `Ok` when the portal stops sending changes.
pub(crate) async fn watch_reduced_motion(
    sender: smol::channel::Sender<bool>,
) -> anyhow::Result<()> {
    let settings = Settings::new().await?;
    // Subscribing before the reads keeps a change made between the two.
    let reduced_motion_changed = settings
        .receive_setting_changed_with_args::<u32>(APPEARANCE_NAMESPACE, REDUCED_MOTION_KEY)
        .await?
        .map(|value| value.map(MotionSetting::ReducedMotion));
    let enable_animations_changed = settings
        .receive_setting_changed_with_args::<bool>(GNOME_INTERFACE_NAMESPACE, ENABLE_ANIMATIONS_KEY)
        .await?
        .map(|value| value.map(MotionSetting::EnableAnimations));
    let mut changes = std::pin::pin!(reduced_motion_changed.or(enable_animations_changed));

    let mut motion = MotionSettings {
        reduced_motion: settings
            .read::<u32>(APPEARANCE_NAMESPACE, REDUCED_MOTION_KEY)
            .await
            .ok(),
        enable_animations: settings
            .read::<bool>(GNOME_INTERFACE_NAMESPACE, ENABLE_ANIMATIONS_KEY)
            .await
            .ok(),
    };
    sender.send(motion.reduce_motion()).await?;

    while let Some(setting) = changes.next().await {
        match setting {
            Ok(setting) => {
                if let Some(reduce_motion) = motion.apply(setting) {
                    sender.send(reduce_motion).await?;
                }
            }
            Err(error) => {
                log::warn!("ignoring a reduced-motion setting of the XDG desktop portal: {error}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! WHY: closes the class "the portal settings select the wrong
    //! reduced-motion preference": a misread value, the GNOME key overriding
    //! the freedesktop key, a missing key read as a request, and a change of
    //! either key that is dropped or reported without altering the
    //! preference. Not caught: the D-Bus reads and subscriptions of
    //! [`watch_reduced_motion`], which need a running portal.

    use super::{MotionSetting, MotionSettings};

    #[test]
    fn the_freedesktop_key_takes_precedence_and_absent_keys_request_no_reduction() {
        // (reduced-motion, enable-animations, reduce motion)
        let cases = [
            (None, None, false),
            (None, Some(true), false),
            (None, Some(false), true),
            (Some(0), None, false),
            (Some(1), None, true),
            (Some(2), None, false),
            (Some(0), Some(false), false),
            (Some(0), Some(true), false),
            (Some(1), Some(true), true),
            (Some(1), Some(false), true),
            (Some(2), Some(false), false),
            (Some(u32::MAX), Some(false), false),
        ];
        for (reduced_motion, enable_animations, expected) in cases {
            let settings = MotionSettings {
                reduced_motion,
                enable_animations,
            };
            assert_eq!(settings.reduce_motion(), expected, "{settings:?}");
        }
    }

    #[test]
    fn a_change_reports_the_preference_only_when_it_alters_it() {
        let mut settings = MotionSettings::default();
        // (setting, reported preference)
        let steps = [
            (MotionSetting::EnableAnimations(true), None),
            (MotionSetting::EnableAnimations(false), Some(true)),
            (MotionSetting::EnableAnimations(false), None),
            // The freedesktop key appears and overrides the GNOME key.
            (MotionSetting::ReducedMotion(0), Some(false)),
            (MotionSetting::EnableAnimations(true), None),
            (MotionSetting::EnableAnimations(false), None),
            (MotionSetting::ReducedMotion(1), Some(true)),
            (MotionSetting::EnableAnimations(true), None),
            (MotionSetting::ReducedMotion(2), Some(false)),
            (MotionSetting::ReducedMotion(0), None),
        ];
        for (step, (setting, reported)) in steps.into_iter().enumerate() {
            let before = settings;
            assert_eq!(
                settings.apply(setting),
                reported,
                "step {step}: {setting:?} applied to {before:?}"
            );
        }
    }
}
