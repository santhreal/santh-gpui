use super::*;
use gpui::{ClipboardEntry, ExternalPaths};
use std::path::PathBuf;

/// Every X11 selection must decode file-manager URI lists as files and reject
/// invalid lists. This exercises get_any, including target selection, rather
/// than only the parser. External-owner transfers also have a native GUI smoke.
#[test]
#[ignore = "requires a dedicated X server and GPUI_CLIPBOARD_TEST_DISPLAY"]
fn uri_lists_remain_file_entries_on_every_selection() -> anyhow::Result<()> {
    let display = std::env::var("GPUI_CLIPBOARD_TEST_DISPLAY")?;
    anyhow::ensure!(
        display != ":0" && display != ":1",
        "a private display is required"
    );
    anyhow::ensure!(
        std::env::var("DISPLAY")? == display,
        "test display must match DISPLAY"
    );
    let clipboard = Clipboard::new()?;
    for selection in ClipboardKind::iter() {
        let expected = ClipboardItem {
            entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths::from_paths(
                smallvec::smallvec![
                    PathBuf::from("/workspace/project notes.txt"),
                    PathBuf::from("/workspace/#second.txt"),
                ],
            ))],
        };
        clipboard.inner.write(vec![ClipboardData {
			bytes: b"# selection\r\nfile:///workspace/project%20notes.txt\r\nfile:///workspace/%23second.txt\r\n".to_vec(),
			format: clipboard.inner.atoms.URI_LIST,
		}], selection, WaitConfig::None)?;
        assert_eq!(clipboard.get_any(selection)?, expected, "{selection:?}");
        for bytes in [
            b"https://example.com/file".as_slice(),
            b"file:///workspace/ok\nfile://remote/file",
            b"\xff",
            b"# empty\n",
        ] {
            clipboard.inner.write(
                vec![ClipboardData {
                    bytes: bytes.to_vec(),
                    format: clipboard.inner.atoms.URI_LIST,
                }],
                selection,
                WaitConfig::None,
            )?;
            assert!(
                matches!(clipboard.get_any(selection), Err(Error::ConversionFailure)),
                "{selection:?}: {bytes:?}"
            );
        }
        clipboard.set_text(
            Cow::Borrowed("ordinary clipboard text"),
            selection,
            WaitConfig::None,
        )?;
        assert_eq!(
            clipboard.get_any(selection)?,
            ClipboardItem::new_string("ordinary clipboard text".into())
        );
    }
    Ok(())
}
