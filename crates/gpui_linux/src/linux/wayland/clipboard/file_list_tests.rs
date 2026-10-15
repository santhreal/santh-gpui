use super::*;
use gpui::ExternalPaths;
use std::{os::unix::net::UnixStream, path::PathBuf};

struct FileOffer(&'static [u8]);
impl ReceiveData for FileOffer {
    fn receive_data(&self, mime_type: String, fd: BorrowedFd<'_>) {
        assert_eq!(
            mime_type, FILE_LIST_MIME_TYPE,
            "files must precede their text representation"
        );
        let result = fd
            .try_clone_to_owned()
            .map(File::from)
            .and_then(|mut file| file.write_all(self.0));
        assert!(result.is_ok(), "cannot write fixture offer: {result:?}");
    }
}

/// Exercise the MIME selection and timed pipe decoder used by both Wayland
/// clipboards. The offer is synthetic; compositor dispatch is not covered here.
#[test]
fn file_offers_precede_text_and_invalid_files_do_not_fall_back() -> anyhow::Result<()> {
    let (socket, _server) = UnixStream::pair()?;
    let connection = Connection::from_socket(socket)?;
    for files_first in [false, true] {
        for (bytes, accepted) in [
            (b"file:///workspace/notes%20one.txt\r\n".as_slice(), true),
            (
                b"file:///workspace/ok\nhttps://example.com/file".as_slice(),
                false,
            ),
            (b"\xff".as_slice(), false),
            (b"# empty\n".as_slice(), false),
        ] {
            let mut offer = DataOffer::new(FileOffer(bytes));
            let formats = if files_first {
                [FILE_LIST_MIME_TYPE, "UTF8_STRING"]
            } else {
                ["UTF8_STRING", FILE_LIST_MIME_TYPE]
            };
            for format in formats {
                offer.add_mime_type(format.into());
            }
            let expected = accepted.then(|| ClipboardItem {
                entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths::from_paths(
                    smallvec::smallvec![PathBuf::from("/workspace/notes one.txt")],
                ))],
            });
            assert_eq!(offer.read_item(&connection), expected);
        }
    }
    Ok(())
}
