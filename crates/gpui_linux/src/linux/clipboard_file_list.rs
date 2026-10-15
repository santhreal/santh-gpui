use gpui::{ClipboardEntry, ClipboardItem, ExternalPaths, SharedString};
use smallvec::SmallVec;
use std::os::unix::ffi::OsStrExt;
use url::Url;

pub(crate) const FILE_LIST_MIME_TYPE: &str = "text/uri-list";

pub(crate) fn read_file_list(bytes: &[u8]) -> Result<ClipboardItem, &'static str> {
    let text = std::str::from_utf8(bytes).map_err(|_| "file list is not UTF-8")?;
    let mut paths = SmallVec::new();
    for line in text
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let url = Url::parse(line).map_err(|_| "file list contains an invalid URI")?;
        if url.query().is_some() || url.fragment().is_some() {
            return Err("file URI contains a query or fragment");
        }
        let path = url
            .to_file_path()
            .map_err(|_| "file list contains a non-local URI")?;
        if path.as_os_str().as_bytes().contains(&0) {
            return Err("file path contains a null byte");
        }
        paths.push(path);
    }
    if paths.is_empty() {
        return Err("file list contains no files");
    }
    Ok(ClipboardItem {
        entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths::from_paths(
            paths,
        ))],
    })
}

pub(crate) fn parse_uri_list_for_drop(text: &str) -> ExternalPaths {
    let mut paths = SmallVec::new();
    let mut urls = SmallVec::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let Ok(url) = Url::parse(line) else {
            log::error!("Invalid URI in drop list: {line:?}");
            continue;
        };
        if url.scheme() == "file" {
            match url.to_file_path() {
                Ok(path) => paths.push(path),
                Err(()) => {
                    log::error!("Failed turn {url:?} into a file path");
                }
            }
        } else {
            urls.push(SharedString::from(line.to_owned()));
        }
    }
    ExternalPaths::new(paths, urls)
}

/// URI-list clipboard payloads must remain local file entries, not pasted text or
/// partially accepted lists. These tests cover decoding; backend negotiation is
/// exercised separately on an isolated display.
#[cfg(test)]
mod tests {
    use super::*;
    use std::{ffi::OsStr, path::PathBuf};

    #[test]
    fn comments_line_endings_and_escaped_paths_preserve_order() -> Result<(), &'static str> {
        for newline in ["\n", "\r\n"] {
            let input = [
                "# file manager selection",
                "",
                "file:///workspace/a%20b.txt",
                "file://localhost/workspace/%23notes",
                "",
            ]
            .join(newline);
            assert_eq!(
                read_file_list(input.as_bytes())?,
                ClipboardItem {
                    entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths::from_paths(
                        smallvec::smallvec![
                            PathBuf::from("/workspace/a b.txt"),
                            PathBuf::from("/workspace/#notes"),
                        ],
                    ))],
                }
            );
        }
        Ok(())
    }

    #[test]
    fn non_utf8_file_names_round_trip_without_loss() -> Result<(), &'static str> {
        let path = PathBuf::from(OsStr::from_bytes(b"/workspace/\xff.txt"));
        let url = Url::from_file_path(&path).map_err(|_| "invalid fixture path")?;
        assert_eq!(
            read_file_list(url.as_str().as_bytes())?,
            ClipboardItem {
                entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths::from_paths(
                    smallvec::smallvec![path],
                ))],
            }
        );
        Ok(())
    }

    #[test]
    fn invalid_lists_never_return_partial_file_selections() {
        for (bytes, error) in [
            (&b""[..], "file list contains no files"),
            (&b"# comment\r\n\r\n"[..], "file list contains no files"),
            (&b"\xff"[..], "file list is not UTF-8"),
            (&b"not a URI"[..], "file list contains an invalid URI"),
            (
                &b"file:///workspace/ok\nhttps://example.com/file"[..],
                "file list contains a non-local URI",
            ),
            (
                &b"file://remote/workspace/file"[..],
                "file list contains a non-local URI",
            ),
            (
                &b"file:///workspace/file?query"[..],
                "file URI contains a query or fragment",
            ),
            (
                &b"file:///workspace/file#fragment"[..],
                "file URI contains a query or fragment",
            ),
            (
                &b"file:///workspace/a%00b"[..],
                "file path contains a null byte",
            ),
        ] {
            assert_eq!(read_file_list(bytes), Err(error), "input: {bytes:?}");
        }
    }

    #[test]
    fn drop_uri_list_collects_non_file_urls_and_local_paths() {
        let url_only = parse_uri_list_for_drop(
            "# browser drag\r\nhttps://example.com/clip.mp4\r\nhttps://example.com/sub.vtt\r\n",
        );
        assert!(url_only.paths().is_empty());
        assert_eq!(
            url_only.urls(),
            &[
                SharedString::from("https://example.com/clip.mp4"),
                SharedString::from("https://example.com/sub.vtt"),
            ]
        );

        let mixed = parse_uri_list_for_drop(
            "file:///workspace/a%20b.mp4\r\nhttps://example.com/stream.m3u8\r\n",
        );
        assert_eq!(mixed.paths(), &[PathBuf::from("/workspace/a b.mp4")]);
        assert_eq!(
            mixed.urls(),
            &[SharedString::from("https://example.com/stream.m3u8")]
        );

        let invalid = parse_uri_list_for_drop("not a uri\r\n# comment\r\n");
        assert!(invalid.is_empty());
    }
}
