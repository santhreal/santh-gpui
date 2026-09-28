//! WHY: an open of the clipboard failed at once while another process held
//! it, as a clipboard listener does right after each change, so a copy
//! wrote nothing and a paste read nothing. The class closed here is an
//! open that does not outlast a brief hold by another process, and an
//! open that waits without bound on a hold that does not end. The round
//! trip covers text with and without metadata, each read back right after
//! its write, when a listener on the desktop reads the change.
//!
//! Not covered: a holder that keeps the clipboard past the attempts,
//! whose copy or paste still fails, and formats other than text.

#![allow(
    clippy::disallowed_methods,
    reason = "a test process spawns its holder and waits for it"
)]

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gpui::ClipboardItem;
use windows::Win32::System::DataExchange::{CloseClipboard, OpenClipboard};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, HWND_MESSAGE, WINDOW_EX_STYLE, WINDOW_STYLE,
};
use windows::core::w;

use super::{ClipboardGuard, OPEN_ATTEMPTS, OPEN_RETRY, read_from_clipboard, write_to_clipboard};

/// The clipboard is one per session: every case that opens it holds this.
static CLIPBOARD: Mutex<()> = Mutex::new(());

/// The variable that makes `hold_the_clipboard` hold it, for its value in
/// milliseconds, and the line it prints once it holds it.
const HOLD_MS: &str = "GPUI_TEST_CLIPBOARD_HOLD_MS";
const HELD: &str = "clipboard held";

fn exclusive() -> MutexGuard<'static, ()> {
    CLIPBOARD.lock().unwrap_or_else(|e| e.into_inner())
}

/// A process of this test binary that holds the clipboard open, and its
/// output. The clipboard admits one process at a time; a thread of this
/// process would share its open.
struct Holder {
    child: Child,
    stdout: BufReader<ChildStdout>,
}

impl Holder {
    /// Returns once the holder holds the clipboard, which it closes `hold`
    /// later.
    fn start(hold: Duration) -> Holder {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "clipboard::tests::hold_the_clipboard",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(HOLD_MS, hold.as_millis().to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        // The harness writes the test's name on the same line first.
        let held = stdout
            .by_ref()
            .lines()
            .map_while(Result::ok)
            .any(|line| line.ends_with(HELD));
        assert!(held, "the holder exited before it held the clipboard");
        Holder { child, stdout }
    }

    /// Waits for the holder to exit, reading the rest of its output so its
    /// writes never meet a closed pipe.
    fn finish(mut self) {
        std::io::copy(&mut self.stdout, &mut std::io::sink()).unwrap();
        let status = self.child.wait().unwrap();
        assert!(status.success(), "the holder exited {status}");
    }
}

/// Run by `Holder::start` in a child process. It opens the clipboard for a
/// message-only window, as a listener opens it for its own window: an open
/// for no window takes over another process's open for no window.
#[test]
#[ignore = "run by Holder::start in a child process"]
fn hold_the_clipboard() {
    let Ok(ms) = std::env::var(HOLD_MS) else {
        return;
    };
    let hold = Duration::from_millis(ms.parse().unwrap());
    let window = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("STATIC"),
            None,
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            None,
            None,
        )
    }
    .expect("the holder creates a message-only window");
    unsafe { OpenClipboard(Some(window)) }.expect("the holder opens the clipboard");
    println!("{HELD}");
    std::io::stdout().flush().unwrap();
    std::thread::sleep(hold);
    unsafe { CloseClipboard() }.expect("the holder closes the clipboard");
    unsafe { DestroyWindow(window) }.expect("the holder destroys its window");
}

#[test]
fn an_open_waits_out_a_hold_shorter_than_its_attempts() {
    let _clipboard = exclusive();
    // Past the first attempt, inside the waits between the rest.
    let hold = OPEN_RETRY * 2;
    let holder = Holder::start(hold);
    let clip = ClipboardGuard::open();
    assert!(
        clip.is_some(),
        "an open failed while another process held the clipboard for {hold:?}"
    );
    drop(clip);
    holder.finish();
}

#[test]
fn an_open_gives_up_on_a_hold_that_outlasts_its_attempts() {
    let _clipboard = exclusive();
    let holder = Holder::start(Duration::from_secs(1));
    let start = Instant::now();
    let clip = ClipboardGuard::open();
    let took = start.elapsed();
    assert!(
        clip.is_none(),
        "an open succeeded while another process held the clipboard"
    );
    // A sleep rounds up to the timer resolution, 15.6 ms by default.
    let bound = (OPEN_RETRY + Duration::from_millis(16)) * OPEN_ATTEMPTS;
    assert!(
        took < bound,
        "an open gave up after {took:?}, past its bound of {bound:?}"
    );
    holder.finish();
}

#[test]
fn text_reads_back_as_written() {
    let _clipboard = exclusive();
    for item in [
        ClipboardItem::new_string("你好，我是张小白".to_string()),
        ClipboardItem::new_string("12345".to_string()),
        ClipboardItem::new_string_with_json_metadata("abcdef".to_string(), vec![3, 4]),
    ] {
        write_to_clipboard(item.clone());
        assert_eq!(read_from_clipboard(), Some(item));
    }
}
