//! [`windowed_platform`](super::windowed_platform) returns an error for every
//! way the environment leaves a windowed application without a display
//! server, where [`current_platform`](super::current_platform) runs headless
//! or panics: no display variable set, `ZED_HEADLESS` set, an X server that is
//! not running, and a Wayland compositor that is not running.
//!
//! The display variables belong to the whole process, so each case runs this
//! test binary again as a child with its own environment, and the child
//! prints what `windowed_platform` returned.
//!
//! Not covered: a display server that accepts the connection and then fails,
//! which needs a running server.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Set in the child's environment; the child test does nothing without it.
const CHILD: &str = "GPUI_LINUX_WINDOWED_PLATFORM_CHILD";

/// Test name of [`connect_in_child`], as the child process is filtered to it.
const CHILD_TEST: &str = "linux::windowed_platform_tests::connect_in_child";

/// An X display number with no server. The parent checks that its socket is
/// absent before the X case runs.
const ABSENT_X: &str = ":9973";
const ABSENT_X_SOCKET: &str = "/tmp/.X11-unix/X9973";

/// How long a child may take. A connection to an absent server fails at
/// once; a child that runs headless or blocks on a socket ends here.
const DEADLINE: Duration = Duration::from_secs(30);

/// One environment without a display server to connect to: the variables
/// set over an environment with every display variable removed, and the
/// text the error must contain.
struct Case {
    name: &'static str,
    env: &'static [(&'static str, &'static str)],
    reason: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "no display variable",
        env: &[],
        reason: "neither WAYLAND_DISPLAY nor DISPLAY is set",
    },
    Case {
        name: "ZED_HEADLESS over a display",
        env: &[("ZED_HEADLESS", "1"), ("DISPLAY", ABSENT_X)],
        reason: "ZED_HEADLESS is set, so no window can open",
    },
    Case {
        name: "X server not running",
        env: &[("DISPLAY", ABSENT_X)],
        reason: "cannot connect to the X server at DISPLAY",
    },
    Case {
        name: "Wayland compositor not running",
        env: &[
            ("WAYLAND_DISPLAY", "gpui-absent-wayland"),
            ("XDG_RUNTIME_DIR", "/nonexistent-gpui-runtime-dir"),
        ],
        reason: "cannot connect to the Wayland compositor at WAYLAND_DISPLAY",
    },
];

/// The child: prints the result of `windowed_platform` on one line.
#[test]
fn connect_in_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    match super::windowed_platform() {
        Ok(_) => println!("RESULT connected"),
        Err(err) => println!("RESULT error: {err:#}"),
    }
}

/// Every case ends within [`DEADLINE`] with an error that states its reason,
/// and none panics.
#[test]
fn windowed_platform_reports_every_unreachable_display() {
    assert!(
        !Path::new(ABSENT_X_SOCKET).exists(),
        "{ABSENT_X_SOCKET} exists, so {ABSENT_X} has a server; pick a free display number"
    );
    let exe = std::env::current_exe().expect("the test binary has a path");
    let mut wrong = Vec::new();
    for case in CASES {
        let mut child = Command::new(&exe)
            .args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .env_remove("ZED_HEADLESS")
            .env(CHILD, "1")
            .envs(case.env.iter().copied())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the test binary runs");
        let started = Instant::now();
        let ended = loop {
            if child
                .try_wait()
                .expect("the child can be waited on")
                .is_some()
            {
                break true;
            }
            if started.elapsed() > DEADLINE {
                child.kill().ok();
                break false;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let output = child.wait_with_output().expect("the child's output");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        // libtest prints the test's name on the same line before the result.
        let result = stdout
            .lines()
            .find_map(|line| line.split_once("RESULT ").map(|(_, result)| result))
            .unwrap_or("no result");
        if !ended {
            wrong.push(format!("{}: still running after {DEADLINE:?}", case.name));
        } else if !result.starts_with("error: ") || !result.contains(case.reason) {
            wrong.push(format!(
                "{}: expected an error containing {:?}, got {result:?}; stderr: {}",
                case.name,
                case.reason,
                stderr.trim()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
