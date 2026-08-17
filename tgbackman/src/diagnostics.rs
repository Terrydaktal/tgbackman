//! Low-overhead GUI diagnostics: build identity, panic evidence and a live
//! state snapshot that can be collected without attaching a debugger.

use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Serialize)]
pub(crate) struct BuildIdentity {
    pub(crate) package: &'static str,
    pub(crate) version: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) dirty: &'static str,
    pub(crate) profile: &'static str,
}

pub(crate) fn build_identity() -> &'static BuildIdentity {
    static IDENTITY: OnceLock<BuildIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| BuildIdentity {
        package: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        revision: option_env!("TGBACKMAN_BUILD_REVISION").unwrap_or("unknown"),
        dirty: option_env!("TGBACKMAN_BUILD_DIRTY").unwrap_or("unknown"),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    })
}

fn state_root() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_fallback_home().join(".local").join("state"))
        .join("tgbackman")
}

fn dirs_fallback_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn now_unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn protect_file(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, PermissionsExt::from_mode(0o600));
    }
}

fn process_resources() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut rss_kib = 0;
        let mut threads = 0;
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                rss_kib = value
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            } else if let Some(value) = line.strip_prefix("Threads:") {
                threads = value.trim().parse().unwrap_or(0);
            }
        }
        (rss_kib, threads)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0)
    }
}

#[derive(Serialize)]
struct GuiState<'a> {
    schema: u32,
    service: &'static str,
    pid: u32,
    updated_unix: u64,
    status: &'a str,
    db_path: Option<&'a str>,
    generation: u64,
    max_rss_kib: u64,
    threads: u64,
    build: &'static BuildIdentity,
}

pub(crate) fn write_state(status: &str, db_path: Option<&str>, generation: u64) {
    let root = state_root();
    let path = root.join("gui-state.json");
    let temporary = root.join(format!(".gui-state-{}.tmp", std::process::id()));
    let (max_rss_kib, threads) = process_resources();
    let state = GuiState {
        schema: 1,
        service: "tgbackman-gui",
        pid: std::process::id(),
        updated_unix: now_unix(),
        status,
        db_path,
        generation,
        max_rss_kib,
        threads,
        build: build_identity(),
    };
    let result = (|| -> std::io::Result<()> {
        std::fs::create_dir_all(&root)?;
        let bytes = serde_json::to_vec_pretty(&state).map_err(std::io::Error::other)?;
        std::fs::write(&temporary, bytes)?;
        protect_file(&temporary);
        std::fs::rename(&temporary, &path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
}

pub(crate) fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let root = state_root().join("crashes");
        static CRASH_COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = CRASH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "{}-{}-{}.json",
            now_unix_nanos(),
            std::process::id(),
            counter
        ));
        let payload = serde_json::json!({
            "schema": 1,
            "service": "tgbackman-gui",
            "pid": std::process::id(),
            "updated_unix": now_unix(),
            "build": build_identity(),
            "thread": std::thread::current().name().unwrap_or("unnamed"),
            "panic": panic_info.to_string(),
        });
        let temporary = root.join(format!(
            ".{}.{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy(),
            counter
        ));
        let write_result = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(&root)?;
            let bytes = serde_json::to_vec_pretty(&payload).map_err(std::io::Error::other)?;
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            protect_file(&temporary);
            std::fs::rename(&temporary, &path)
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        previous(panic_info);
    }));
}

pub(crate) fn spawn_named<F>(name: &'static str, job: F) -> std::thread::JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(job)
        .expect("failed to create tgbackman worker thread")
}
