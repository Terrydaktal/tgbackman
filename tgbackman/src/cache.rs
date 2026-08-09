use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(crate) fn get_cache_path(db_path: &str) -> String {
    if db_path.ends_with(".db") {
        format!("{}_overlaps.json", &db_path[..db_path.len() - 3])
    } else {
        format!("{}_overlaps.json", db_path)
    }
}
pub(crate) fn get_media_cache_path(db_path: &str) -> String {
    if db_path.ends_with(".db") {
        format!("{}_media_stats.json", &db_path[..db_path.len() - 3])
    } else {
        format!("{}_media_stats.json", db_path)
    }
}
pub(crate) fn get_clusters_cache_path(db_path: &str) -> String {
    if db_path.ends_with(".db") {
        format!("{}_clusters.json", &db_path[..db_path.len() - 3])
    } else {
        format!("{}_clusters.json", db_path)
    }
}
pub(crate) fn newest_database_mtime(path: &Path) -> Option<SystemTime> {
    let mut candidates = vec![path.to_path_buf()];
    let wal = PathBuf::from(format!("{}-wal", path.display()));
    if std::fs::metadata(&wal)
        .map(|meta| meta.len() > 0)
        .unwrap_or(false)
    {
        candidates.push(wal);
    }
    candidates
        .into_iter()
        .filter_map(|candidate| {
            std::fs::metadata(candidate)
                .and_then(|meta| meta.modified())
                .ok()
        })
        .max()
}
pub(crate) fn cache_is_fresh(cache_path: &str, db_path: &str) -> bool {
    let cache_mtime = std::fs::metadata(cache_path)
        .and_then(|meta| meta.modified())
        .ok();
    matches!((cache_mtime, newest_database_mtime(Path::new(db_path))), (Some(cache), Some(db)) if cache >= db)
}

pub(crate) fn secure_cache_file(path: &str) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o600);
            let _ = std::fs::set_permissions(path, permissions);
        }
    }
}
fn local_database_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current) = std::env::current_dir() {
        roots.extend(current.ancestors().map(Path::to_path_buf));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.extend(parent.ancestors().map(Path::to_path_buf));
        }
    }
    roots
        .into_iter()
        .filter(|root| root.join("pyproject.toml").is_file() && root.join("src/tgbackup").is_dir())
        .map(|root| root.join("sqlitedb/telegram_backup.db"))
        .filter(|path| path.is_file())
        .collect()
}

fn removable_database_candidates() -> Vec<PathBuf> {
    let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
    let media_root = PathBuf::from(format!("/media/{}", user));
    if let Ok(volume) = std::env::var("TGBACKMAN_REMOVABLE_VOLUME")
        && !volume.trim().is_empty()
    {
        return vec![media_root.join(volume).join("sqlitedb/telegram_backup.db")];
    }
    let Ok(entries) = std::fs::read_dir(media_root) else {
        return Vec::new();
    };
    let mut candidates = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("sqlitedb/telegram_backup.db"))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
}

pub(crate) fn default_database_path() -> String {
    if let Ok(value) = std::env::var("TGBACKMAN_DB") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    let removable = removable_database_candidates();
    let mut candidates = local_database_candidates();
    candidates.extend(removable.iter().cloned());
    candidates.sort();
    candidates.dedup();
    // The first repository-local candidate is deterministic.  Modification
    // time cannot establish which of two databases is authoritative.
    candidates
        .into_iter()
        .next()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "telegram_backup.db".to_string())
}
