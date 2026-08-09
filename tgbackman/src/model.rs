use std::collections::HashMap;

pub(crate) enum CalcMessage {
    Progress(String),
    Finished(HashMap<String, Vec<String>>),
    Error(String),
}
pub(crate) enum MediaCalcMessage {
    Progress(String),
    Finished(HashMap<String, MediaStats>),
    Error(String),
}
pub(crate) enum LoadMessage {
    Loading(String),
    Finished(Vec<ChatGroup>),
    Error(String),
}
pub(crate) enum CompareMessage {
    Loading(String),
    Finished(ActiveComparison),
    Error(String),
}
pub(crate) enum SingleChatMessage {
    Loading(String),
    Finished(ActiveChatView),
    Error(String),
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct ActiveChatView {
    pub(crate) backup_name: String,
    pub(crate) chat_id: String,
    pub(crate) messages: Vec<BackupMessage>,
    pub(crate) total_messages: i64,
    pub(crate) truncated: bool,
    pub(crate) scroll_to_bottom: bool,
    pub(crate) search_query: String,
    pub(crate) filtered_indices: Vec<usize>,
    pub(crate) search_matches_count: usize,
    pub(crate) current_search_match_idx: Option<usize>,
    pub(crate) scroll_to_row_idx: Option<usize>,
}
#[derive(Clone)]
pub(crate) struct BackupMessage {
    pub(crate) message_id: i64,
    pub(crate) sender: String,
    pub(crate) timestamp_unix: i64,
    pub(crate) timestamp_str: String,
    pub(crate) text: String,
    pub(crate) clean_text: String,
    pub(crate) media_type: Option<String>,
    pub(crate) media_path: Option<String>,
}
#[derive(Clone)]
pub(crate) struct AlignedMessageRow {
    pub(crate) msg_a: Option<BackupMessage>,
    pub(crate) msg_b: Option<BackupMessage>,
    pub(crate) is_discrepancy: bool,
}
#[derive(Clone)]
pub(crate) struct ActiveComparison {
    pub(crate) backup_a_letter: char,
    pub(crate) backup_b_letter: char,
    pub(crate) backup_a_name: String,
    pub(crate) backup_b_name: String,
    pub(crate) rows: Vec<AlignedMessageRow>,
    pub(crate) discrepancies: Vec<usize>,
    pub(crate) current_discrepancy_idx: Option<usize>,
    pub(crate) scroll_to_row_idx: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct BackupInfo {
    pub(crate) chat_id: String,
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) min_id: Option<i64>,
    pub(crate) max_id: Option<i64>,
    pub(crate) count: i64,
    pub(crate) min_ts: String,
    pub(crate) max_ts: String,
    pub(crate) min_unix: Option<i64>,
    pub(crate) max_unix: Option<i64>,
    pub(crate) is_active: bool,
    pub(crate) is_blacklisted: bool,
    pub(crate) last_backup_unix: Option<i64>,
    pub(crate) last_backup_run_unix: Option<i64>,
    pub(crate) last_backup_run_status: String,
    pub(crate) last_backup_source: String,
    pub(crate) last_backup_confidence: String,
    pub(crate) last_backup_evidence: String,
    pub(crate) media_stats: Option<MediaStats>,
}

impl BackupInfo {
    pub(crate) fn compute_media_stats(&self, db_path: &str) -> MediaStats {
        let mut stats = MediaStats::default();
        if let Ok(conn) = rusqlite::Connection::open(db_path) {
            if let Ok(mut stmt) = conn.prepare("SELECT media_type, media_path FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0 AND media_type IN ('photo', 'video', 'voice_message', 'file')") {
                if let Ok(rows) = stmt.query_map(rusqlite::params![self.chat_id], |row| {
                    Ok((row.get::<_, Option<String>>(0)?, row.get::<_, Option<String>>(1)?))
                }) {
                    for row in rows.flatten() {
                        let (media_type, media_path) = row;
                        let resolved = media_path.as_deref().is_some_and(|mp| {
                            if mp.is_empty() { return false; }
                            let base = std::path::Path::new(&self.path);
                            if base.join(mp).exists() { return true; }
                            let normalized = mp.replace('\\', "/");
                            let parts: Vec<&str> = normalized.split('/').collect();
                            let known = ["photos", "video_files", "voice_messages", "audio_files", "stickers", "sticker_files", "files", "documents", "animations"];
                            if parts.iter().enumerate().rev().any(|(i, part)| known.contains(&part.to_lowercase().as_str()) && base.join(parts[i..].join("/")).exists()) { return true; }
                            std::path::Path::new(mp).file_name().is_some_and(|name| base.join(name).exists() || base.join("files").join(name).exists())
                        });
                        match media_type.as_deref() {
                            Some("photo") => { stats.photos_count += 1; if resolved { stats.photos_resolved += 1; } }
                            Some("video") => { stats.videos_count += 1; if resolved { stats.videos_resolved += 1; } }
                            Some("voice_message") => { stats.voice_count += 1; if resolved { stats.voice_resolved += 1; } }
                            Some("file") => { stats.files_count += 1; if resolved { stats.files_resolved += 1; } }
                            _ => {}
                        }
                    }
                }
            }
        }
        stats
    }
}

#[derive(Clone)]
pub(crate) struct ChatGroup {
    pub(crate) name: String,
    pub(crate) max_count: i64,
    pub(crate) backups: Vec<BackupInfo>,
}

impl ChatGroup {
    pub(crate) fn is_active(&self) -> bool {
        self.backups.iter().any(|b| b.is_active)
    }
    pub(crate) fn is_blacklisted(&self) -> bool {
        self.backups.iter().any(|b| b.is_blacklisted)
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct MediaStats {
    pub(crate) photos_count: i64,
    pub(crate) photos_resolved: i64,
    pub(crate) videos_count: i64,
    pub(crate) videos_resolved: i64,
    pub(crate) voice_count: i64,
    pub(crate) voice_resolved: i64,
    pub(crate) files_count: i64,
    pub(crate) files_resolved: i64,
}
