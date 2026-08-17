//! GUI controller state and background-task orchestration.

use eframe::egui;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::cache::{
    cache_is_fresh, default_database_path, get_cache_path, get_inventory_cache_path,
    get_media_cache_path, secure_cache_file,
};
use crate::database::{
    MESSAGE_SEARCH_PAGE_SIZE, count_search_messages, load_chat_page_with_aliases,
    load_search_results_by_rowids, run_inventory, search_message_page, search_message_rowids,
};
use crate::diagnostics;
use crate::matching::{clean_text_for_match, count_missing_messages, format_unix_to_ts};
use crate::model::{
    ActiveChatView, ActiveComparison, AlignedMessageRow, BackupMessage, CalcMessage, ChatGroup,
    ChatPageRequest, CompareMessage, LoadMessage, MediaCalcMessage, MediaStats,
    MessageSearchMessage, MessageSearchResult, SingleChatMessage,
};

fn ensure_column(
    conn: &rusqlite::Connection,
    table: &str,
    column: &str,
    sql: &str,
) -> rusqlite::Result<()> {
    let present: i64 = conn.query_row(
        &format!(
            "SELECT count(*) FROM pragma_table_info('{}') WHERE name = ?1",
            table.replace('\'', "''")
        ),
        [column],
        |row| row.get(0),
    )?;
    if present == 0 {
        conn.execute(sql, [])?;
    }
    Ok(())
}

fn atomic_write_json<T: Serialize>(path: &str, value: &T) -> std::io::Result<()> {
    let temporary = format!("{}.tmp-{}", path, std::process::id());
    let result = (|| {
        let file = std::fs::File::create(&temporary)?;
        serde_json::to_writer_pretty(file, value).map_err(std::io::Error::other)?;
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn open_readonly_database(path: &str) -> rusqlite::Result<rusqlite::Connection> {
    rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
            | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
}

const INVENTORY_CACHE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct InventoryCache {
    version: u32,
    groups: Vec<ChatGroup>,
}

fn load_inventory_cache(cache_path: &str, db_path: &str) -> Option<Vec<ChatGroup>> {
    if !cache_is_fresh(cache_path, db_path) {
        return None;
    }
    let file = std::fs::File::open(cache_path).ok()?;
    let cache: InventoryCache = serde_json::from_reader(file).ok()?;
    (cache.version == INVENTORY_CACHE_VERSION).then_some(cache.groups)
}

pub(crate) struct OverlapApp {
    pub(crate) db_path: String,
    pub(crate) loaded_db_path: Option<String>,
    pub(crate) loading_db_path: Option<String>,
    pub(crate) db_generation: u64,
    pub(crate) groups: Vec<ChatGroup>,
    pub(crate) filtered_groups: Vec<usize>,
    pub(crate) selected_group_idx: Option<usize>,
    pub(crate) comparison_results: Vec<String>,
    pub(crate) status_msg: String,
    pub(crate) search_query: String,
    pub(crate) calculating_overlaps: bool,
    pub(crate) rx: Option<std::sync::mpsc::Receiver<CalcMessage>>,
    pub(crate) cached_results: HashMap<String, Vec<String>>,
    pub(crate) calculating_media: bool,
    pub(crate) media_rx: Option<std::sync::mpsc::Receiver<MediaCalcMessage>>,
    pub(crate) loading_data: bool,
    pub(crate) load_rx: Option<std::sync::mpsc::Receiver<LoadMessage>>,
    pub(crate) active_comparison: Arc<Mutex<Option<ActiveComparison>>>,
    pub(crate) loading_comparison: bool,
    pub(crate) compare_rx: Option<std::sync::mpsc::Receiver<CompareMessage>>,
    pub(crate) active_chat_view: Arc<Mutex<Option<ActiveChatView>>>,
    pub(crate) loading_chat_view: bool,
    pub(crate) chat_view_rx: Option<std::sync::mpsc::Receiver<SingleChatMessage>>,
    pub(crate) global_search_results: HashMap<usize, MessageSearchResult>,
    pub(crate) global_search_row_ids: Vec<i64>,
    pub(crate) global_search_total_matches: usize,
    pub(crate) global_searching: bool,
    pub(crate) global_search_error: Option<String>,
    pub(crate) global_search_rx: Option<std::sync::mpsc::Receiver<MessageSearchMessage>>,
    pub(crate) global_search_request_id: u64,
    pub(crate) global_search_scheduled_at: Option<Instant>,
    pub(crate) global_search_last_submitted: String,
    pub(crate) chat_search_rx: Option<std::sync::mpsc::Receiver<MessageSearchMessage>>,
    pub(crate) chat_search_request_id: u64,
    pub(crate) pending_chat_highlight_query: Option<String>,
}

impl Default for OverlapApp {
    fn default() -> Self {
        OverlapApp {
            db_path: default_database_path(),
            loaded_db_path: None,
            loading_db_path: None,
            db_generation: 0,
            groups: Vec::new(),
            filtered_groups: Vec::new(),
            selected_group_idx: None,
            comparison_results: Vec::new(),
            status_msg: "Database not loaded yet. Click 'Load Database' to begin.".to_string(),
            search_query: "".to_string(),
            calculating_overlaps: false,
            rx: None,
            cached_results: HashMap::new(),
            calculating_media: false,
            media_rx: None,
            loading_data: false,
            load_rx: None,
            active_comparison: Arc::new(Mutex::new(None)),
            loading_comparison: false,
            compare_rx: None,
            active_chat_view: Arc::new(Mutex::new(None)),
            loading_chat_view: false,
            chat_view_rx: None,
            global_search_results: HashMap::new(),
            global_search_row_ids: Vec::new(),
            global_search_total_matches: 0,
            global_searching: false,
            global_search_error: None,
            global_search_rx: None,
            global_search_request_id: 0,
            global_search_scheduled_at: None,
            global_search_last_submitted: String::new(),
            chat_search_rx: None,
            chat_search_request_id: 0,
            pending_chat_highlight_query: None,
        }
    }
}

impl Drop for OverlapApp {
    fn drop(&mut self) {
        diagnostics::write_state(
            "stopped",
            self.loaded_db_path.as_deref(),
            self.db_generation,
        );
    }
}
impl OverlapApp {
    fn group_cache_key(group: &crate::model::ChatGroup) -> String {
        let mut ids: Vec<&str> = group
            .backups
            .iter()
            .map(|backup| backup.chat_id.as_str())
            .collect();
        ids.sort_unstable();
        ids.join("\0")
    }

    pub(crate) fn active_db_path(&self) -> String {
        self.loaded_db_path
            .clone()
            .unwrap_or_else(|| self.db_path.clone())
    }

    pub(crate) fn load_cache(&mut self) {
        self.cached_results.clear();
        let db_path = self.active_db_path();
        let cache_path = get_cache_path(&db_path);
        if cache_is_fresh(&cache_path, &db_path) {
            if let Ok(file) = std::fs::File::open(&cache_path) {
                if let Ok(cache) = serde_json::from_reader(file) {
                    self.cached_results = cache;
                    self.status_msg = format!(
                        "Loaded database and found cached overlaps for {} groups.",
                        self.cached_results.len()
                    );
                    return;
                }
            }
        }
        self.status_msg =
            "Database loaded. No cached overlaps found. Click '🔄 Recompute All Overlaps'."
                .to_string();
    }

    pub(crate) fn load_media_cache(&mut self) {
        for group in &mut self.groups {
            for backup in &mut group.backups {
                backup.media_stats = None;
            }
        }
        let db_path = self.active_db_path();
        let cache_path = get_media_cache_path(&db_path);
        if cache_is_fresh(&cache_path, &db_path) {
            if let Ok(file) = std::fs::File::open(&cache_path) {
                if let Ok(cache) = serde_json::from_reader::<_, HashMap<String, MediaStats>>(file) {
                    for group in &mut self.groups {
                        for b in &mut group.backups {
                            let key = format!("{}:{}", b.chat_id, b.path);
                            if let Some(stats) = cache.get(&key) {
                                b.media_stats = Some(stats.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn recompute_all_media_stats(&mut self, ctx: egui::Context) {
        self.calculating_media = true;
        self.status_msg = "Starting media stats calculation...".to_string();

        let db_path = self.active_db_path();
        let groups = self.groups.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.media_rx = Some(rx);

        diagnostics::spawn_named("media-stats", move || {
            let mut all_stats = HashMap::new();

            // Collect all backups
            let mut backups_to_calc = Vec::new();
            for group in &groups {
                for b in &group.backups {
                    backups_to_calc.push(b.clone());
                }
            }

            let total = backups_to_calc.len();
            for (idx, b) in backups_to_calc.iter().enumerate() {
                let msg = format!(
                    "Recomputing media stats: {} of {} ({})",
                    idx + 1,
                    total,
                    b.name
                );
                let _ = tx.send(MediaCalcMessage::Progress(msg));
                ctx.request_repaint();

                let stats = b.compute_media_stats(&db_path);
                let key = format!("{}:{}", b.chat_id, b.path);
                all_stats.insert(key, stats);
            }

            // Write cache to file
            let cache_path = get_media_cache_path(&db_path);
            if let Err(e) = atomic_write_json(&cache_path, &all_stats) {
                let _ = tx.send(MediaCalcMessage::Error(format!(
                    "Failed to write cache: {}",
                    e
                )));
                ctx.request_repaint();
                return;
            }
            secure_cache_file(&cache_path);

            let _ = tx.send(MediaCalcMessage::Finished(all_stats));
            ctx.request_repaint();
        });
    }

    pub(crate) fn select_group(&mut self, idx: usize) {
        self.selected_group_idx = Some(idx);
        let group = &self.groups[idx];
        if group.backups.len() < 2 {
            self.comparison_results = vec![
                "Only 1 backup available in this group. No overlap analysis needed.".to_string(),
            ];
        } else if let Some(results) = self.cached_results.get(&Self::group_cache_key(group)) {
            self.comparison_results = results.clone();
        } else {
            self.comparison_results = vec![
                "No cached overlaps found for this chat.".to_string(),
                "Click '🔄 Recompute All Overlaps' to calculate and cache results.".to_string(),
            ];
        }
    }

    pub(crate) fn recompute_all_overlaps(&mut self, ctx: egui::Context) {
        self.calculating_overlaps = true;
        self.status_msg = "Starting overlaps calculation...".to_string();

        let db_path = self.active_db_path();
        let groups = self.groups.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.rx = Some(rx);

        diagnostics::spawn_named("overlap-calculation", move || {
            let conn = match rusqlite::Connection::open(&db_path) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(CalcMessage::Error(format!("Failed to open DB: {}", e)));
                    ctx.request_repaint();
                    return;
                }
            };

            let _ = conn.execute("PRAGMA cache_size = -1048576;", []);
            let _ = conn.execute("PRAGMA temp_store = MEMORY;", []);

            let mut all_results = HashMap::new();
            let groups_to_calc: Vec<&ChatGroup> =
                groups.iter().filter(|g| g.backups.len() >= 2).collect();
            let total_to_calc = groups_to_calc.len();

            for (idx, group) in groups_to_calc.into_iter().enumerate() {
                let msg = format!(
                    "Recomputing: group {} of {} ({})",
                    idx + 1,
                    total_to_calc,
                    group.name
                );
                let _ = tx.send(CalcMessage::Progress(msg));
                ctx.request_repaint();

                let mut results = Vec::new();
                for i in 0..group.backups.len() {
                    for j in i + 1..group.backups.len() {
                        let a = &group.backups[i];
                        let b = &group.backups[j];

                        let letter_a = (b'A' + i as u8) as char;
                        let letter_b = (b'A' + j as u8) as char;

                        if let (Some(a_min), Some(a_max), Some(b_min), Some(b_max)) =
                            (a.min_unix, a.max_unix, b.min_unix, b.max_unix)
                        {
                            let mut a_contains_b = false;
                            let mut b_contains_a = false;
                            if a_min <= b_min + 86400 && a_max >= b_max - 86400 {
                                a_contains_b = true;
                            } else if b_min <= a_min + 86400 && b_max >= a_max - 86400 {
                                b_contains_a = true;
                            }

                            let overlap_start = a_min.max(b_min);
                            let overlap_end = a_max.min(b_max);

                            if overlap_start <= overlap_end {
                                let overlap_days = (overlap_end - overlap_start) as f64 / 86400.0;
                                let relationship = if b_contains_a {
                                    format!(
                                        "Backup {} fully contains the chronological span of Backup {}!",
                                        letter_b, letter_a
                                    )
                                } else if a_contains_b {
                                    format!(
                                        "Backup {} fully contains the chronological span of Backup {}!",
                                        letter_a, letter_b
                                    )
                                } else {
                                    format!(
                                        "Backup {} and Backup {} overlap chronologically by {:.1} days!",
                                        letter_a, letter_b, overlap_days
                                    )
                                };

                                results.push(format!("⚖️ {}", relationship));
                                results.push(format!(
                                    "    Overlap span: {} to {}",
                                    format_unix_to_ts(overlap_start),
                                    format_unix_to_ts(overlap_end)
                                ));

                                match count_missing_messages(&conn, &a.chat_id, &b.chat_id, overlap_start, overlap_end) {
                                    Ok(missing_a_in_b) => {
                                        match count_missing_messages(&conn, &b.chat_id, &a.chat_id, overlap_start, overlap_end) {
                                            Ok(missing_b_in_a) => {
                                                if missing_a_in_b == 0 && missing_b_in_a == 0 {
                                                    results.push("    ✅ SUCCESS: Perfect alignment! 0 missing messages in the overlapping region for both backups.".to_string());
                                                } else {
                                                    if missing_a_in_b > 0 {
                                                        results.push(format!("    ⚠️ WARNING: Backup {} is missing {} individual messages that exist in Backup {}'s range!", letter_b, missing_a_in_b, letter_a));
                                                    }
                                                    if missing_b_in_a > 0 {
                                                        results.push(format!("    ⚠️ WARNING: Backup {} is missing {} individual messages that exist in Backup {}'s range!", letter_a, missing_b_in_a, letter_b));
                                                    }
                                                }
                                            }
                                            Err(e) => results.push(format!("    ❌ ERROR: Failed to count missing messages for Backup A: {}", e))
                                        }
                                    }
                                    Err(e) => results.push(format!("    ❌ ERROR: Failed to count missing messages for Backup B: {}", e))
                                }
                            } else {
                                results.push(format!(
                                    "⚠️ Backup {} and Backup {} do not overlap chronologically.",
                                    letter_a, letter_b
                                ));
                            }
                        }
                        results.push("".to_string()); // Divider
                    }
                }
                all_results.insert(Self::group_cache_key(group), results);
            }

            // Write cache to file
            let cache_path = get_cache_path(&db_path);
            if atomic_write_json(&cache_path, &all_results).is_ok() {
                secure_cache_file(&cache_path);
            }

            let _ = tx.send(CalcMessage::Finished(all_results));
            ctx.request_repaint();
        });
    }

    pub(crate) fn filter_groups(&mut self) {
        if self.search_query.is_empty() {
            self.filtered_groups = (0..self.groups.len()).collect();
        } else {
            let query = self.search_query.to_lowercase();
            self.filtered_groups = self
                .groups
                .iter()
                .enumerate()
                .filter(|(_, g)| g.name.to_lowercase().contains(&query))
                .map(|(idx, _)| idx)
                .collect();
        }
    }

    pub(crate) fn trigger_load_data(&mut self, ctx: egui::Context) {
        self.loading_data = true;
        self.status_msg = "Opening database...".to_string();
        self.groups.clear();
        self.filtered_groups.clear();
        self.selected_group_idx = None;
        self.comparison_results.clear();
        self.rx = None;
        self.media_rx = None;
        self.compare_rx = None;
        self.chat_view_rx = None;
        self.global_search_rx = None;
        self.chat_search_rx = None;
        self.calculating_overlaps = false;
        self.calculating_media = false;
        self.loading_comparison = false;
        self.loading_chat_view = false;
        self.global_searching = false;
        self.global_search_results.clear();
        self.global_search_row_ids.clear();
        self.global_search_total_matches = 0;
        self.global_search_error = None;
        self.global_search_scheduled_at = None;
        self.global_search_last_submitted.clear();
        self.pending_chat_highlight_query = None;
        self.active_comparison = Arc::new(Mutex::new(None));
        self.active_chat_view = Arc::new(Mutex::new(None));

        let db_path = self.db_path.clone();
        self.loading_db_path = Some(db_path.clone());
        self.db_generation = self.db_generation.wrapping_add(1);
        diagnostics::write_state("loading_database", Some(&db_path), self.db_generation);
        let (tx, rx) = std::sync::mpsc::channel();
        self.load_rx = Some(rx);

        diagnostics::spawn_named("load-database", move || {
            let _ = tx.send(LoadMessage::Loading("Connecting to SQLite...".to_string()));
            ctx.request_repaint();

            let conn = match rusqlite::Connection::open(&db_path) {
                Ok(c) => {
                    let _ = c.execute("PRAGMA cache_size = -1048576;", []);
                    let _ = c.execute("PRAGMA temp_store = MEMORY;", []);
                    for (table, column, sql) in [
                        (
                            "chats",
                            "is_active",
                            "ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0",
                        ),
                        (
                            "chats",
                            "last_backup_unix",
                            "ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER",
                        ),
                        (
                            "chats",
                            "last_backup_run_unix",
                            "ALTER TABLE chats ADD COLUMN last_backup_run_unix INTEGER",
                        ),
                        (
                            "chats",
                            "last_backup_run_status",
                            "ALTER TABLE chats ADD COLUMN last_backup_run_status TEXT",
                        ),
                        (
                            "chats",
                            "min_msg_id",
                            "ALTER TABLE chats ADD COLUMN min_msg_id INTEGER",
                        ),
                        (
                            "chats",
                            "max_msg_id",
                            "ALTER TABLE chats ADD COLUMN max_msg_id INTEGER",
                        ),
                        (
                            "chats",
                            "msg_count",
                            "ALTER TABLE chats ADD COLUMN msg_count INTEGER",
                        ),
                        (
                            "chats",
                            "min_timestamp",
                            "ALTER TABLE chats ADD COLUMN min_timestamp TEXT",
                        ),
                        (
                            "chats",
                            "max_timestamp",
                            "ALTER TABLE chats ADD COLUMN max_timestamp TEXT",
                        ),
                        (
                            "chats",
                            "min_timestamp_unix",
                            "ALTER TABLE chats ADD COLUMN min_timestamp_unix INTEGER",
                        ),
                        (
                            "chats",
                            "max_timestamp_unix",
                            "ALTER TABLE chats ADD COLUMN max_timestamp_unix INTEGER",
                        ),
                        (
                            "messages",
                            "is_deleted",
                            "ALTER TABLE messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
                        ),
                        (
                            "messages",
                            "deleted_unix",
                            "ALTER TABLE messages ADD COLUMN deleted_unix INTEGER",
                        ),
                        (
                            "messages",
                            "reply_to_chat_id",
                            "ALTER TABLE messages ADD COLUMN reply_to_chat_id TEXT",
                        ),
                        (
                            "messages",
                            "reply_to_peer_kind",
                            "ALTER TABLE messages ADD COLUMN reply_to_peer_kind TEXT",
                        ),
                        (
                            "messages",
                            "reply_to_peer_id",
                            "ALTER TABLE messages ADD COLUMN reply_to_peer_id INTEGER",
                        ),
                        (
                            "messages",
                            "reply_to_top_id",
                            "ALTER TABLE messages ADD COLUMN reply_to_top_id INTEGER",
                        ),
                        (
                            "messages",
                            "reply_to_story_id",
                            "ALTER TABLE messages ADD COLUMN reply_to_story_id INTEGER",
                        ),
                        (
                            "messages",
                            "reply_quote_text",
                            "ALTER TABLE messages ADD COLUMN reply_quote_text TEXT",
                        ),
                        (
                            "messages",
                            "reply_quote_entities_json",
                            "ALTER TABLE messages ADD COLUMN reply_quote_entities_json TEXT",
                        ),
                        (
                            "messages",
                            "reply_quote_offset",
                            "ALTER TABLE messages ADD COLUMN reply_quote_offset INTEGER",
                        ),
                        (
                            "messages",
                            "reply_media_json",
                            "ALTER TABLE messages ADD COLUMN reply_media_json TEXT",
                        ),
                    ] {
                        if let Err(error) = ensure_column(&c, table, column, sql) {
                            let _ = tx.send(LoadMessage::Error(format!(
                                "Database migration failed for {table}.{column}: {error}"
                            )));
                            ctx.request_repaint();
                            return;
                        }
                    }
                    if let Err(error) = c.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_chat_ts_id
                         ON messages(chat_id, timestamp_unix, message_id)",
                        [],
                    ) {
                        let _ = tx.send(LoadMessage::Error(format!(
                            "Database index migration failed: {error}"
                        )));
                        ctx.request_repaint();
                        return;
                    }
                    c
                }
                Err(e) => {
                    let _ = tx.send(LoadMessage::Error(format!("Failed to open DB: {}", e)));
                    ctx.request_repaint();
                    return;
                }
            };

            let _ = tx.send(LoadMessage::Loading(
                "Loading cached database inventory...".to_string(),
            ));
            ctx.request_repaint();

            let inventory_cache_path = get_inventory_cache_path(&db_path);
            if let Some(groups) = load_inventory_cache(&inventory_cache_path, &db_path) {
                let _ = tx.send(LoadMessage::Finished(groups));
                ctx.request_repaint();
                return;
            }

            let _ = tx.send(LoadMessage::Loading(
                "Refreshing changed database inventory...".to_string(),
            ));
            ctx.request_repaint();

            match run_inventory(&conn, &db_path) {
                Ok(groups) => {
                    let cache = InventoryCache {
                        version: INVENTORY_CACHE_VERSION,
                        groups: groups.clone(),
                    };
                    if atomic_write_json(&inventory_cache_path, &cache).is_ok() {
                        secure_cache_file(&inventory_cache_path);
                    }
                    let _ = tx.send(LoadMessage::Finished(groups));
                }
                Err(e) => {
                    let _ = tx.send(LoadMessage::Error(format!(
                        "Failed to run inventory: {}",
                        e
                    )));
                }
            }
            ctx.request_repaint();
        });
    }

    pub(crate) fn trigger_comparison(&mut self, idx_a: usize, idx_b: usize, ctx: egui::Context) {
        if let Some(group_idx) = self.selected_group_idx {
            let group = &self.groups[group_idx];
            let backup_a = &group.backups[idx_a];
            let backup_b = &group.backups[idx_b];

            let letter_a = (b'A' + idx_a as u8) as char;
            let letter_b = (b'A' + idx_b as u8) as char;

            self.loading_comparison = true;
            self.status_msg = format!(
                "Loading side-by-side messages for {} vs {}...",
                letter_a, letter_b
            );

            let db_path = self.active_db_path();

            let chat_a_id = backup_a.chat_id.clone();
            let chat_b_id = backup_b.chat_id.clone();

            let min_a = backup_a.min_unix.unwrap_or(0);
            let max_a = backup_a.max_unix.unwrap_or(0);
            let min_b = backup_b.min_unix.unwrap_or(0);
            let max_b = backup_b.max_unix.unwrap_or(0);

            let start_unix = min_a.max(min_b);
            let end_unix = max_a.min(max_b);

            let backup_a_name = format!("Backup {} ({})", letter_a, backup_a.name);
            let backup_b_name = format!("Backup {} ({})", letter_b, backup_b.name);

            let (tx, rx) = std::sync::mpsc::channel();
            self.compare_rx = Some(rx);

            diagnostics::spawn_named("comparison-load", move || {
                let _ = tx.send(CompareMessage::Loading(
                    "Connecting to database...".to_string(),
                ));
                ctx.request_repaint();

                let conn = match rusqlite::Connection::open(&db_path) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Failed to open DB: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading(
                    "Fetching Backup A messages in overlap range...".to_string(),
                ));
                ctx.request_repaint();

                let mut stmt_a = match conn.prepare(
                    "SELECT message_id, sender, timestamp_unix, timestamp, text, media_type, media_path FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0 AND timestamp_unix BETWEEN ? AND ? ORDER BY timestamp_unix ASC, message_id ASC"
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Prepare query A failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let rows_a = match stmt_a.query_map(
                    rusqlite::params![chat_a_id, start_unix, end_unix],
                    |r| {
                        let text: String = r.get(4)?;
                        let clean_text = clean_text_for_match(&text);
                        Ok(BackupMessage {
                            message_id: r.get(0)?,
                            sender: r.get(1)?,
                            timestamp_unix: r.get(2)?,
                            timestamp_str: r.get(3)?,
                            text,
                            clean_text,
                            media_type: r.get(5)?,
                            media_path: r.get(6)?,
                            reply: None,
                            forwarded_from: None,
                            edit_timestamp: None,
                            reactions_json: None,
                            message_type: None,
                            action_json: None,
                            is_outgoing: false,
                        })
                    },
                ) {
                    Ok(mapped) => {
                        let mut msgs = Vec::new();
                        for item in mapped {
                            match item {
                                Ok(m) => msgs.push(m),
                                Err(error) => {
                                    let _ = tx.send(CompareMessage::Error(format!(
                                        "Row decode A failed: {error}"
                                    )));
                                    ctx.request_repaint();
                                    return;
                                }
                            }
                        }
                        msgs
                    }
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Query A failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading(
                    "Fetching Backup B messages in overlap range...".to_string(),
                ));
                ctx.request_repaint();

                let mut stmt_b = match conn.prepare(
                    "SELECT message_id, sender, timestamp_unix, timestamp, text, media_type, media_path FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0 AND timestamp_unix BETWEEN ? AND ? ORDER BY timestamp_unix ASC, message_id ASC"
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Prepare query B failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let rows_b = match stmt_b.query_map(
                    rusqlite::params![chat_b_id, start_unix - 3605, end_unix + 3605],
                    |r| {
                        let text: String = r.get(4)?;
                        let clean_text = clean_text_for_match(&text);
                        Ok(BackupMessage {
                            message_id: r.get(0)?,
                            sender: r.get(1)?,
                            timestamp_unix: r.get(2)?,
                            timestamp_str: r.get(3)?,
                            text,
                            clean_text,
                            media_type: r.get(5)?,
                            media_path: r.get(6)?,
                            reply: None,
                            forwarded_from: None,
                            edit_timestamp: None,
                            reactions_json: None,
                            message_type: None,
                            action_json: None,
                            is_outgoing: false,
                        })
                    },
                ) {
                    Ok(mapped) => {
                        let mut msgs = Vec::new();
                        for item in mapped {
                            match item {
                                Ok(m) => msgs.push(m),
                                Err(error) => {
                                    let _ = tx.send(CompareMessage::Error(format!(
                                        "Row decode B failed: {error}"
                                    )));
                                    ctx.request_repaint();
                                    return;
                                }
                            }
                        }
                        msgs
                    }
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Query B failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading(
                    "Aligning message streams side-by-side...".to_string(),
                ));
                ctx.request_repaint();

                let mut matched_b_ids = HashSet::new();
                let mut aligned_rows = Vec::new();

                // Group B messages by timestamp for O(1) candidate lookup
                let mut b_by_ts: HashMap<i64, Vec<BackupMessage>> = HashMap::new();
                for b in &rows_b {
                    b_by_ts.entry(b.timestamp_unix).or_default().push(b.clone());
                }

                for a in &rows_a {
                    let mut matched_b: Option<BackupMessage> = None;
                    let is_a_media = a.clean_text.starts_with('[') && a.clean_text.ends_with(']');

                    'outer: for &offset in &[0, 3600, -3600] {
                        let target_ts = a.timestamp_unix + offset;
                        for candidate_ts in (target_ts - 5)..=(target_ts + 5) {
                            if let Some(candidates) = b_by_ts.get(&candidate_ts) {
                                for b in candidates {
                                    if matched_b_ids.contains(&b.message_id) {
                                        continue;
                                    }
                                    let is_b_media = b.clean_text.starts_with('[')
                                        && b.clean_text.ends_with(']');

                                    if a.clean_text == b.clean_text
                                        || (a.clean_text.is_empty() && is_b_media)
                                        || (b.clean_text.is_empty() && is_a_media)
                                    {
                                        matched_b = Some(b.clone());
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }

                    if let Some(b) = matched_b {
                        matched_b_ids.insert(b.message_id);
                        aligned_rows.push(AlignedMessageRow {
                            msg_a: Some(a.clone()),
                            msg_b: Some(b),
                            is_discrepancy: false,
                        });
                    } else {
                        aligned_rows.push(AlignedMessageRow {
                            msg_a: Some(a.clone()),
                            msg_b: None,
                            is_discrepancy: true,
                        });
                    }
                }

                for b in &rows_b {
                    if !matched_b_ids.contains(&b.message_id) {
                        if b.timestamp_unix >= start_unix && b.timestamp_unix <= end_unix {
                            aligned_rows.push(AlignedMessageRow {
                                msg_a: None,
                                msg_b: Some(b.clone()),
                                is_discrepancy: true,
                            });
                        }
                    }
                }

                aligned_rows.sort_by_key(|r| {
                    r.msg_a
                        .as_ref()
                        .or(r.msg_b.as_ref())
                        .map(|m| m.timestamp_unix)
                        .unwrap_or(0)
                });

                let discrepancies: Vec<usize> = aligned_rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.is_discrepancy)
                    .map(|(idx, _)| idx)
                    .collect();

                let active_comp = ActiveComparison {
                    backup_a_letter: letter_a,
                    backup_b_letter: letter_b,
                    backup_a_name,
                    backup_b_name,
                    rows: aligned_rows,
                    current_discrepancy_idx: if discrepancies.is_empty() {
                        None
                    } else {
                        Some(0)
                    },
                    scroll_to_row_idx: if discrepancies.is_empty() {
                        None
                    } else {
                        Some(discrepancies[0])
                    },
                    discrepancies,
                };

                let _ = tx.send(CompareMessage::Finished(active_comp));
                ctx.request_repaint();
            });
        }
    }

    pub(crate) fn trigger_load_chat(&mut self, idx: usize, ctx: egui::Context) {
        let Some(group_idx) = self.selected_group_idx else {
            return;
        };
        let Some(group) = self.groups.get(group_idx) else {
            return;
        };
        let Some(backup) = group.backups.get(idx) else {
            return;
        };
        self.trigger_load_chat_page(
            backup.chat_id.clone(),
            group.name.clone(),
            ChatPageRequest::Latest,
            ctx,
        );
    }

    pub(crate) fn trigger_load_preferred_chat(&mut self, group_idx: usize, ctx: egui::Context) {
        let Some(group) = self.groups.get(group_idx) else {
            return;
        };
        let Some((backup_idx, _)) = group
            .backups
            .iter()
            .enumerate()
            .filter(|(_, backup)| backup.count > 0)
            .max_by_key(|(_, backup)| backup.count)
        else {
            self.status_msg = format!("{} has no saved messages yet.", group.name);
            return;
        };
        self.selected_group_idx = Some(group_idx);
        self.trigger_load_chat(backup_idx, ctx);
    }

    pub(crate) fn trigger_load_chat_page(
        &mut self,
        chat_id: String,
        backup_name: String,
        request: ChatPageRequest,
        ctx: egui::Context,
    ) {
        let cached_self_sender_aliases = self.active_chat_view.lock().ok().and_then(|view| {
            view.as_ref()
                .filter(|view| view.chat_id == chat_id)
                .map(|view| view.self_sender_aliases.clone())
        });
        let switching_chat = self
            .active_chat_view
            .lock()
            .ok()
            .and_then(|view| view.as_ref().map(|view| view.chat_id != chat_id))
            .unwrap_or(true);
        if switching_chat {
            if let Ok(mut view) = self.active_chat_view.lock() {
                *view = None;
            }
            self.chat_search_rx = None;
        }
        self.loading_chat_view = true;
        self.status_msg = format!("Loading messages for {backup_name}...");
        let db_path = self.active_db_path();
        let (tx, rx) = std::sync::mpsc::channel();
        self.chat_view_rx = Some(rx);

        diagnostics::spawn_named("chat-load", move || {
            let _ = tx.send(SingleChatMessage::Loading(
                "Fetching a bounded message page from SQLite...".to_string(),
            ));
            ctx.request_repaint();
            let result = open_readonly_database(&db_path).and_then(|conn| {
                let transaction = conn.unchecked_transaction()?;
                let page = load_chat_page_with_aliases(
                    &transaction,
                    &chat_id,
                    backup_name,
                    request,
                    cached_self_sender_aliases.as_ref(),
                )?;
                transaction.commit()?;
                Ok(page)
            });
            match result {
                Ok(page) => {
                    let _ = tx.send(SingleChatMessage::Finished(page));
                }
                Err(error) => {
                    let _ = tx.send(SingleChatMessage::Error(format!(
                        "Message query failed: {error}"
                    )));
                }
            }
            ctx.request_repaint();
        });
    }

    pub(crate) fn trigger_global_message_search(
        &mut self,
        query: String,
        offset: usize,
        ctx: egui::Context,
    ) {
        let query = query.trim().to_string();
        self.global_search_scheduled_at = None;
        if query.is_empty() {
            self.global_search_request_id = self.global_search_request_id.wrapping_add(1);
            self.global_search_last_submitted.clear();
            self.global_search_results.clear();
            self.global_search_row_ids.clear();
            self.global_search_total_matches = 0;
            self.global_search_error = None;
            self.global_searching = false;
            self.global_search_rx = None;
            return;
        }
        let new_search = offset == 0 || query != self.global_search_last_submitted;
        if !new_search && self.global_searching {
            return;
        }
        if new_search {
            self.global_search_request_id = self.global_search_request_id.wrapping_add(1);
            self.global_search_last_submitted.clone_from(&query);
            self.global_search_results.clear();
            self.global_search_row_ids.clear();
            self.global_search_total_matches = 0;
        }
        let request_id = self.global_search_request_id;
        let page_row_ids = if new_search {
            None
        } else {
            let end = offset
                .saturating_add(MESSAGE_SEARCH_PAGE_SIZE as usize)
                .min(self.global_search_row_ids.len());
            Some((
                self.global_search_row_ids
                    .get(offset..end)
                    .unwrap_or_default()
                    .to_vec(),
                self.global_search_row_ids.len(),
            ))
        };
        self.global_searching = true;
        self.global_search_error = None;
        let db_path = self.active_db_path();
        let (tx, rx) = std::sync::mpsc::channel();
        self.global_search_rx = Some(rx);
        diagnostics::spawn_named("global-search", move || {
            let conn = match open_readonly_database(&db_path) {
                Ok(conn) => conn,
                Err(error) => {
                    let _ = tx.send(MessageSearchMessage::Error {
                        request_id,
                        offset,
                        message: error.to_string(),
                    });
                    ctx.request_repaint();
                    return;
                }
            };
            let result = if let Some((row_ids, total_matches)) = page_row_ids {
                load_search_results_by_rowids(&conn, &row_ids).map(|results| {
                    MessageSearchMessage::Finished {
                        request_id,
                        offset,
                        total_matches,
                        done: offset.saturating_add(row_ids.len()) >= total_matches,
                        results,
                    }
                })
            } else {
                search_message_rowids(&conn, &query).and_then(|row_ids| {
                    let first_page_end = row_ids.len().min(MESSAGE_SEARCH_PAGE_SIZE as usize);
                    load_search_results_by_rowids(&conn, &row_ids[..first_page_end]).map(
                        |results| MessageSearchMessage::IndexReady {
                            request_id,
                            row_ids,
                            results,
                        },
                    )
                })
            };
            match result {
                Ok(message) => {
                    let _ = tx.send(message);
                }
                Err(error) => {
                    let _ = tx.send(MessageSearchMessage::Error {
                        request_id,
                        offset,
                        message: error.to_string(),
                    });
                }
            }
            ctx.request_repaint();
        });
    }

    pub(crate) fn trigger_chat_message_search(
        &mut self,
        chat_id: String,
        query: String,
        offset: usize,
        ctx: egui::Context,
    ) {
        let query = query.trim().to_string();
        self.chat_search_request_id = self.chat_search_request_id.wrapping_add(1);
        let request_id = self.chat_search_request_id;
        if query.is_empty() {
            if let Ok(mut view) = self.active_chat_view.lock()
                && let Some(view) = view.as_mut()
            {
                view.search_results.clear();
                view.total_search_matches = 0;
                view.current_search_match_idx = None;
                view.search_error = None;
                view.searching = false;
            }
            self.chat_search_rx = None;
            return;
        }
        if let Ok(mut view) = self.active_chat_view.lock()
            && let Some(view) = view.as_mut()
        {
            view.searching = true;
            view.search_error = None;
            view.highlight_query.clone_from(&query);
        }
        let db_path = self.active_db_path();
        let (tx, rx) = std::sync::mpsc::channel();
        self.chat_search_rx = Some(rx);
        diagnostics::spawn_named("chat-search", move || {
            let conn = match open_readonly_database(&db_path) {
                Ok(conn) => conn,
                Err(error) => {
                    let _ = tx.send(MessageSearchMessage::Error {
                        request_id,
                        offset,
                        message: error.to_string(),
                    });
                    ctx.request_repaint();
                    return;
                }
            };
            let result =
                count_search_messages(&conn, &query, Some(&chat_id)).and_then(|total_matches| {
                    let limit = i64::try_from(total_matches).unwrap_or(i64::MAX);
                    search_message_page(&conn, &query, Some(&chat_id), limit, offset)
                        .map(|results| (results, total_matches))
                });
            match result {
                Ok((results, total_matches)) => {
                    let _ = tx.send(MessageSearchMessage::Finished {
                        request_id,
                        offset,
                        total_matches,
                        results,
                        done: true,
                    });
                }
                Err(error) => {
                    let _ = tx.send(MessageSearchMessage::Error {
                        request_id,
                        offset,
                        message: error.to_string(),
                    });
                }
            }
            ctx.request_repaint();
        });
    }
}
