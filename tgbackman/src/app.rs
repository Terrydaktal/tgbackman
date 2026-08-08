//! GUI controller state and background-task orchestration.

use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::cache::{cache_is_fresh, default_database_path, get_cache_path, get_media_cache_path};
use crate::database::run_inventory;
use crate::matching::{clean_text_for_match, count_missing_messages, format_unix_to_ts};
use crate::model::{
    ActiveChatView, ActiveComparison, AlignedMessageRow, BackupMessage, CalcMessage, ChatGroup,
    CompareMessage, LoadMessage, MediaCalcMessage, MediaStats, SingleChatMessage,
};

pub(crate) struct OverlapApp {
    pub(crate) db_path: String,
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
}

impl Default for OverlapApp {
    fn default() -> Self {
        OverlapApp {
            db_path: default_database_path(),
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
        }
    }
}
impl OverlapApp {
    pub(crate) fn load_cache(&mut self) {
        self.cached_results.clear();
        let cache_path = get_cache_path(&self.db_path);
        if cache_is_fresh(&cache_path, &self.db_path) {
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
        let cache_path = get_media_cache_path(&self.db_path);
        if cache_is_fresh(&cache_path, &self.db_path) {
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

        let db_path = self.db_path.clone();
        let groups = self.groups.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.media_rx = Some(rx);

        std::thread::spawn(move || {
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
            match std::fs::File::create(&cache_path) {
                Ok(file) => {
                    if let Err(e) = serde_json::to_writer_pretty(file, &all_stats) {
                        let _ = tx.send(MediaCalcMessage::Error(format!(
                            "Failed to write cache: {}",
                            e
                        )));
                        ctx.request_repaint();
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(MediaCalcMessage::Error(format!(
                        "Failed to create cache file: {}",
                        e
                    )));
                    ctx.request_repaint();
                    return;
                }
            }

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
        } else if let Some(results) = self.cached_results.get(&group.name) {
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

        let db_path = self.db_path.clone();
        let groups = self.groups.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.rx = Some(rx);

        std::thread::spawn(move || {
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
                all_results.insert(group.name.clone(), results);
            }

            // Write cache to file
            let cache_path = get_cache_path(&db_path);
            if let Ok(file) = std::fs::File::create(&cache_path) {
                let _ = serde_json::to_writer_pretty(file, &all_results);
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

        let db_path = self.db_path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.load_rx = Some(rx);

        std::thread::spawn(move || {
            let _ = tx.send(LoadMessage::Loading("Connecting to SQLite...".to_string()));
            ctx.request_repaint();

            let conn = match rusqlite::Connection::open(&db_path) {
                Ok(c) => {
                    let _ = c.execute("PRAGMA cache_size = -1048576;", []);
                    let _ = c.execute("PRAGMA temp_store = MEMORY;", []);
                    let _ = c.execute(
                        "ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0;",
                        [],
                    );
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER;", []);
                    let _ = c.execute(
                        "ALTER TABLE chats ADD COLUMN last_backup_run_unix INTEGER;",
                        [],
                    );
                    let _ = c.execute(
                        "ALTER TABLE chats ADD COLUMN last_backup_run_status TEXT;",
                        [],
                    );
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN min_msg_id INTEGER;", []);
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN max_msg_id INTEGER;", []);
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN msg_count INTEGER;", []);
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN min_timestamp TEXT;", []);
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN max_timestamp TEXT;", []);
                    let _ = c.execute(
                        "ALTER TABLE chats ADD COLUMN min_timestamp_unix INTEGER;",
                        [],
                    );
                    let _ = c.execute(
                        "ALTER TABLE chats ADD COLUMN max_timestamp_unix INTEGER;",
                        [],
                    );
                    c
                }
                Err(e) => {
                    let _ = tx.send(LoadMessage::Error(format!("Failed to open DB: {}", e)));
                    ctx.request_repaint();
                    return;
                }
            };

            let _ = tx.send(LoadMessage::Loading(
                "Scanning backup folders and calculating timeline boundaries...".to_string(),
            ));
            ctx.request_repaint();

            match run_inventory(&conn, &db_path) {
                Ok(groups) => {
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

            let db_path = self.db_path.clone();

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

            std::thread::spawn(move || {
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
                    "SELECT message_id, sender, timestamp_unix, timestamp, text, media_type, media_path FROM messages WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ? ORDER BY timestamp_unix ASC, message_id ASC"
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
                        })
                    },
                ) {
                    Ok(mapped) => {
                        let mut msgs = Vec::new();
                        for item in mapped {
                            if let Ok(m) = item {
                                msgs.push(m);
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
                    "SELECT message_id, sender, timestamp_unix, timestamp, text, media_type, media_path FROM messages WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ? ORDER BY timestamp_unix ASC, message_id ASC"
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
                        })
                    },
                ) {
                    Ok(mapped) => {
                        let mut msgs = Vec::new();
                        for item in mapped {
                            if let Ok(m) = item {
                                msgs.push(m);
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
        if let Some(group_idx) = self.selected_group_idx {
            let group = &self.groups[group_idx];
            let backup = &group.backups[idx];

            let letter = (b'A' + idx as u8) as char;
            self.loading_chat_view = true;
            self.status_msg = format!("Loading chat history for Backup {}...", letter);

            let db_path = self.db_path.clone();
            let chat_id = backup.chat_id.clone();
            let backup_name = format!("Backup {} ({})", letter, backup.name);

            let (tx, rx) = std::sync::mpsc::channel();
            self.chat_view_rx = Some(rx);

            std::thread::spawn(move || {
                let _ = tx.send(SingleChatMessage::Loading(
                    "Connecting to database...".to_string(),
                ));
                ctx.request_repaint();

                let conn = match rusqlite::Connection::open(&db_path) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(SingleChatMessage::Error(format!(
                            "Failed to open DB: {}",
                            e
                        )));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(SingleChatMessage::Loading(
                    "Fetching messages from database...".to_string(),
                ));
                ctx.request_repaint();

                let mut stmt = match conn.prepare(
                    "SELECT message_id, sender, timestamp_unix, timestamp, text, media_type, media_path FROM messages WHERE chat_id = ? ORDER BY timestamp_unix ASC, message_id ASC"
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(SingleChatMessage::Error(format!("Prepare query failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let messages = match stmt.query_map(rusqlite::params![chat_id], |r| {
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
                    })
                }) {
                    Ok(mapped) => {
                        let mut msgs = Vec::new();
                        for item in mapped {
                            if let Ok(m) = item {
                                msgs.push(m);
                            }
                        }
                        msgs
                    }
                    Err(e) => {
                        let _ = tx.send(SingleChatMessage::Error(format!("Query failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let chat_view = ActiveChatView {
                    backup_name,
                    chat_id,
                    messages,
                    scroll_to_bottom: true,
                    search_query: String::new(),
                    filtered_indices: Vec::new(),
                    search_matches_count: 0,
                    current_search_match_idx: None,
                    scroll_to_row_idx: None,
                };

                let _ = tx.send(SingleChatMessage::Finished(chat_view));
                ctx.request_repaint();
            });
        }
    }
}
