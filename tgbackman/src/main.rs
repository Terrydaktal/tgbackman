#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // Hide console window on Windows in release

use eframe::egui;
use std::collections::{HashMap, HashSet};
use chrono::{TimeZone, Utc};
use std::sync::{Arc, Mutex, OnceLock};
use regex::Regex;

enum CalcMessage {
    Progress(String),
    Finished(HashMap<String, Vec<String>>),
    Error(String),
}

enum LoadMessage {
    Loading(String),
    Finished(Vec<ChatGroup>),
    Error(String),
}

enum CompareMessage {
    Loading(String),
    Finished(ActiveComparison),
    Error(String),
}

enum SingleChatMessage {
    Loading(String),
    Finished(ActiveChatView),
    Error(String),
}

#[allow(dead_code)]
#[derive(Clone)]
struct ActiveChatView {
    backup_name: String,
    chat_id: String,
    messages: Vec<BackupMessage>,
    scroll_to_bottom: bool,
    search_query: String,
    filtered_indices: Vec<usize>,
    search_matches_count: usize,
    current_search_match_idx: Option<usize>,
    scroll_to_row_idx: Option<usize>,
}

#[derive(Clone)]
struct BackupMessage {
    message_id: i64,
    sender: String,
    timestamp_unix: i64,
    timestamp_str: String,
    text: String,
    clean_text: String,
    media_type: Option<String>,
    media_path: Option<String>,
}

#[derive(Clone)]
struct AlignedMessageRow {
    msg_a: Option<BackupMessage>,
    msg_b: Option<BackupMessage>,
    is_discrepancy: bool,
}

#[derive(Clone)]
struct ActiveComparison {
    backup_a_letter: char,
    backup_b_letter: char,
    backup_a_name: String,
    backup_b_name: String,
    rows: Vec<AlignedMessageRow>,
    discrepancies: Vec<usize>,
    current_discrepancy_idx: Option<usize>,
    scroll_to_row_idx: Option<usize>,
}


fn get_cache_path(db_path: &str) -> String {
    if db_path.ends_with(".db") {
        format!("{}_overlaps.json", &db_path[..db_path.len() - 3])
    } else {
        format!("{}_overlaps.json", db_path)
    }
}

fn get_backup_execution_time(backup_path: &str) -> Option<i64> {
    let media_dirs = [
        "files", "photos", "video_files", "voice_messages", 
        "audio_files", "sticker_files", "stickers", "documents", "animations"
    ];

    let mut max_mtime: Option<i64> = None;

    // Helper to recursively scan a directory for files and find max mtime
    fn scan_dir_max_mtime(dir: &std::path::Path, max_val: &mut Option<i64>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(ft) = entry.file_type() {
                    if ft.is_dir() {
                        scan_dir_max_mtime(&entry.path(), max_val);
                    } else if ft.is_file() {
                        if let Ok(meta) = entry.metadata() {
                            if let Ok(modified) = meta.modified() {
                                if let Ok(dur) = modified.duration_since(std::time::SystemTime::UNIX_EPOCH) {
                                    let secs = dur.as_secs() as i64;
                                    *max_val = Some(max_val.map_or(secs, |mv| mv.max(secs)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Scan matching media directories under backup_path
    if let Ok(entries) = std::fs::read_dir(backup_path) {
        for entry in entries.flatten() {
            if let Ok(ft) = entry.file_type() {
                if ft.is_dir() {
                    let path = entry.path();
                    if let Some(name_str) = path.file_name().and_then(|n| n.to_str()) {
                        let name_lower = name_str.to_lowercase();
                        if media_dirs.contains(&name_lower.as_str()) {
                            scan_dir_max_mtime(&path, &mut max_mtime);
                        }
                    }
                }
            }
        }
    }

    // If we found a max mtime from the media files, return it!
    if let Some(mtime) = max_mtime {
        return Some(mtime);
    }

    // Fallback: check metadata of database.sqlite, messages.html, result.json, results.json, and the backup_path directory itself
    let check_paths = [
        format!("{}/database.sqlite", backup_path),
        format!("{}/messages.html", backup_path),
        format!("{}/result.json", backup_path),
        format!("{}/results.json", backup_path),
        backup_path.to_string(),
    ];
    
    for path in &check_paths {
        if let Ok(meta) = std::fs::metadata(path) {
            if let Ok(modified) = meta.modified() {
                if let Ok(dur) = modified.duration_since(std::time::SystemTime::UNIX_EPOCH) {
                    return Some(dur.as_secs() as i64);
                }
            }
        }
    }

    None
}

// Union-Find helper for alias linking
struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new() -> Self {
        UnionFind { parent: HashMap::new() }
    }

    fn find(&mut self, x: &str) -> String {
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }
        let px = self.parent.get(x).unwrap().clone();
        if px == x {
            return x.to_string();
        }
        let root = self.find(&px);
        self.parent.insert(x.to_string(), root.clone());
        root
    }

    fn union(&mut self, x: &str, y: &str) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            self.parent.insert(root_x, root_y);
        }
    }
}

#[derive(Clone, Default)]
struct MediaStats {
    photos_count: i64,
    photos_resolved: i64,
    videos_count: i64,
    videos_resolved: i64,
    voice_count: i64,
    voice_resolved: i64,
    files_count: i64,
    files_resolved: i64,
}

#[derive(Clone)]
struct BackupInfo {
    chat_id: String,
    name: String,
    path: String,
    min_id: Option<i64>,
    max_id: Option<i64>,
    count: i64,
    min_ts: String,
    max_ts: String,
    min_unix: Option<i64>,
    max_unix: Option<i64>,
    is_active: bool,
    last_backup_unix: Option<i64>,
    media_stats: Option<MediaStats>,
}

impl BackupInfo {
    fn compute_media_stats(&self, db_path: &str) -> MediaStats {
        let mut photos_count = 0;
        let mut photos_resolved = 0;
        let mut videos_count = 0;
        let mut videos_resolved = 0;
        let mut voice_count = 0;
        let mut voice_resolved = 0;
        let mut files_count = 0;
        let mut files_resolved = 0;

        if let Ok(conn) = rusqlite::Connection::open(db_path) {
            let stmt = conn.prepare(
                "SELECT media_type, media_path FROM messages WHERE chat_id = ? AND media_type IN ('photo', 'video', 'voice_message', 'file')"
            );
            if let Ok(mut stmt) = stmt {
                let rows = stmt.query_map(rusqlite::params![self.chat_id], |row| {
                    let mt: Option<String> = row.get(0)?;
                    let mp: Option<String> = row.get(1)?;
                    Ok((mt, mp))
                });
                if let Ok(rows) = rows {
                    for row in rows {
                        if let Ok((media_type, media_path)) = row {
                            if let Some(mt) = media_type {
                                let path_exists = if let Some(ref mp) = media_path {
                                    if mp.is_empty() {
                                        false
                                    } else {
                                        let base = std::path::Path::new(&self.path);
                                        let mut exists = base.join(mp).exists();
                                        if !exists {
                                            let normalized = mp.replace("\\", "/");
                                            let parts: Vec<&str> = normalized.split('/').collect();
                                            let known_folders = [
                                                "photos", "video_files", "voice_messages", "audio_files",
                                                "stickers", "sticker_files", "files", "documents", "animations"
                                            ];
                                            for i in (0..parts.len()).rev() {
                                                if known_folders.contains(&parts[i].to_lowercase().as_str()) {
                                                    let subpath = parts[i..].join("/");
                                                    if base.join(&subpath).exists() {
                                                        exists = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if !exists {
                                            if let Some(fname) = std::path::Path::new(mp).file_name() {
                                                if base.join(fname).exists() {
                                                    exists = true;
                                                } else if base.join("files").join(fname).exists() {
                                                    exists = true;
                                                }
                                            }
                                        }
                                        exists
                                    }
                                } else {
                                    false
                                };
                                
                                match mt.as_str() {
                                    "photo" => {
                                        photos_count += 1;
                                        if path_exists {
                                            photos_resolved += 1;
                                        }
                                    }
                                    "video" => {
                                        videos_count += 1;
                                        if path_exists {
                                            videos_resolved += 1;
                                        }
                                    }
                                    "voice_message" => {
                                        voice_count += 1;
                                        if path_exists {
                                            voice_resolved += 1;
                                        }
                                    }
                                    "file" => {
                                        files_count += 1;
                                        if path_exists {
                                            files_resolved += 1;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        MediaStats {
            photos_count,
            photos_resolved,
            videos_count,
            videos_resolved,
            voice_count,
            voice_resolved,
            files_count,
            files_resolved,
        }
    }
}

#[derive(Clone)]
struct ChatGroup {
    name: String,
    max_count: i64,
    backups: Vec<BackupInfo>,
}

impl ChatGroup {
    fn is_active(&self) -> bool {
        self.backups.iter().any(|b| b.is_active)
    }
}

struct OverlapApp {
    db_path: String,
    groups: Vec<ChatGroup>,
    filtered_groups: Vec<usize>,
    selected_group_idx: Option<usize>,
    comparison_results: Vec<String>,
    status_msg: String,
    search_query: String,
    calculating_overlaps: bool,
    rx: Option<std::sync::mpsc::Receiver<CalcMessage>>,
    cached_results: HashMap<String, Vec<String>>,
    loading_data: bool,
    load_rx: Option<std::sync::mpsc::Receiver<LoadMessage>>,
    active_comparison: Arc<Mutex<Option<ActiveComparison>>>,
    loading_comparison: bool,
    compare_rx: Option<std::sync::mpsc::Receiver<CompareMessage>>,
    active_chat_view: Arc<Mutex<Option<ActiveChatView>>>,
    loading_chat_view: bool,
    chat_view_rx: Option<std::sync::mpsc::Receiver<SingleChatMessage>>,
}

impl Default for OverlapApp {
    fn default() -> Self {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        OverlapApp {
            db_path: format!("/media/{}/1b/sqlitedb/telegram_backup.db", user),
            groups: Vec::new(),
            filtered_groups: Vec::new(),
            selected_group_idx: None,
            comparison_results: Vec::new(),
            status_msg: "Database not loaded yet. Click 'Load Database' to begin.".to_string(),
            search_query: "".to_string(),
            calculating_overlaps: false,
            rx: None,
            cached_results: HashMap::new(),
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

fn strip_boundaries(word: &str) -> String {
    let start = word.find(|c: char| c.is_alphanumeric()).unwrap_or(word.len());
    let sub = &word[start..];
    let end = sub.rfind(|c: char| c.is_alphanumeric()).map(|idx| {
        let c = sub[idx..].chars().next().unwrap();
        idx + c.len_utf8()
    }).unwrap_or(0);
    sub[..end].to_lowercase()
}

// Custom text cleaner for DST and formatting invariant checks
fn clean_text_for_match(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    
    let trimmed = text.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed.to_string();
    }
    
    let mut clean = text.trim();
    
    // 1. Strip legacy forwarded headers like {{FWD: ...}} at start
    if clean.starts_with("{{FWD:") {
        if let Some(end_fwd) = clean.find("}}") {
            clean = clean[end_fwd + 2..].trim_start();
        }
    }
    
    // 2. Strip legacy double bracket prefixes/blocks like [[Webpage]] at start
    if clean.starts_with("[[") {
        if let Some(end_meta) = clean.find("]]") {
            clean = clean[end_meta + 2..].trim_start();
        }
    }

    let t = clean.to_lowercase();
    
    // Compile regexes statically for fast reuse
    static URL_RE: OnceLock<Regex> = OnceLock::new();
    let url_re = URL_RE.get_or_init(|| {
        Regex::new(r#"(?i)(?:https?://|tel:|mailto:|tg:)[^\s'"“‘’”<>]+"#).unwrap()
    });
    
    static DOMAIN_RE: OnceLock<Regex> = OnceLock::new();
    let domain_re = DOMAIN_RE.get_or_init(|| {
        Regex::new(r#"(?i)\b[a-z0-9\-]+\.[a-z]{2,24}(?:\.[a-z]{2,24})*\b"#).unwrap()
    });
    
    // 1. Replace URLs with space
    let t_url = url_re.replace_all(&t, " ");
    
    // 2. Replace Domains with space
    let t_dom = domain_re.replace_all(&t_url, " ");
    
    // Stage 1: Split by whitespace, and deduplicate adjacent words using their boundary-stripped normalized form
    let words: Vec<&str> = t_dom.split_whitespace().collect();
    let mut stage1_words: Vec<&str> = Vec::new();
    for w in words {
        if stage1_words.is_empty() {
            stage1_words.push(w);
        } else {
            let w_norm = strip_boundaries(w);
            let last_norm = strip_boundaries(stage1_words.last().unwrap());
            if !w_norm.is_empty() && w_norm == last_norm {
                // Skip duplicate
                continue;
            } else {
                stage1_words.push(w);
            }
        }
    }
    
    let t_stage1 = stage1_words.join(" ");
    
    // Stage 2: Keep only alphanumeric characters and spaces
    let mut clean_chars = String::new();
    for c in t_stage1.chars() {
        if c.is_alphanumeric() || c.is_whitespace() {
            clean_chars.push(c);
        } else {
            clean_chars.push(' ');
        }
    }
    
    // Stage 3: Split, final adjacent deduplication, and join
    let final_words: Vec<&str> = clean_chars.split_whitespace().collect();
    let mut deduped: Vec<&str> = Vec::new();
    for w in final_words {
        if deduped.is_empty() {
            deduped.push(w);
        } else if *deduped.last().unwrap() != w {
            deduped.push(w);
        }
    }
    
    deduped.join(" ").trim().to_string()
}

// In-memory timezone and media placeholder aware missing message counter
fn count_missing_messages(
    conn: &rusqlite::Connection,
    chat_a_id: &str,
    chat_b_id: &str,
    start_unix: i64,
    end_unix: i64,
) -> Result<i64, rusqlite::Error> {
    // 1. Fetch A messages in range
    let mut stmt_a = conn.prepare(
        "SELECT timestamp_unix, text FROM messages WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ?"
    )?;
    let mut rows_a = stmt_a.query(rusqlite::params![chat_a_id, start_unix, end_unix])?;
    let mut messages_a = Vec::new();
    while let Some(row) = rows_a.next()? {
        let ts: Option<i64> = row.get(0)?;
        let txt: Option<String> = row.get(1)?;
        if let Some(t) = ts {
            messages_a.push((t, clean_text_for_match(&txt.unwrap_or_default())));
        }
    }

    // 2. Fetch B messages in expanded range (accounting for 1-hour BST shifts)
    let mut stmt_b = conn.prepare(
        "SELECT timestamp_unix, text FROM messages WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ?"
    )?;
    let mut rows_b = stmt_b.query(rusqlite::params![chat_b_id, start_unix - 3605, end_unix + 3605])?;
    let mut b_by_ts: HashMap<i64, Vec<String>> = HashMap::new();
    while let Some(row) = rows_b.next()? {
        let ts: Option<i64> = row.get(0)?;
        let txt: Option<String> = row.get(1)?;
        if let Some(t) = ts {
            b_by_ts.entry(t).or_default().push(clean_text_for_match(&txt.unwrap_or_default()));
        }
    }

    // 3. Compare sets
    let mut missing_count = 0;
    for (ts, txt_clean) in messages_a {
        let mut found = false;
        'offset_search: for offset in &[0, 3600, -3600] {
            let target_ts = ts + offset;
            for candidate_ts in target_ts - 5..=target_ts + 5 {
                if let Some(candidates) = b_by_ts.get(&candidate_ts) {
                    for b_txt_clean in candidates {
                        if txt_clean == *b_txt_clean {
                            found = true;
                            break 'offset_search;
                        }
                        let is_a_empty = txt_clean.is_empty();
                        let is_b_media = b_txt_clean.starts_with('[') && b_txt_clean.ends_with(']');
                        let is_b_empty = b_txt_clean.is_empty();
                        let is_a_media = txt_clean.starts_with('[') && txt_clean.ends_with(']');
                        if (is_a_empty && is_b_media) || (is_b_empty && is_a_media) {
                            found = true;
                            break 'offset_search;
                        }
                    }
                }
            }
        }
        if !found {
            missing_count += 1;
        }
    }

    Ok(missing_count)
}

fn format_unix_to_ts(unix_ts: i64) -> String {
    if let Some(dt) = Utc.timestamp_opt(unix_ts, 0).single() {
        dt.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        "Unknown".to_string()
    }
}

// Main logic to parse DB and cluster aliased backups
fn run_inventory(conn: &rusqlite::Connection) -> Result<Vec<ChatGroup>, rusqlite::Error> {
    let start_total = std::time::Instant::now();
    
    let mut stmt = conn.prepare("SELECT chat_id, chat_name, backup_path, COALESCE(is_active, 0), last_backup_unix FROM chats")?;
    let mut rows = stmt.query([])?;
    let mut chats = Vec::new();
    let mut mtime_calls = 0;
    let mut mtime_time = std::time::Duration::ZERO;
    
    while let Some(row) = rows.next()? {
        let chat_id: String = row.get(0)?;
        let name: Option<String> = row.get(1)?;
        let path: Option<String> = row.get(2)?;
        let active: i32 = row.get(3)?;
        let mut last_backup: Option<i64> = row.get(4)?;
        
        if last_backup.is_none() {
            let start_m = std::time::Instant::now();
            mtime_calls += 1;
            if let Some(ts) = get_backup_execution_time(path.as_deref().unwrap_or("")) {
                let _ = conn.execute("UPDATE chats SET last_backup_unix = ? WHERE chat_id = ?", rusqlite::params![ts, chat_id]);
                last_backup = Some(ts);
            }
            mtime_time += start_m.elapsed();
        }
        
        chats.push((chat_id, name.unwrap_or_default(), path, active != 0, last_backup));
    }
    
    println!("Phase 1 (Chats fetch + mtimes): {:?}", start_total.elapsed());
    println!("  -> mtime calls: {}, time: {:?}", mtime_calls, mtime_time);
    
    let start_fuzzy = std::time::Instant::now();
    let mut uf = UnionFind::new();

    // 1. FUZZY ALIAS LINKING via oldest signatures
    let mut exact_signatures: HashMap<(i64, String), Vec<String>> = HashMap::new();
    for (cid, _, _, _, _) in &chats {
        let mut stmt_msgs = conn.prepare(
            "SELECT timestamp_unix, text FROM messages WHERE chat_id = ? AND text != '' AND timestamp_unix IS NOT NULL ORDER BY timestamp_unix ASC LIMIT 50"
        )?;
        let mut rows_msgs = stmt_msgs.query(rusqlite::params![cid])?;
        while let Some(row) = rows_msgs.next()? {
            let ts: i64 = row.get(0)?;
            let text: String = row.get(1)?;
            let clean_text = text.trim();
            if clean_text.len() >= 6 {
                exact_signatures.entry((ts, clean_text.to_string())).or_default().push(cid.clone());
            }
        }
    }

    let mut exact_shared_counts: HashMap<(String, String), i32> = HashMap::new();
    for (_, cids) in exact_signatures {
        if cids.len() < 2 {
            continue;
        }
        let unique_cids: HashSet<String> = cids.into_iter().collect();
        let unique_cids: Vec<String> = unique_cids.into_iter().collect();
        for i in 0..unique_cids.len() {
            for j in i + 1..unique_cids.len() {
                let c1 = unique_cids[i].clone();
                let c2 = unique_cids[j].clone();
                let pair = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                *exact_shared_counts.entry(pair).or_default() += 1;
            }
        }
    }

    for (pair, count) in exact_shared_counts {
        if count >= 3 {
            uf.union(&pair.0, &pair.1);
        }
    }
    
    println!("Phase 2 (Fuzzy linking): {:?}", start_fuzzy.elapsed());
    
    let start_joins = std::time::Instant::now();

    // 2. SAME-NAME DUPLICATES LINKING
    let mut chats_by_norm_name: HashMap<String, Vec<String>> = HashMap::new();
    for (cid, name, _, _, _) in &chats {
        if !name.is_empty() {
            let norm = name.trim().to_lowercase();
            if norm != "deleted account" && norm != "telegram" && norm != "group" && norm != "unknown" {
                chats_by_norm_name.entry(norm).or_default().push(cid.clone());
            }
        }
    }

    for (_, cids) in chats_by_norm_name {
        if cids.len() < 2 {
            continue;
        }
        for i in 0..cids.len() {
            for j in i + 1..cids.len() {
                let c1 = &cids[i];
                let c2 = &cids[j];
                if uf.find(c1) == uf.find(c2) {
                    continue;
                }
                let mut stmt_join = conn.prepare(
                    "SELECT COUNT(*) FROM messages a JOIN messages b ON a.timestamp_unix = b.timestamp_unix WHERE a.chat_id = ? AND b.chat_id = ? AND a.text = b.text AND a.text != '' AND length(a.text) >= 6"
                )?;
                let count: i64 = stmt_join.query_row(rusqlite::params![c1, c2], |r| r.get(0))?;
                if count >= 3 {
                    uf.union(c1, c2);
                }
            }
        }
    }
    
    println!("Phase 3 (Same-name joins): {:?}", start_joins.elapsed());
    
    let start_stats = std::time::Instant::now();

    // Query stats for all chats in a single GROUP BY query to avoid N separate scans
    let mut stats_map = HashMap::new();
    let mut stmt_stats = conn.prepare(
        "SELECT chat_id, MIN(message_id), MAX(message_id), COUNT(*), MIN(timestamp), MAX(timestamp), MIN(timestamp_unix), MAX(timestamp_unix) FROM messages GROUP BY chat_id"
    )?;
    let mut rows_stats = stmt_stats.query([])?;
    while let Some(row) = rows_stats.next()? {
        let cid: String = row.get(0)?;
        let min_id: Option<i64> = row.get(1)?;
        let max_id: Option<i64> = row.get(2)?;
        let count: i64 = row.get(3)?;
        let min_ts: Option<String> = row.get(4)?;
        let max_ts: Option<String> = row.get(5)?;
        let min_unix: Option<i64> = row.get(6)?;
        let max_unix: Option<i64> = row.get(7)?;
        
        stats_map.insert(cid, (min_id, max_id, count, min_ts, max_ts, min_unix, max_unix));
    }

    // Mappings and Stats
    let mut logical_groups: HashMap<String, Vec<BackupInfo>> = HashMap::new();
    for (cid, name, path, is_active, last_backup_unix) in chats {
        let norm_name = name.trim().to_lowercase();
        if (norm_name == "deleted account" || norm_name == "telegram" || norm_name == "group" || norm_name == "unknown") && uf.find(&cid) == cid {
            continue;
        }
        
        let root = uf.find(&cid);
        
        if let Some(stats) = stats_map.get(&cid) {
            let (min_id, max_id, count, min_ts, max_ts, min_unix, max_unix) = stats;
            if min_id.is_some() && *count > 0 {
                let format_ts = |ts_str: &Option<String>| -> String {
                    match ts_str {
                        Some(s) => s.replace("T", " ").replace("Z", ""),
                        None => "Unknown".to_string()
                    }
                };
                
                logical_groups.entry(root).or_default().push(BackupInfo {
                    chat_id: cid,
                    name: if name.is_empty() { "Unknown".to_string() } else { name },
                    path: path.unwrap_or_default(),
                    min_id: *min_id,
                    max_id: *max_id,
                    count: *count,
                    min_ts: format_ts(min_ts),
                    max_ts: format_ts(max_ts),
                    min_unix: *min_unix,
                    max_unix: *max_unix,
                    is_active,
                    last_backup_unix,
                    media_stats: None,
                });
            }
        }
    }
    
    println!("Phase 4 (Stats queries): {:?}", start_stats.elapsed());
    
    let start_groups = std::time::Instant::now();

    let mut result_groups = Vec::new();
    for (_, mut entries) in logical_groups {
        if entries.is_empty() {
            continue;
        }
        entries.sort_by_key(|e| e.count);
        let max_count = entries.iter().map(|e| e.count).max().unwrap_or(0);
        let names: HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
        let mut names_vec: Vec<String> = names.into_iter().collect();
        names_vec.sort();
        let display_name = names_vec.join(" / ");
        
        result_groups.push(ChatGroup {
            name: display_name,
            max_count,
            backups: entries,
        });
    }

    result_groups.sort_by(|a, b| b.max_count.cmp(&a.max_count));
    println!("Phase 5 (Grouping + final sorting): {:?}", start_groups.elapsed());
    
    Ok(result_groups)
}

impl OverlapApp {
    fn load_cache(&mut self) {
        let cache_path = get_cache_path(&self.db_path);
        if let Ok(file) = std::fs::File::open(&cache_path) {
            if let Ok(cache) = serde_json::from_reader(file) {
                self.cached_results = cache;
                self.status_msg = format!("Loaded database and found cached overlaps for {} groups.", self.cached_results.len());
                return;
            }
        }
        self.cached_results.clear();
        self.status_msg = "Database loaded. No cached overlaps found. Click '🔄 Recompute All Overlaps'.".to_string();
    }

    fn select_group(&mut self, idx: usize) {
        self.selected_group_idx = Some(idx);
        let group = &self.groups[idx];
        if group.backups.len() < 2 {
            self.comparison_results = vec!["Only 1 backup available in this group. No overlap analysis needed.".to_string()];
        } else if let Some(results) = self.cached_results.get(&group.name) {
            self.comparison_results = results.clone();
        } else {
            self.comparison_results = vec![
                "No cached overlaps found for this chat.".to_string(),
                "Click '🔄 Recompute All Overlaps' to calculate and cache results.".to_string(),
            ];
        }
    }

    fn recompute_all_overlaps(&mut self, ctx: egui::Context) {
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
            let groups_to_calc: Vec<&ChatGroup> = groups.iter().filter(|g| g.backups.len() >= 2).collect();
            let total_to_calc = groups_to_calc.len();

            for (idx, group) in groups_to_calc.into_iter().enumerate() {
                let msg = format!("Recomputing: group {} of {} ({})", idx + 1, total_to_calc, group.name);
                let _ = tx.send(CalcMessage::Progress(msg));
                ctx.request_repaint();

                let mut results = Vec::new();
                for i in 0..group.backups.len() {
                    for j in i + 1..group.backups.len() {
                        let a = &group.backups[i];
                        let b = &group.backups[j];
                        
                        let letter_a = (b'A' + i as u8) as char;
                        let letter_b = (b'A' + j as u8) as char;

                        if let (Some(a_min), Some(a_max), Some(b_min), Some(b_max)) = (a.min_unix, a.max_unix, b.min_unix, b.max_unix) {
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
                                    format!("Backup {} fully contains the chronological span of Backup {}!", letter_b, letter_a)
                                } else if a_contains_b {
                                    format!("Backup {} fully contains the chronological span of Backup {}!", letter_a, letter_b)
                                } else {
                                    format!("Backup {} and Backup {} overlap chronologically by {:.1} days!", letter_a, letter_b, overlap_days)
                                };

                                results.push(format!("⚖️ {}", relationship));
                                results.push(format!("    Overlap span: {} to {}", format_unix_to_ts(overlap_start), format_unix_to_ts(overlap_end)));

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
                                results.push(format!("⚠️ Backup {} and Backup {} do not overlap chronologically.", letter_a, letter_b));
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

    fn filter_groups(&mut self) {
        if self.search_query.is_empty() {
            self.filtered_groups = (0..self.groups.len()).collect();
        } else {
            let query = self.search_query.to_lowercase();
            self.filtered_groups = self.groups
                .iter()
                .enumerate()
                .filter(|(_, g)| g.name.to_lowercase().contains(&query))
                .map(|(idx, _)| idx)
                .collect();
        }
    }

    fn trigger_load_data(&mut self, ctx: egui::Context) {
        self.loading_data = true;
        self.status_msg = "Opening database...".to_string();
        self.groups.clear();
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
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0;", []);
                    let _ = c.execute("ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER;", []);
                    c
                }
                Err(e) => {
                    let _ = tx.send(LoadMessage::Error(format!("Failed to open DB: {}", e)));
                    ctx.request_repaint();
                    return;
                }
            };

            let _ = tx.send(LoadMessage::Loading("Scanning backup folders and calculating timeline boundaries...".to_string()));
            ctx.request_repaint();

            match run_inventory(&conn) {
                Ok(groups) => {
                    let _ = tx.send(LoadMessage::Finished(groups));
                }
                Err(e) => {
                    let _ = tx.send(LoadMessage::Error(format!("Failed to run inventory: {}", e)));
                }
            }
            ctx.request_repaint();
        });
    }

    fn trigger_comparison(&mut self, idx_a: usize, idx_b: usize, ctx: egui::Context) {
        if let Some(group_idx) = self.selected_group_idx {
            let group = &self.groups[group_idx];
            let backup_a = &group.backups[idx_a];
            let backup_b = &group.backups[idx_b];
            
            let letter_a = (b'A' + idx_a as u8) as char;
            let letter_b = (b'A' + idx_b as u8) as char;

            self.loading_comparison = true;
            self.status_msg = format!("Loading side-by-side messages for {} vs {}...", letter_a, letter_b);

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
                let _ = tx.send(CompareMessage::Loading("Connecting to database...".to_string()));
                ctx.request_repaint();

                let conn = match rusqlite::Connection::open(&db_path) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(CompareMessage::Error(format!("Failed to open DB: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading("Fetching Backup A messages in overlap range...".to_string()));
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

                let rows_a = match stmt_a.query_map(rusqlite::params![chat_a_id, start_unix, end_unix], |r| {
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
                        let _ = tx.send(CompareMessage::Error(format!("Query A failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading("Fetching Backup B messages in overlap range...".to_string()));
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

                let rows_b = match stmt_b.query_map(rusqlite::params![chat_b_id, start_unix - 3605, end_unix + 3605], |r| {
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
                        let _ = tx.send(CompareMessage::Error(format!("Query B failed: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(CompareMessage::Loading("Aligning message streams side-by-side...".to_string()));
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
                                    let is_b_media = b.clean_text.starts_with('[') && b.clean_text.ends_with(']');
                                    
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
                    r.msg_a.as_ref()
                        .or(r.msg_b.as_ref())
                        .map(|m| m.timestamp_unix)
                        .unwrap_or(0)
                });

                let discrepancies: Vec<usize> = aligned_rows.iter()
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
                    current_discrepancy_idx: if discrepancies.is_empty() { None } else { Some(0) },
                    scroll_to_row_idx: if discrepancies.is_empty() { None } else { Some(discrepancies[0]) },
                    discrepancies,
                };

                let _ = tx.send(CompareMessage::Finished(active_comp));
                ctx.request_repaint();
            });
        }
    }

    fn trigger_load_chat(&mut self, idx: usize, ctx: egui::Context) {
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
                let _ = tx.send(SingleChatMessage::Loading("Connecting to database...".to_string()));
                ctx.request_repaint();

                let conn = match rusqlite::Connection::open(&db_path) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(SingleChatMessage::Error(format!("Failed to open DB: {}", e)));
                        ctx.request_repaint();
                        return;
                    }
                };

                let _ = tx.send(SingleChatMessage::Loading("Fetching messages from database...".to_string()));
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

impl eframe::App for OverlapApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background database loading
        if let Some(ref rx) = self.load_rx {
            match rx.try_recv() {
                Ok(LoadMessage::Loading(msg)) => {
                    self.status_msg = msg;
                }
                Ok(LoadMessage::Finished(groups)) => {
                    self.groups = groups;
                    self.loading_data = false;
                    self.load_cache();
                    self.filter_groups();
                    self.load_rx = None;
                }
                Ok(LoadMessage::Error(err)) => {
                    self.status_msg = err;
                    self.loading_data = false;
                    self.load_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.loading_data = false;
                    self.load_rx = None;
                }
            }
        }

        // Poll background overlap calculations
        if let Some(ref rx) = self.rx {
            match rx.try_recv() {
                Ok(CalcMessage::Progress(msg)) => {
                    self.status_msg = msg;
                }
                Ok(CalcMessage::Finished(results)) => {
                    self.cached_results = results;
                    self.calculating_overlaps = false;
                    self.status_msg = "Overlaps calculation completed & cached successfully.".to_string();
                    self.rx = None;
                    if let Some(idx) = self.selected_group_idx {
                        self.select_group(idx);
                    }
                }
                Ok(CalcMessage::Error(err)) => {
                    self.status_msg = format!("Calculation failed: {}", err);
                    self.calculating_overlaps = false;
                    self.rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.calculating_overlaps = false;
                    self.rx = None;
                }
            }
        }

        // Poll background comparison loading
        if let Some(ref rx) = self.compare_rx {
            match rx.try_recv() {
                Ok(CompareMessage::Loading(msg)) => {
                    self.status_msg = msg;
                }
                Ok(CompareMessage::Finished(comp)) => {
                    *self.active_comparison.lock().unwrap() = Some(comp);
                    self.loading_comparison = false;
                    self.status_msg = "Comparison messages aligned and loaded successfully.".to_string();
                    self.compare_rx = None;
                }
                Ok(CompareMessage::Error(err)) => {
                    self.status_msg = format!("Comparison failed: {}", err);
                    self.loading_comparison = false;
                    self.compare_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.loading_comparison = false;
                    self.compare_rx = None;
                }
            }
        }

        // Poll background chat viewer loading
        if let Some(ref rx) = self.chat_view_rx {
            match rx.try_recv() {
                Ok(SingleChatMessage::Loading(msg)) => {
                    self.status_msg = msg;
                }
                Ok(SingleChatMessage::Finished(chat_view)) => {
                    *self.active_chat_view.lock().unwrap() = Some(chat_view);
                    self.loading_chat_view = false;
                    self.status_msg = "Chat messages loaded successfully.".to_string();
                    self.chat_view_rx = None;
                }
                Ok(SingleChatMessage::Error(err)) => {
                    self.status_msg = format!("Chat loading failed: {}", err);
                    self.loading_chat_view = false;
                    self.chat_view_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.loading_chat_view = false;
                    self.chat_view_rx = None;
                }
            }
        }

        // Dark theme adjustments
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(20, 20, 25);
        ctx.set_visuals(visuals);

        // Top Control Panel
        egui::TopBottomPanel::top("control_panel")
            .frame(egui::Frame::none().inner_margin(12.0).fill(egui::Color32::from_rgb(25, 25, 30)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("📊 tgbackman");
                    ui.label("  |  ");
                    ui.label("Database Path:");
                    ui.text_edit_singleline(&mut self.db_path);
                    if self.loading_data {
                        ui.add(egui::Spinner::new());
                        ui.label("Loading Database...");
                    } else if ui.button("🔄 Load Database").clicked() {
                        self.trigger_load_data(ctx.clone());
                    }
                    ui.add_space(10.0);
                    if self.calculating_overlaps {
                        ui.add(egui::Spinner::new());
                        ui.label("Computing...");
                    } else if ui.button("🔄 Recompute All Overlaps").clicked() {
                        self.recompute_all_overlaps(ctx.clone());
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(&self.status_msg);
                });
            });

        // Left Side Chat List Panel
        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(320.0)
            .frame(egui::Frame::none().inner_margin(12.0).fill(egui::Color32::from_rgb(15, 15, 20)))
            .show(ctx, |ui| {
                ui.label("🔍 Search Chats:");
                if ui.text_edit_singleline(&mut self.search_query).changed() {
                    self.filter_groups();
                }
                ui.add_space(8.0);
                
                ui.heading("Conversations");
                ui.separator();
                
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut next_selected_idx = None;
                    let now = Utc::now().timestamp();
                    for &idx in &self.filtered_groups {
                        let group = &self.groups[idx];
                        let selected = self.selected_group_idx == Some(idx);
                        
                        let latest_backup_unix = group.backups.iter()
                            .filter_map(|b| b.last_backup_unix)
                            .max();
                        
                        let ago_str = match latest_backup_unix {
                            Some(ts) => {
                                let days = (now - ts) / 86400;
                                if days >= 0 {
                                    format!("{}d ago", days)
                                } else {
                                    "0d ago".to_string()
                                }
                            }
                            None => "never".to_string(),
                        };
                        
                        let item_color = if group.is_active() {
                            egui::Color32::from_rgb(46, 204, 113) // Green
                        } else {
                            egui::Color32::from_rgb(231, 76, 60)  // Red
                        };
                        let label_text = egui::RichText::new(format!("{} ({} msgs) • {}", group.name, group.max_count, ago_str))
                            .color(item_color);
                        let response = ui.selectable_label(selected, label_text);
                        
                        if response.clicked() {
                            next_selected_idx = Some(idx);
                        }
                    }
                    if let Some(idx) = next_selected_idx {
                        self.select_group(idx);
                    }
                });
            });

        // Central Panel (Gantt and Detail View)
        let mut compare_pair = None;
        let mut open_chat_idx = None;
        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(16.0).fill(egui::Color32::from_rgb(20, 20, 25)))
            .show(ctx, |ui| {
                if let Some(idx) = self.selected_group_idx {
                    let mut recompute_media = false;
                    let mut toggle_group_active = None;
                    
                    {
                        let group = &self.groups[idx];
                        
                        ui.horizontal(|ui| {
                            let group_active = group.is_active();
                            let status_color = if group_active {
                                egui::Color32::from_rgb(46, 204, 113) // green
                            } else {
                                egui::Color32::from_rgb(231, 76, 60)  // red
                            };
                            let status_text = if group_active { "Active" } else { "Inactive" };
                            let label_text = egui::RichText::new(format!("Selected: {} ({})", group.name, status_text)).strong().color(status_color);
                            if ui.selectable_label(false, label_text).on_hover_text("Click to toggle conversation Active status").clicked() {
                                toggle_group_active = Some((idx, !group_active));
                            }
                            
                            ui.add_space(20.0);
                            if ui.button("📊 Recompute Media Counts").on_hover_text("Scan backup directories and count/validate photos, videos, and voice messages").clicked() {
                                recompute_media = true;
                            }
                        });
                        ui.separator();
                        ui.add_space(10.0);
                        
                        ui.label("📅 Backup Chronological Timeline (Gantt Chart)");
                        ui.add_space(4.0);
                        
                        // Render Gantt
                        draw_gantt_chart(ui, &group.backups);
                        
                        ui.add_space(15.0);
                        
                        // Render Backup Information Table
                        ui.heading("📦 Backup Inventories");
                        egui::ScrollArea::vertical().id_source("inventories_scroll").max_height(200.0).show(ui, |ui| {
                            for (b_idx, b) in group.backups.iter().enumerate() {
                                let letter = (b'A' + b_idx as u8) as char;
                                let backup_run_ts = match b.last_backup_unix {
                                    Some(ts) => format_unix_to_ts(ts),
                                    None => "Unknown".to_string(),
                                };
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(get_color_by_idx(b_idx), format!("Backup {}", letter));
                                        ui.add_space(8.0);
                                        if ui.small_button("💬 Open Chat").on_hover_text("Open message history in a Telegram-styled window").clicked() {
                                            open_chat_idx = Some(b_idx);
                                        }
                                        ui.label(format!("| Path: {}", b.path));
                                    });
                                    ui.label(format!("   Last Backup Run: {}", backup_run_ts));
                                    ui.label(format!("   Message IDs:     {} to {} (Total: {} messages)", b.min_id.unwrap_or(0), b.max_id.unwrap_or(0), b.count));
                                    ui.label(format!("   Time span:       {} to {}", b.min_ts, b.max_ts));
                                    if let Some(ref stats) = b.media_stats {
                                        ui.horizontal(|ui| {
                                            ui.label("   Media Assets:    ");
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 240), format!("📷 Photos: {}/{}", stats.photos_resolved, stats.photos_count));
                                            ui.label(" | ");
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 240), format!("🎥 Videos: {}/{}", stats.videos_resolved, stats.videos_count));
                                            ui.label(" | ");
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 240), format!("🎤 Voice: {}/{}", stats.voice_resolved, stats.voice_count));
                                            ui.label(" | ");
                                            ui.colored_label(egui::Color32::from_rgb(100, 180, 240), format!("📂 Files: {}/{}", stats.files_resolved, stats.files_count));
                                        });
                                    } else {
                                        ui.horizontal(|ui| {
                                            ui.label("   Media Assets:    ");
                                            ui.colored_label(egui::Color32::from_rgb(140, 150, 160), "Not scanned. Click '📊 Recompute Media Counts' at the top to scan.");
                                        });
                                    }
                                });
                                ui.add_space(4.0);
                            }
                        });

                        if group.backups.len() >= 2 {
                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                ui.label("🔍 Compare Backups Side-by-Side:");
                                for i in 0..group.backups.len() {
                                    for j in i + 1..group.backups.len() {
                                        let letter_a = (b'A' + i as u8) as char;
                                        let letter_b = (b'A' + j as u8) as char;
                                        if ui.button(format!("⚖️ {} vs {}", letter_a, letter_b)).on_hover_text(format!("Compare messages in overlapping regions between Backup {} and Backup {}", letter_a, letter_b)).clicked() {
                                            compare_pair = Some((i, j));
                                        }
                                    }
                                }
                            });
                        }
                    }
                    
                    if recompute_media {
                        if let Some(group) = self.groups.get_mut(idx) {
                            for b in &mut group.backups {
                                b.media_stats = Some(b.compute_media_stats(&self.db_path));
                            }
                        }
                    }
                    
                    if let Some((g_idx, active)) = toggle_group_active {
                        if let Ok(conn) = rusqlite::Connection::open(&self.db_path) {
                            let val = if active { 1 } else { 0 };
                            let group = &self.groups[g_idx];
                            for b in &group.backups {
                                let _ = conn.execute(
                                    "UPDATE chats SET is_active = ? WHERE chat_id = ?",
                                    rusqlite::params![val, b.chat_id],
                                );
                            }
                        }
                        if let Some(g) = self.groups.get_mut(g_idx) {
                            for b in &mut g.backups {
                                b.is_active = active;
                            }
                        }
                    }
                    
                    ui.add_space(15.0);
                    ui.heading("⚖️ Containment & Overlaps Analysis");
                    ui.separator();
                    ui.add_space(4.0);
                    
                    egui::ScrollArea::vertical().id_source("overlaps_scroll").show(ui, |ui| {
                        for line in &self.comparison_results {
                            if line.trim().is_empty() {
                                ui.add_space(4.0);
                            } else {
                                ui.label(line);
                            }
                        }
                    });
                    
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Select a conversation from the sidebar list to view timelines, Gantt spans, and overlaps analysis.");
                    });
                }
            });

        if let Some((i, j)) = compare_pair {
            self.trigger_comparison(i, j, ctx.clone());
        }

        if let Some(idx) = open_chat_idx {
            self.trigger_load_chat(idx, ctx.clone());
        }

        if self.loading_comparison {
            egui::Window::new("⏳ Loading Side-by-Side Comparison...")
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(&self.status_msg);
                    });
                });
        }

        if self.loading_chat_view {
            egui::Window::new("⏳ Loading Chat History...")
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(&self.status_msg);
                    });
                });
        }

        let active_chat_clone = self.active_chat_view.clone();
        let is_chat_active = active_chat_clone.lock().unwrap().is_some();
        if is_chat_active {
            let title = {
                let lock = active_chat_clone.lock().unwrap();
                let chat = lock.as_ref().unwrap();
                format!("💬 Chat Viewer: {}", chat.backup_name)
            };

            let viewport_id = egui::ViewportId::from_hash_of("chat_viewer_window");
            ctx.show_viewport_immediate(
                viewport_id,
                egui::ViewportBuilder::default()
                    .with_title(title)
                    .with_inner_size([650.0, 700.0]),
                move |ctx, class| {
                    if class == egui::ViewportClass::Immediate {
                        egui::CentralPanel::default()
                            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(20, 20, 25)))
                            .show(ctx, |ui| {
                                let mut chat_lock = active_chat_clone.lock().unwrap();
                                if let Some(ref mut chat_view) = *chat_lock {
                                    // Header controls
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.strong(format!("Chat: {}", chat_view.backup_name));
                                            ui.colored_label(egui::Color32::from_rgb(130, 150, 170), format!("{} messages", chat_view.messages.len()));
                                        });
                                        ui.add_space(20.0);

                                        // Search
                                        ui.label("🔍 Search:");
                                        let search_changed = ui.text_edit_singleline(&mut chat_view.search_query).changed();

                                        if search_changed {
                                            let q = chat_view.search_query.to_lowercase();
                                            chat_view.filtered_indices.clear();
                                            if !q.is_empty() {
                                                for (i, m) in chat_view.messages.iter().enumerate() {
                                                    if m.clean_text.contains(&q) || m.sender.to_lowercase().contains(&q) {
                                                        chat_view.filtered_indices.push(i);
                                                    }
                                                }
                                            }
                                            chat_view.search_matches_count = chat_view.filtered_indices.len();
                                            if chat_view.search_matches_count > 0 {
                                                chat_view.current_search_match_idx = Some(0);
                                                chat_view.scroll_to_row_idx = Some(chat_view.filtered_indices[0]);
                                            } else {
                                                chat_view.current_search_match_idx = None;
                                            }
                                        }

                                        if chat_view.search_matches_count > 0 {
                                            let curr = chat_view.current_search_match_idx.unwrap_or(0);
                                            ui.label(format!("{} of {}", curr + 1, chat_view.search_matches_count));

                                            if ui.button("⬅️").on_hover_text("Previous match").clicked() {
                                                let prev = if curr == 0 { chat_view.search_matches_count - 1 } else { curr - 1 };
                                                chat_view.current_search_match_idx = Some(prev);
                                                chat_view.scroll_to_row_idx = Some(chat_view.filtered_indices[prev]);
                                            }
                                            if ui.button("➡️").on_hover_text("Next match").clicked() {
                                                let next = (curr + 1) % chat_view.search_matches_count;
                                                chat_view.current_search_match_idx = Some(next);
                                                chat_view.scroll_to_row_idx = Some(chat_view.filtered_indices[next]);
                                            }
                                        } else if !chat_view.search_query.is_empty() {
                                            ui.colored_label(egui::Color32::from_rgb(231, 76, 60), "No matches");
                                        }
                                    });
                                    ui.separator();

                                    let num_rows = chat_view.messages.len();
                                    let row_height = 80.0; // Estimated average height of a bubble message row

                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(14, 22, 33)) // Telegram dark background
                                        .inner_margin(12.0)
                                        .show(ui, |ui| {
                                            let mut scroll_area = egui::ScrollArea::vertical()
                                                .id_source("chat_view_scroll_area")
                                                .auto_shrink([false; 2]);

                                            if chat_view.scroll_to_bottom {
                                                let target_y = (num_rows as f32 * 85.0).max(0.0);
                                                scroll_area = scroll_area.scroll_offset(egui::vec2(0.0, target_y));
                                                chat_view.scroll_to_bottom = false;
                                            }

                                            if let Some(target_idx) = chat_view.scroll_to_row_idx {
                                                let spacing_y = ui.spacing().item_spacing.y;
                                                let target_y = (target_idx as f32 * (row_height + spacing_y) - 200.0).max(0.0);
                                                scroll_area = scroll_area.scroll_offset(egui::vec2(0.0, target_y));
                                                chat_view.scroll_to_row_idx = None;
                                            }

                                            scroll_area.show_rows(ui, row_height, num_rows, |ui, row_range| {
                                                for idx in row_range {
                                                    let msg = &chat_view.messages[idx];

                                                    // Parse date string for Date Separator
                                                    let date_part = if msg.timestamp_str.len() >= 10 {
                                                        &msg.timestamp_str[0..10]
                                                    } else {
                                                        ""
                                                    };

                                                    let show_date_header = if idx == 0 {
                                                        true
                                                    } else {
                                                        let prev_msg = &chat_view.messages[idx - 1];
                                                        let prev_msg_date = if prev_msg.timestamp_str.len() >= 10 {
                                                            &prev_msg.timestamp_str[0..10]
                                                        } else {
                                                            ""
                                                        };
                                                        date_part != prev_msg_date
                                                    };

                                                    if show_date_header && !date_part.is_empty() {
                                                        ui.add_space(8.0);
                                                        ui.vertical_centered(|ui| {
                                                            egui::Frame::none()
                                                                .fill(egui::Color32::from_rgba_unmultiplied(16, 30, 47, 180))
                                                                .rounding(12.0)
                                                                .inner_margin(egui::Margin::symmetric(14.0, 4.0))
                                                                .show(ui, |ui| {
                                                                    ui.colored_label(egui::Color32::from_rgb(170, 190, 210), date_part);
                                                                });
                                                        });
                                                        ui.add_space(8.0);
                                                    }

                                                    render_message_bubble(ui, msg, false, false, true);
                                                    ui.add_space(6.0);
                                                }
                                            });
                                        });

                                    if ctx.input(|i| i.viewport().close_requested()) {
                                        *chat_lock = None;
                                    }
                                }
                            });
                    }
                }
            );
        }

        let active_comp_clone = self.active_comparison.clone();
        let is_comp_active = active_comp_clone.lock().unwrap().is_some();
        if is_comp_active {
            let title = {
                let lock = active_comp_clone.lock().unwrap();
                let comp = lock.as_ref().unwrap();
                format!("⚖️ Side-by-Side Comparison: Backup {} vs {}", comp.backup_a_letter, comp.backup_b_letter)
            };

            let viewport_id = egui::ViewportId::from_hash_of("side_by_side_comparison");
            ctx.show_viewport_immediate(
                viewport_id,
                egui::ViewportBuilder::default()
                    .with_title(title)
                    .with_inner_size([950.0, 600.0]),
                move |ctx, class| {
                    if class == egui::ViewportClass::Immediate {
                        egui::CentralPanel::default()
                            .frame(egui::Frame::none().inner_margin(16.0).fill(egui::Color32::from_rgb(20, 20, 25)))
                            .show(ctx, |ui| {
                                let mut comp_lock = active_comp_clone.lock().unwrap();
                                if let Some(ref mut comp) = *comp_lock {
                                    // Header controls
                                    ui.horizontal(|ui| {
                                        ui.label(format!("Comparing overlapping messages. Total messages: {}.", comp.rows.len()));
                                        ui.add_space(20.0);
                                        if !comp.discrepancies.is_empty() {
                                            let curr = comp.current_discrepancy_idx.unwrap_or(0);
                                            ui.label(format!("⚠️ Discrepancy {} of {}", curr + 1, comp.discrepancies.len()));
                                            
                                            if ui.button("⬅️ Previous Missing").clicked() {
                                                let prev_idx = if curr == 0 { comp.discrepancies.len() - 1 } else { curr - 1 };
                                                comp.current_discrepancy_idx = Some(prev_idx);
                                                comp.scroll_to_row_idx = Some(comp.discrepancies[prev_idx]);
                                            }
                                            
                                            if ui.button("Next Missing ➡️").clicked() {
                                                let next_idx = (curr + 1) % comp.discrepancies.len();
                                                comp.current_discrepancy_idx = Some(next_idx);
                                                comp.scroll_to_row_idx = Some(comp.discrepancies[next_idx]);
                                            }
                                        } else {
                                            ui.colored_label(egui::Color32::from_rgb(46, 204, 113), "✅ Perfect alignment! No discrepancies found.");
                                        }
                                    });
                                    ui.separator();

                                    ui.columns(2, |cols| {
                                        cols[0].heading(&comp.backup_a_name);
                                        cols[1].heading(&comp.backup_b_name);
                                    });
                                    ui.separator();

                                    // Scrollable messages list
                                    let num_rows = comp.rows.len();
                                    let row_height = 95.0; // Estimated average height of a bubble message row
                                    
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(14, 22, 33)) // Telegram dark background
                                        .inner_margin(8.0)
                                        .show(ui, |ui| {
                                            let mut scroll_area = egui::ScrollArea::vertical().id_source("compare_scroll_area");
                                            if let Some(target_idx) = comp.scroll_to_row_idx {
                                                let spacing_y = ui.spacing().item_spacing.y;
                                                let target_y = (target_idx as f32 * (row_height + spacing_y) - 200.0).max(0.0);
                                                scroll_area = scroll_area.scroll_offset(egui::vec2(0.0, target_y));
                                                comp.scroll_to_row_idx = None; // Reset it!
                                            }

                                            scroll_area.show_rows(ui, row_height, num_rows, |ui, row_range| {
                                                for idx in row_range {
                                                    let row = &comp.rows[idx];
                                                    ui.columns(2, |cols| {
                                                        // Column A
                                                        cols[0].vertical(|ui| {
                                                            if let Some(ref msg) = row.msg_a {
                                                                render_message_bubble(ui, msg, row.is_discrepancy, true, false);
                                                            } else {
                                                                render_missing_placeholder(ui, "Missing in Backup A");
                                                            }
                                                        });

                                                        // Column B
                                                        cols[1].vertical(|ui| {
                                                            if let Some(ref msg) = row.msg_b {
                                                                render_message_bubble(ui, msg, row.is_discrepancy, false, false);
                                                            } else {
                                                                render_missing_placeholder(ui, "Missing in Backup B");
                                                            }
                                                        });
                                                    });
                                                    ui.add_space(6.0);
                                                }
                                            });
                                        });

                                    if ctx.input(|i| i.viewport().close_requested()) {
                                        *comp_lock = None;
                                    }
                                }
                            });
                    }
                }
            );
        }

        if let Some((i, j)) = compare_pair {
            self.trigger_comparison(i, j, ctx.clone());
        }
    }
}

fn get_color_by_idx(idx: usize) -> egui::Color32 {
    let colors = [
        egui::Color32::from_rgb(99, 102, 241),   // Premium Indigo
        egui::Color32::from_rgb(16, 185, 129),   // Premium Emerald Green
        egui::Color32::from_rgb(168, 85, 247),   // Premium Vibrant Purple
        egui::Color32::from_rgb(245, 158, 11),   // Premium Amber/Gold
        egui::Color32::from_rgb(244, 63, 94),    // Premium Rose/Coral
        egui::Color32::from_rgb(6, 182, 212),    // Premium Cyan
    ];
    colors[idx % colors.len()]
}

fn draw_gantt_chart(ui: &mut egui::Ui, backups: &[BackupInfo]) {
    let mut min_time = i64::MAX;
    let mut max_time = i64::MIN;
    for b in backups {
        if let Some(min_t) = b.min_unix {
            if min_t < min_time { min_time = min_t; }
        }
        if let Some(max_t) = b.max_unix {
            if max_t > max_time { max_time = max_t; }
        }
    }

    if min_time == i64::MAX || max_time == i64::MIN || min_time == max_time {
        ui.label("No valid timeline data available.");
        return;
    }

    let span = max_time - min_time;
    let pad = (span as f64 * 0.05) as i64;
    let timeline_min = min_time - pad;
    let timeline_max = max_time + pad;
    let timeline_span = timeline_max - timeline_min;

    let height = (backups.len() * 40 + 40) as f32;
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );

    let painter = ui.painter_at(rect);

    // Draw frame background
    painter.rect_filled(
        rect,
        5.0,
        egui::Color32::from_rgb(30, 30, 35),
    );

    // Timeline boundaries within the canvas
    let chart_left = rect.left() + 90.0;
    let chart_right = rect.right() - 15.0;
    let chart_width = chart_right - chart_left;

    if chart_width <= 0.0 {
        return;
    }

    // Grid ticks (Years or Months)
    let min_dt = Utc.timestamp_opt(timeline_min, 0).single().unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    let max_dt = Utc.timestamp_opt(timeline_max, 0).single().unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());

    let start_year = min_dt.format("%Y").to_string().parse::<i32>().unwrap_or(2015);
    let end_year = max_dt.format("%Y").to_string().parse::<i32>().unwrap_or(2026);

    let font_id = egui::FontId::proportional(11.0);

    if start_year == end_year {
        // Draw monthly ticks if it's the same year
        for m in 1..=12 {
            if let Some(month_dt) = Utc.with_ymd_and_hms(start_year, m, 1, 0, 0, 0).single() {
                let month_unix = month_dt.timestamp();
                if month_unix >= timeline_min && month_unix <= timeline_max {
                    let x_pct = (month_unix - timeline_min) as f32 / timeline_span as f32;
                    let x_pos = chart_left + x_pct * chart_width;

                    painter.line_segment(
                        [egui::pos2(x_pos, rect.top() + 5.0), egui::pos2(x_pos, rect.bottom() - 25.0)],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 65)),
                    );

                    let month_name = match m {
                        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr", 5 => "May", 6 => "Jun",
                        7 => "Jul", 8 => "Aug", 9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
                        _ => "",
                    };
                    painter.text(
                        egui::pos2(x_pos, rect.bottom() - 12.0),
                        egui::Align2::CENTER_CENTER,
                        format!("{} {}", month_name, start_year),
                        font_id.clone(),
                        egui::Color32::from_rgb(180, 180, 180),
                    );
                }
            }
        }
    } else {
        // Draw annual ticks
        for y in start_year..=end_year {
            if let Some(year_dt) = Utc.with_ymd_and_hms(y, 1, 1, 0, 0, 0).single() {
                let year_unix = year_dt.timestamp();
                if year_unix >= timeline_min && year_unix <= timeline_max {
                    let x_pct = (year_unix - timeline_min) as f32 / timeline_span as f32;
                    let x_pos = chart_left + x_pct * chart_width;

                    painter.line_segment(
                        [egui::pos2(x_pos, rect.top() + 5.0), egui::pos2(x_pos, rect.bottom() - 25.0)],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 65)),
                    );

                    painter.text(
                        egui::pos2(x_pos, rect.bottom() - 12.0),
                        egui::Align2::CENTER_CENTER,
                        y.to_string(),
                        font_id.clone(),
                        egui::Color32::from_rgb(180, 180, 180),
                    );
                }
            }
        }
    }

    // Paint Gantt Range Bars
    for (idx, b) in backups.iter().enumerate() {
        let letter = (b'A' + idx as u8) as char;
        let label = format!("Backup {}", letter);

        // Draw row label on the left
        let label_pos = egui::pos2(rect.left() + 10.0, rect.top() + (idx * 40 + 27) as f32);
        painter.text(
            label_pos,
            egui::Align2::LEFT_CENTER,
            &label,
            egui::FontId::proportional(12.0),
            get_color_by_idx(idx),
        );

        let b_min = b.min_unix.unwrap_or(timeline_min);
        let b_max = b.max_unix.unwrap_or(timeline_max);

        let start_pct = (b_min - timeline_min) as f32 / timeline_span as f32;
        let end_pct = (b_max - timeline_min) as f32 / timeline_span as f32;

        let y_top = rect.top() + (idx * 40 + 15) as f32;
        let y_bottom = y_top + 25.0;

        let bar_left = chart_left + start_pct * chart_width;
        let bar_right = chart_left + end_pct * chart_width;

        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(bar_left.max(chart_left), y_top),
            egui::pos2(bar_right.min(chart_right), y_bottom),
        );

        let color = get_color_by_idx(idx);
        painter.rect_filled(bar_rect, 4.0, color);

        // Interactivity & Tooltips
        if let Some(hover_pos) = ui.ctx().input(|i| i.pointer.hover_pos()) {
            if bar_rect.contains(hover_pos) {
                painter.rect_stroke(bar_rect, 4.0, egui::Stroke::new(1.5, egui::Color32::WHITE));

                egui::show_tooltip(ui.ctx(), egui::Id::new(format!("gantt_tooltip_{}", idx)), |ui| {
                    ui.style_mut().spacing.item_spacing.y = 4.0;
                    ui.colored_label(color, format!("Backup {}", letter));
                    ui.label(format!("Path: {}", b.path));
                    ui.label(format!("Range: {} to {}", b.min_ts, b.max_ts));
                    ui.label(format!("Messages: {} msgs", b.count));
                    if let Some(ref stats) = b.media_stats {
                        ui.label(format!(
                            "Media: 📷 {}/{} | 🎥 {}/{} | 🎤 {}/{} | 📂 {}/{}",
                            stats.photos_resolved, stats.photos_count,
                            stats.videos_resolved, stats.videos_count,
                            stats.voice_resolved, stats.voice_count,
                            stats.files_resolved, stats.files_count
                        ));
                    }
                });
            }
        }
    }
}

fn is_outgoing_sender(sender: &str) -> bool {
    let s = sender.to_lowercase();
    let me_name = std::env::var("USER").unwrap_or_default().to_lowercase();
    let me_rev: String = me_name.chars().rev().collect();
    s == "me" || s == "self" || s == "outgoing" || (!me_name.is_empty() && (s == me_name || s == me_rev))
}

fn get_sender_color(sender: &str) -> egui::Color32 {
    let mut hash: u32 = 0;
    for c in sender.chars() {
        hash = hash.wrapping_add(c as u32).wrapping_mul(31);
    }
    let colors = [
        egui::Color32::from_rgb(224, 112, 112), // Premium Red
        egui::Color32::from_rgb(112, 224, 112), // Premium Green
        egui::Color32::from_rgb(240, 176, 80),  // Gold/Orange
        egui::Color32::from_rgb(80, 160, 240),  // Light Blue
        egui::Color32::from_rgb(176, 112, 224), // Purple
        egui::Color32::from_rgb(112, 224, 224), // Cyan
    ];
    colors[(hash as usize) % colors.len()]
}

fn render_telegram_media_box(ui: &mut egui::Ui, msg: &BackupMessage) {
    let mt = msg.media_type.as_deref().unwrap_or("file");
    let mp = msg.media_path.as_deref().unwrap_or("");
    let file_name = std::path::Path::new(mp)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(mp);

    let icon = match mt {
        "photo" => "📷",
        "video" => "🎥",
        "voice_message" => "🎤",
        "audio_file" => "🎵",
        "sticker" => "🖼️",
        _ => "📄",
    };

    egui::Frame::none()
        .fill(egui::Color32::from_rgb(15, 25, 35))
        .rounding(4.0)
        .inner_margin(6.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(icon).size(18.0));
                ui.vertical(|ui| {
                    ui.strong(egui::RichText::new(file_name).color(egui::Color32::from_rgb(98, 172, 232)).size(11.0));
                    ui.colored_label(egui::Color32::from_rgb(150, 170, 190), format!("Type: {}", mt));
                });
            });
        });
}

fn render_message_bubble(
    ui: &mut egui::Ui,
    msg: &BackupMessage,
    is_discrepancy: bool,
    is_left: bool,
    is_single_chat: bool,
) {
    let (bubble_color, border_color, border_width) = if is_discrepancy {
        (
            egui::Color32::from_rgb(45, 25, 25), 
            egui::Color32::from_rgb(231, 76, 60), 
            1.0
        )
    } else if is_left {
        (
            egui::Color32::from_rgb(24, 37, 51), 
            egui::Color32::from_rgb(33, 47, 61), 
            0.0
        )
    } else {
        (
            egui::Color32::from_rgb(43, 82, 120), 
            egui::Color32::from_rgb(52, 101, 145), 
            0.0
        )
    };

    let is_outgoing = is_single_chat && is_outgoing_sender(&msg.sender);

    let rounding = if is_single_chat {
        if is_outgoing {
            egui::Rounding { nw: 10.0, ne: 10.0, sw: 10.0, se: 2.0 }
        } else {
            egui::Rounding { nw: 10.0, ne: 10.0, sw: 2.0, se: 10.0 }
        }
    } else {
        egui::Rounding::same(10.0)
    };

    let width = ui.available_width();
    let max_bubble_width = if is_single_chat { width * 0.75 } else { width };

    ui.horizontal(|ui| {
        if is_single_chat && is_outgoing {
            ui.add_space(width * 0.22);
        }

        ui.allocate_ui(egui::vec2(max_bubble_width, 0.0), |ui| {
            ui.set_max_width(max_bubble_width);

            egui::Frame::none()
                .fill(bubble_color)
                .stroke(egui::Stroke::new(border_width, border_color))
                .rounding(rounding)
                .inner_margin(egui::Margin::symmetric(10.0, 8.0))
                .show(ui, |ui| {
                    let show_sender_name = !is_single_chat || !is_outgoing;
                    if show_sender_name {
                        let sender_color = get_sender_color(&msg.sender);
                        ui.strong(egui::RichText::new(&msg.sender).color(sender_color).size(12.5));
                        if is_discrepancy && !is_single_chat {
                            ui.add_space(4.0);
                            ui.colored_label(egui::Color32::from_rgb(231, 76, 60), "⚠️ Missing opposite");
                        }
                        ui.add_space(2.0);
                    } else if is_discrepancy {
                        ui.colored_label(egui::Color32::from_rgb(231, 76, 60), "⚠️ Missing opposite");
                    }

                    ui.add(egui::Label::new(egui::RichText::new(&msg.text).color(egui::Color32::WHITE).size(13.0)).wrap(true));

                    if msg.media_type.is_some() || msg.media_path.is_some() {
                        ui.add_space(6.0);
                        render_telegram_media_box(ui, msg);
                    }

                    ui.add_space(2.0);

                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let check_marks = if !is_single_chat && is_discrepancy { "⚠️" } else { "✓✓" };
                            ui.colored_label(egui::Color32::from_rgb(110, 140, 160), check_marks);
                            ui.add_space(4.0);
                            
                            let time_part = if msg.timestamp_str.len() >= 19 {
                                msg.timestamp_str[11..19].to_string()
                            } else if msg.timestamp_str.len() >= 16 {
                                format!("{}:00", &msg.timestamp_str[11..16])
                            } else {
                                msg.timestamp_str.clone()
                            };
                            ui.colored_label(egui::Color32::from_rgb(120, 145, 165), time_part);
                            ui.add_space(10.0);
                            ui.colored_label(egui::Color32::from_rgb(90, 110, 130), format!("ID: {}", msg.message_id));
                        });
                    });
                });
        });

        if is_single_chat && !is_outgoing {
            ui.add_space(width * 0.22);
        }
    });
}

fn render_missing_placeholder(ui: &mut egui::Ui, text: &str) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(20, 15, 15))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 40, 40)))
        .rounding(10.0)
        .inner_margin(egui::Margin::symmetric(10.0, 12.0))
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.colored_label(egui::Color32::from_rgb(231, 76, 60), format!("🚫 {}", text));
            });
        });
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("tgbackman"),
        ..Default::default()
    };
    
    eframe::run_native(
        "tgbackman_overlaps",
        options,
        Box::new(|cc| {
            let mut app = OverlapApp::default();
            app.trigger_load_data(cc.egui_ctx.clone());
            Box::new(app)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_execution_time() {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let path = format!("/media/{}/1b/Telegram Backup/Telegram (unofficial)/Aug_25_2015-present/.telegram_backup/+447926045540", user);
        if let Some(ts) = get_backup_execution_time(&path) {
            println!("FOUND TIME: {} -> {}", ts, format_unix_to_ts(ts));
            assert!(ts > 0);
        } else {
            panic!("Could not get backup execution time!");
        }
    }

    #[test]
    fn test_run_inventory_performance() {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let db_path = format!("/media/{}/1b/sqlitedb/telegram_backup.db", user);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let start = std::time::Instant::now();
        let result = run_inventory(&conn);
        let duration = start.elapsed();
        println!("run_inventory took: {:?}", duration);
        assert!(result.is_ok());
        let groups = result.unwrap();
        println!("Loaded {} groups", groups.len());
    }

    #[test]
    fn test_compute_media_stats_split_and_unofficial() {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let db_path = format!("/media/{}/1b/sqlitedb/telegram_backup.db", user);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let groups = run_inventory(&conn).unwrap();
        
        // Find split chat
        let mut found_split = false;
        for g in &groups {
            for b in &g.backups {
                if b.chat_id == "2015-02-27T19-06-48Z__2015-06-14T00-52-25Z" {
                    let stats = b.compute_media_stats(&db_path);
                    println!("Split stats: photos_resolved={}/{}, videos_resolved={}/{}, voice_resolved={}/{}, files_resolved={}/{}",
                        stats.photos_resolved, stats.photos_count,
                        stats.videos_resolved, stats.videos_count,
                        stats.voice_resolved, stats.voice_count,
                        stats.files_resolved, stats.files_count
                    );
                    assert!(stats.photos_resolved > 0);
                    assert!(stats.videos_resolved > 0);
                    assert!(stats.voice_resolved > 0);
                    assert!(stats.files_resolved > 0);
                    found_split = true;
                }
            }
        }
        assert!(found_split, "Should have found the split backup chat");
        
        // Find unofficial chat
        let mut found_unofficial = false;
        for g in &groups {
            for b in &g.backups {
                if b.chat_id == "group_293206044" {
                    let stats = b.compute_media_stats(&db_path);
                    println!("Unofficial stats: photos_resolved={}/{}, videos_resolved={}/{}, voice_resolved={}/{}, files_resolved={}/{}",
                        stats.photos_resolved, stats.photos_count,
                        stats.videos_resolved, stats.videos_count,
                        stats.voice_resolved, stats.voice_count,
                        stats.files_resolved, stats.files_count
                    );
                    assert!(stats.photos_resolved > 0);
                    assert!(stats.videos_resolved > 0);
                    assert!(stats.voice_resolved > 0);
                    assert!(stats.files_resolved > 0);
                    found_unofficial = true;
                }
            }
        }
        assert!(found_unofficial, "Should have found the unofficial backup chat");
    }
}

