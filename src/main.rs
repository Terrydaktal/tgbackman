use clap::Parser;
use colored::*;
use rusqlite::{params, Connection, Result};
use chrono::{DateTime, Utc};

#[derive(Parser, Debug)]
#[command(author, version, about = "Search Telegram SQLite backups with context windows.")]
struct Cli {
    /// Path to the SQLite database file
    db_path: String,

    /// The full-text search query term
    query: String,

    /// Number of context messages to show before and after each match
    #[arg(short, long, default_value_t = 3)]
    context: i64,

    /// Maximum number of search matches to display
    #[arg(short, long, default_value_t = 10, allow_negative_numbers = true)]
    limit: i64,

    /// Filter search results by chat name or chat ID (substring filter)
    #[arg(short, long)]
    chat: Option<String>,

    /// Disable colored ANSI terminal output
    #[arg(long)]
    no_color: bool,

    /// Strip the date and time from the output
    #[arg(long)]
    no_time: bool,

    /// Match exact words only (disable default partial/prefix matching)
    #[arg(long)]
    exact: bool,

    /// Deduplicate identical messages matching in multiple backups
    #[arg(long)]
    dedupe: bool,

    /// Replace names in output using a file containing target:replacement pairs (one per line)
    #[arg(long)]
    sanitise: Option<String>,

    /// Disable detailed chat/message headers in output (shows simple block dividers instead)
    #[arg(long)]
    no_header: bool,
}

struct MatchRow {
    _id: i64,
    chat_name: String,
    chat_id: String,
    message_id: i64,
    backup_path: Option<String>,
    timestamp_unix: Option<i64>,
    sender: Option<String>,
    text: Option<String>,
    media_path: Option<String>,
}

struct ContextRow {
    _message_id: i64,
    timestamp: Option<String>,
    sender: String,
    text: String,
    media_type: Option<String>,
    media_path: Option<String>,
    is_match: bool,
}

struct MergedGroup {
    chat_id: String,
    chat_name: String,
    backup_path: Option<String>,
    match_ids: Vec<i64>,
    min_id: i64,
    max_id: i64,
}

fn replace_case_insensitive(text: &str, name: &str, replacement: &str) -> String {
    if name.is_empty() {
        return text.to_string();
    }
    
    let mut result = String::new();
    let text_lower = text.to_lowercase();
    let name_lower = name.to_lowercase();
    
    let mut last_idx = 0;
    while let Some(start_idx) = text_lower[last_idx..].find(&name_lower) {
        let absolute_start = last_idx + start_idx;
        let absolute_end = absolute_start + name_lower.len();
        
        result.push_str(&text[last_idx..absolute_start]);
        result.push_str(replacement);
        
        last_idx = absolute_end;
    }
    
    result.push_str(&text[last_idx..]);
    result
}

fn apply_sanitisation(text: &str, pairs: &[(String, String)]) -> String {
    let mut current = text.to_string();
    for (target, replacement) in pairs {
        current = replace_case_insensitive(&current, target, replacement);
    }
    current
}

fn parse_query_tokens(query: &str) -> Vec<(String, bool)> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    
    for c in query.chars() {
        if c == '"' {
            if in_quotes {
                if !current.trim().is_empty() {
                    tokens.push((current.clone(), true));
                }
                current.clear();
                in_quotes = false;
            } else {
                if !current.trim().is_empty() {
                    for w in current.split_whitespace() {
                        tokens.push((w.to_string(), false));
                    }
                }
                current.clear();
                in_quotes = true;
            }
        } else {
            current.push(c);
        }
    }
    
    if !current.trim().is_empty() {
        if in_quotes {
            tokens.push((current.clone(), true));
        } else {
            for w in current.split_whitespace() {
                tokens.push((w.to_string(), false));
            }
        }
    }
    
    tokens
}

fn normalize_sender(sender: &str) -> String {
    let s = sender.to_lowercase();
    let trimmed = s.trim();
    if trimmed.starts_with("siwel") || trimmed.starts_with("lewis") {
        "lewis".to_string()
    } else if trimmed.starts_with("deleted account") || trimmed.starts_with("deleted") {
        "deleted".to_string()
    } else {
        trimmed.to_string()
    }
}

fn format_ts(iso_str: &str) -> String {
    if iso_str.is_empty() {
        return "?".to_string();
    }
    // Attempt to parse ISO-8601 UTC string (e.g. "2026-05-24T21:16:00Z")
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&iso_str.replace("Z", "+00:00")) {
        let local_dt: DateTime<Utc> = parsed.with_timezone(&Utc);
        return local_dt.format("%Y-%m-%d %H:%M:%S").to_string();
    }
    iso_str.replace('T', " ").replace('Z', "")
}

fn highlight_term(text: &str, query: &str, dim_non_matches: bool) -> String {
    if query.is_empty() {
        return if dim_non_matches { text.dimmed().to_string() } else { text.to_string() };
    }
    
    let tokens = parse_query_tokens(query);
    let mut terms = Vec::new();
    
    for (t, is_quoted) in tokens {
        if is_quoted {
            if !t.is_empty() {
                terms.push(t.to_lowercase());
            }
        } else {
            let t_lower = t.to_lowercase();
            if t_lower == "or" || t_lower == "and" || t_lower == "not" {
                continue;
            }
            let cleaned = t.trim_matches(|c: char| !c.is_alphanumeric() && c != '*');
            let cleaned_no_star = cleaned.trim_end_matches('*');
            if !cleaned_no_star.is_empty() {
                terms.push(cleaned_no_star.to_lowercase());
            }
        }
    }
    
    if terms.is_empty() {
        return if dim_non_matches { text.dimmed().to_string() } else { text.to_string() };
    }
    
    let text_lower = text.to_lowercase();
    let mut ranges = Vec::new();
    
    for term in &terms {
        let mut last_idx = 0;
        while let Some(start_idx) = text_lower[last_idx..].find(term) {
            let abs_start = last_idx + start_idx;
            let abs_end = abs_start + term.len();
            ranges.push((abs_start, abs_end));
            last_idx = abs_start + 1; // Slide by 1 to find all occurrences
        }
    }
    
    if ranges.is_empty() {
        return if dim_non_matches { text.dimmed().to_string() } else { text.to_string() };
    }
    
    // Sort ranges by start index, and then by end index descending
    ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    
    // Merge overlapping ranges
    let mut merged = Vec::new();
    let mut current = ranges[0];
    
    for r in ranges.into_iter().skip(1) {
        if r.0 <= current.1 {
            // Overlaps or contiguous, extend current
            if r.1 > current.1 {
                current.1 = r.1;
            }
        } else {
            merged.push(current);
            current = r;
        }
    }
    merged.push(current);
    
    // Render the text using merged ranges
    let mut result = String::new();
    let mut last_idx = 0;
    
    for (start, end) in merged {
        if start > last_idx {
            let before = &text[last_idx..start];
            if dim_non_matches {
                result.push_str(&before.dimmed().to_string());
            } else {
                result.push_str(before);
            }
        }
        let matched = &text[start..end];
        result.push_str(&matched.red().bold().to_string());
        last_idx = end;
    }
    
    if last_idx < text.len() {
        let after = &text[last_idx..];
        if dim_non_matches {
            result.push_str(&after.dimmed().to_string());
        } else {
            result.push_str(after);
        }
    }
    
    result
}

fn prepare_fts_query(query: &str, exact: bool) -> String {
    if exact {
        return query.to_string();
    }
    
    let tokens = parse_query_tokens(query);
    let mut parts = Vec::new();
    
    for (text, is_quoted) in tokens {
        if is_quoted {
            parts.push(format!("\"{}\"", text));
        } else {
            if text.to_lowercase() == "or" {
                parts.push("OR".to_string());
                continue;
            }
            let cleaned = text.trim_matches(|c: char| !c.is_alphanumeric() && c != '*');
            if !cleaned.is_empty() {
                if cleaned.ends_with('*') {
                    parts.push(cleaned.to_string());
                } else {
                    parts.push(format!("{}*", cleaned));
                }
            }
        }
    }
    parts.join(" ")
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut sanitise_pairs = Vec::new();
    if let Some(file_path) = &args.sanitise {
        match std::fs::read_to_string(file_path) {
            Ok(content) => {
                for line in content.lines() {
                    let line_trimmed = line.trim();
                    if line_trimmed.is_empty() || line_trimmed.starts_with('#') {
                        continue;
                    }
                    let delimiter = if line_trimmed.contains(':') {
                        ':'
                    } else if line_trimmed.contains('=') {
                        '='
                    } else {
                        continue;
                    };
                    let subparts: Vec<&str> = line_trimmed.splitn(2, delimiter).collect();
                    if subparts.len() == 2 {
                        let target = subparts[0].trim();
                        let replacement = subparts[1].trim();
                        if !target.is_empty() {
                            let target_lower = target.to_lowercase();
                            if target_lower == "lewis" || target_lower == "siwel" {
                                sanitise_pairs.push(("lewis".to_string(), replacement.to_string()));
                                sanitise_pairs.push(("siwel".to_string(), replacement.to_string()));
                            } else {
                                sanitise_pairs.push((target.to_string(), replacement.to_string()));
                            }
                        }
                    }
                }
            }
            Err(_) => {
                eprintln!(
                    "{}",
                    format!("Error: Could not read sanitise file: {}", file_path).red()
                );
                std::process::exit(1);
            }
        }
    }

    // Disable coloring if flag is present
    if args.no_color {
        control::set_override(false);
    }

    let db_path = &args.db_path;
    if !std::path::Path::new(db_path).exists() {
        eprintln!(
            "{}",
            format!("Error: Database file does not exist: {}", db_path).red()
        );
        std::process::exit(1);
    }

    // Connect to database in read-only mode using query parameters
    let conn = match Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) {
        Ok(c) => c,
        Err(_) => Connection::open(db_path)?,
    };

    // 1. Resolve matching message entry points
    let fts_query = prepare_fts_query(&args.query, args.exact);
    let mut chat_filter_clause = String::new();
    let mut params_vec: Vec<rusqlite::types::Value> = vec![rusqlite::types::Value::Text(fts_query)];

    if let Some(chat_sub) = &args.chat {
        chat_filter_clause = "AND (m.chat_id LIKE ?2 OR c.chat_name LIKE ?3)".to_string();
        let sub = format!("%{}%", chat_sub);
        params_vec.push(rusqlite::types::Value::Text(sub.clone()));
        params_vec.push(rusqlite::types::Value::Text(sub));
    }

    let sql_limit = if args.dedupe {
        if args.limit < 0 {
            -1
        } else {
            args.limit * 10
        }
    } else {
        args.limit
    };

    let limit_param_idx = params_vec.len() + 1;
    params_vec.push(rusqlite::types::Value::Integer(sql_limit));

    let find_matches_query = format!(
        "SELECT m.id, m.chat_id, m.message_id, c.chat_name, c.backup_path, m.timestamp_unix, m.sender, m.text, m.media_path
         FROM messages m
         JOIN chats c ON m.chat_id = c.chat_id
         WHERE m.id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?1)
         {}
         ORDER BY m.timestamp_unix DESC
         LIMIT ?{}",
        chat_filter_clause, limit_param_idx
    );

    let mut stmt = conn.prepare(&find_matches_query)?;
    let match_rows = stmt.query_map(rusqlite::params_from_iter(params_vec.iter()), |row| {
        Ok(MatchRow {
            _id: row.get(0)?,
            chat_id: row.get(1)?,
            message_id: row.get(2)?,
            chat_name: row.get(3)?,
            backup_path: row.get(4)?,
            timestamp_unix: row.get(5)?,
            sender: row.get(6)?,
            text: row.get(7)?,
            media_path: row.get(8)?,
        })
    })?;

    let mut matches: Vec<MatchRow> = Vec::new();
    for r in match_rows {
        let row = r?;
        if args.dedupe {
            let mut is_dup = false;
            for accepted in &matches {
                let same_sender = match (&row.sender, &accepted.sender) {
                    (Some(s1), Some(s2)) => {
                        let n1 = normalize_sender(s1);
                        let n2 = normalize_sender(s2);
                        n1 == n2 || n1 == "deleted" || n2 == "deleted"
                    }
                    (None, None) => true,
                    _ => false,
                };
                let same_content = match (&row.text, &accepted.text) {
                    (Some(t1), Some(t2)) => {
                        if t1.is_empty() && t2.is_empty() {
                            row.media_path == accepted.media_path
                        } else if t1 == t2 {
                            true
                        } else if !t1.is_empty() && !t2.is_empty() {
                            let is_media_marker1 = t1 == "[file]" || t1 == "[photo]";
                            let is_media_marker2 = t2 == "[file]" || t2 == "[photo]";
                            if (is_media_marker1 || is_media_marker2) && row.media_path == accepted.media_path {
                                true
                            } else if t1.len() >= 10 && t2.len() >= 10 && (t1.contains(t2) || t2.contains(t1)) {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                let same_time = match (row.timestamp_unix, accepted.timestamp_unix) {
                    (Some(ts1), Some(ts2)) => {
                        let diff = (ts1 - ts2).abs();
                        diff <= 15 || 
                        (diff >= 3585 && diff <= 3615) || 
                        (diff >= 7185 && diff <= 7215)
                    }
                    _ => false,
                };
                if same_sender && same_content && same_time {
                    is_dup = true;
                    break;
                }
            }
            if !is_dup {
                matches.push(row);
            }
        } else {
            matches.push(row);
        }
    }

    if args.dedupe && args.limit >= 0 {
        matches.truncate(args.limit as usize);
    }

    if matches.is_empty() {
        println!("{}", "No matching messages found.".yellow());
        return Ok(());
    }

    // Query the total count of matches across the database for high-end UX feedback
    let count_query = format!(
        "SELECT COUNT(*), COUNT(DISTINCT m.chat_id)
         FROM messages m
         JOIN chats c ON m.chat_id = c.chat_id
         WHERE m.id IN (SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?1)
         {}",
        chat_filter_clause
    );

    let mut count_stmt = conn.prepare(&count_query)?;
    let count_params = &params_vec[0..params_vec.len() - 1];
    let mut count_rows = count_stmt.query(rusqlite::params_from_iter(count_params.iter()))?;
    
    let mut total_matches = 0;
    let mut total_chats = 0;
    if let Some(row) = count_rows.next()? {
        total_matches = row.get::<_, i64>(0)?;
        total_chats = row.get::<_, i64>(1)?;
    }

    println!(
        "{}",
        format!(
            "Showing {} of {} matches across {} chats (window size: +/-{}):",
            matches.len(),
            total_matches,
            total_chats,
            args.context
        )
        .bold()
        .green()
    );

    // 2. Group matches into non-overlapping context window segments (merging overlapping ranges)
    let mut chat_order = Vec::new();
    let mut chat_to_matches: std::collections::HashMap<String, Vec<MatchRow>> = std::collections::HashMap::new();

    for m in matches {
        if !chat_to_matches.contains_key(&m.chat_id) {
            chat_order.push((m.chat_id.clone(), m.chat_name.clone(), m.backup_path.clone()));
        }
        chat_to_matches.entry(m.chat_id.clone()).or_default().push(m);
    }

    let mut merged_groups = Vec::new();

    for (chat_id, _chat_name, _backup_path) in chat_order {
        if let Some(mut group_matches) = chat_to_matches.remove(&chat_id) {
            // Sort matches chronologically/ascending by message_id
            group_matches.sort_by_key(|m| m.message_id);

            let mut current_group: Option<MergedGroup> = None;

            for m in group_matches {
                match &mut current_group {
                    None => {
                        current_group = Some(MergedGroup {
                            chat_id: m.chat_id.clone(),
                            chat_name: m.chat_name.clone(),
                            backup_path: m.backup_path.clone(),
                            match_ids: vec![m.message_id],
                            min_id: m.message_id,
                            max_id: m.message_id,
                        });
                    }
                    Some(g) => {
                        // Merge if overlapping or directly contiguous
                        if m.message_id - g.max_id <= 2 * args.context + 1 {
                            g.match_ids.push(m.message_id);
                            g.max_id = m.message_id;
                        } else {
                            merged_groups.push(current_group.take().unwrap());
                            current_group = Some(MergedGroup {
                                chat_id: m.chat_id.clone(),
                                chat_name: m.chat_name.clone(),
                                backup_path: m.backup_path.clone(),
                                match_ids: vec![m.message_id],
                                min_id: m.message_id,
                                max_id: m.message_id,
                            });
                        }
                    }
                }
            }

            if let Some(g) = current_group {
                merged_groups.push(g);
            }
        }
    }

    // 3. Query context window for each merged group
    let mut context_stmt = conn.prepare(
        "SELECT m.message_id, m.timestamp, m.sender, m.text, m.media_type, m.media_path
         FROM messages m
         WHERE m.chat_id = ?1 
           AND m.message_id BETWEEN ?2 AND ?3
         ORDER BY m.message_id ASC",
    )?;

    for g in merged_groups {
        let low_bound = g.min_id - args.context;
        let high_bound = g.max_id + args.context;

        let rows = context_stmt.query_map(
            params![g.chat_id, low_bound, high_bound],
            |row| {
                let msg_id: i64 = row.get(0)?;
                let is_m = g.match_ids.contains(&msg_id);
                Ok(ContextRow {
                    _message_id: msg_id,
                    timestamp: row.get(1)?,
                    sender: row.get(2)?,
                    text: row.get(3)?,
                    media_type: row.get(4)?,
                    media_path: row.get(5)?,
                    is_match: is_m,
                })
            },
        )?;

        println!();
        let backup_p = g.backup_path.as_deref().unwrap_or("Unknown path");
        let header_name = apply_sanitisation(&g.chat_name, &sanitise_pairs);
        let header = if g.match_ids.len() == 1 {
            format!(" Chat: {} | Message ID: {} | Path: {} ", header_name, g.min_id, backup_p)
        } else {
            format!(
                " Chat: {} | Message IDs: {}-{} ({} matches) | Path: {} ",
                header_name,
                g.min_id,
                g.max_id,
                g.match_ids.len(),
                backup_p
            )
        };
        if args.no_header {
            let divider = "-".repeat(127);
            println!("{}", divider.dimmed());
        } else {
            let divider = "-".repeat(std::cmp::max(60, header.len() + 4));
            println!("{}", divider.dimmed());
            println!("{}", format!("| {} |", header).bold().cyan());
            println!("{}", divider.dimmed());
        }

        for r_res in rows {
            let r = r_res?;
            let raw_ts = r.timestamp.unwrap_or_default();
            let formatted_date = format_ts(&raw_ts);

            let time_prefix = if args.no_time {
                "".to_string()
            } else {
                if r.is_match {
                    format!("[{}] ", formatted_date).green().to_string()
                } else {
                    format!("[{}] ", formatted_date).dimmed().to_string()
                }
            };

            let mut display_text = r.text.clone();
            if let Some(m_path) = &r.media_path {
                let m_type = r.media_type.as_deref().unwrap_or("Attachment");
                let m_type_cap = if !m_type.is_empty() {
                    let mut chars = m_type.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                } else {
                    "Attachment".to_string()
                };
                
                if display_text.is_empty() {
                    display_text = format!("[{} Path: {}]", m_type_cap, m_path);
                } else {
                    display_text = format!("{} [{} Path: {}]", display_text, m_type_cap, m_path);
                }
            }

            let display_sender = apply_sanitisation(&r.sender, &sanitise_pairs);
            let display_text_sanitised = apply_sanitisation(&display_text, &sanitise_pairs);
            let highlight_query = apply_sanitisation(&args.query, &sanitise_pairs);

            if r.is_match {
                // Display matching message (no leading arrows or indentation)
                print!("{}", time_prefix);
                print!("{}", format!("{}:", display_sender).bold().cyan());
                println!(" {}", highlight_term(&display_text_sanitised, &highlight_query, false));
            } else {
                // Display context message (no leading indentation)
                print!("{}", time_prefix);
                print!("{}", format!("{}:", display_sender).cyan());
                println!(" {}", highlight_term(&display_text_sanitised, &highlight_query, true));
            }
        }
    }

    println!();
    Ok(())
}
