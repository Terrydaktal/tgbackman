//! Message comparison and timestamp helpers.

use chrono::{TimeZone, Utc};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

pub(crate) fn strip_boundaries(word: &str) -> String {
    let start = word
        .find(|c: char| c.is_alphanumeric())
        .unwrap_or(word.len());
    let sub = &word[start..];
    let end = sub
        .rfind(|c: char| c.is_alphanumeric())
        .map(|idx| {
            let c = sub[idx..].chars().next().unwrap();
            idx + c.len_utf8()
        })
        .unwrap_or(0);
    sub[..end].to_lowercase()
}

// Custom text cleaner for DST and formatting invariant checks
pub(crate) fn clean_text_for_match(text: &str) -> String {
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
    let url_re = URL_RE
        .get_or_init(|| Regex::new(r#"(?i)(?:https?://|tel:|mailto:|tg:)[^\s'"“‘’”<>]+"#).unwrap());

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
pub(crate) fn count_missing_messages(
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
    let mut rows_b = stmt_b.query(rusqlite::params![
        chat_b_id,
        start_unix - 3605,
        end_unix + 3605
    ])?;
    let mut b_by_ts: HashMap<i64, Vec<String>> = HashMap::new();
    while let Some(row) = rows_b.next()? {
        let ts: Option<i64> = row.get(0)?;
        let txt: Option<String> = row.get(1)?;
        if let Some(t) = ts {
            b_by_ts
                .entry(t)
                .or_default()
                .push(clean_text_for_match(&txt.unwrap_or_default()));
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

pub(crate) fn format_unix_to_ts(unix_ts: i64) -> String {
    if let Some(dt) = Utc.timestamp_opt(unix_ts, 0).single() {
        dt.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        "Unknown".to_string()
    }
}
