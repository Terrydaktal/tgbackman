use chrono::{DateTime, Utc};
use colored::*;
use db_search::query::SearchQuery;

pub(crate) fn replace_case_insensitive(text: &str, name: &str, replacement: &str) -> String {
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

pub(crate) fn apply_sanitisation(text: &str, pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .fold(text.to_string(), |current, (target, replacement)| {
            replace_case_insensitive(&current, target, replacement)
        })
}

pub(crate) fn normalize_sender(sender: &str) -> String {
    let trimmed = sender.to_lowercase();
    let trimmed = trimmed.trim();
    if trimmed.starts_with("siwel") || trimmed.starts_with("lewis") {
        "lewis".to_string()
    } else if trimmed.starts_with("deleted account") || trimmed.starts_with("deleted") {
        "deleted".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn format_ts(iso_str: &str) -> String {
    if iso_str.is_empty() {
        return "?".to_string();
    }
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&iso_str.replace("Z", "+00:00")) {
        let local_dt: DateTime<Utc> = parsed.with_timezone(&Utc);
        return local_dt.format("%Y-%m-%d %H:%M:%S").to_string();
    }
    iso_str.replace('T', " ").replace('Z', "")
}

pub(crate) fn highlight_term(text: &str, query: &str, dim_non_matches: bool) -> String {
    if query.is_empty() {
        return if dim_non_matches {
            text.dimmed().to_string()
        } else {
            text.to_string()
        };
    }
    let tokens = SearchQuery::new(query);
    let mut terms = Vec::new();
    for token in tokens.tokens() {
        if token.quoted {
            if !token.text.is_empty() {
                terms.push(token.text.to_lowercase());
            }
        } else {
            let t_lower = token.text.to_lowercase();
            if t_lower == "or" || t_lower == "and" || t_lower == "not" {
                continue;
            }
            let cleaned = token
                .text
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '*');
            let cleaned_no_star = cleaned.trim_end_matches('*');
            if !cleaned_no_star.is_empty() {
                terms.push(cleaned_no_star.to_lowercase());
            }
        }
    }
    if terms.is_empty() {
        return if dim_non_matches {
            text.dimmed().to_string()
        } else {
            text.to_string()
        };
    }
    let text_lower = text.to_lowercase();
    let mut ranges = Vec::new();
    for term in &terms {
        let mut last_idx = 0;
        while let Some(start_idx) = text_lower[last_idx..].find(term) {
            let abs_start = last_idx + start_idx;
            ranges.push((abs_start, abs_start + term.len()));
            last_idx = abs_start + 1;
        }
    }
    if ranges.is_empty() {
        return if dim_non_matches {
            text.dimmed().to_string()
        } else {
            text.to_string()
        };
    }
    ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    let mut merged = Vec::new();
    let mut current = ranges[0];
    for range in ranges.into_iter().skip(1) {
        if range.0 <= current.1 {
            current.1 = current.1.max(range.1);
        } else {
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);
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
        result.push_str(&text[start..end].red().bold().to_string());
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
