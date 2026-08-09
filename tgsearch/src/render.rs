use crate::search_support::query::SearchQuery;
use chrono::{DateTime, Utc};
use colored::*;

fn lowercase_with_ranges(text: &str) -> (String, Vec<(usize, usize)>) {
    let mut lowered = String::new();
    let mut ranges = Vec::new();
    for (start, character) in text.char_indices() {
        let end = start + character.len_utf8();
        let lowered_character = character.to_lowercase().collect::<String>();
        lowered.push_str(&lowered_character);
        ranges.extend(std::iter::repeat_n((start, end), lowered_character.len()));
    }
    (lowered, ranges)
}

pub(crate) fn replace_case_insensitive(text: &str, name: &str, replacement: &str) -> String {
    if name.is_empty() {
        return text.to_string();
    }
    let mut result = String::new();
    let (text_lower, source_ranges) = lowercase_with_ranges(text);
    let name_lower = name.to_lowercase();
    let mut lower_cursor = 0;
    let mut original_last = 0;
    while let Some(start_idx) = text_lower[lower_cursor..].find(&name_lower) {
        let absolute_start = lower_cursor + start_idx;
        let absolute_end = absolute_start + name_lower.len();
        let original_start = source_ranges[absolute_start].0;
        let original_end = source_ranges[absolute_end - 1].1;
        result.push_str(&text[original_last..original_start]);
        result.push_str(replacement);
        original_last = original_end;
        lower_cursor = absolute_end;
        // Continue searching after the corresponding lower-case range.
        if absolute_end >= text_lower.len() {
            break;
        }
    }
    result.push_str(&text[original_last..]);
    result
}

pub(crate) fn apply_sanitisation(text: &str, pairs: &[(String, String)]) -> String {
    let replaced = pairs
        .iter()
        .fold(text.to_string(), |current, (target, replacement)| {
            replace_case_insensitive(&current, target, replacement)
        });
    #[derive(Clone, Copy)]
    enum ControlState {
        Normal,
        Esc,
        Csi,
        String,
        StringEsc,
    }

    let mut state = ControlState::Normal;
    let mut sanitized = String::with_capacity(replaced.len());
    for character in replaced.chars() {
        match state {
            ControlState::Normal => {
                let code = character as u32;
                match code {
                    0x1b => state = ControlState::Esc,
                    0x9b => state = ControlState::Csi,
                    0x90 | 0x9d | 0x9e | 0x9f => state = ControlState::String,
                    _ if character == '\n' || character == '\t' => sanitized.push(character),
                    _ if code >= 0x20 && code != 0x7f && !(0x80..=0x9f).contains(&code) => {
                        sanitized.push(character)
                    }
                    _ => {}
                }
            }
            ControlState::Esc => match character {
                '[' => state = ControlState::Csi,
                ']' | 'P' | '^' | '_' => state = ControlState::String,
                _ => state = ControlState::Normal,
            },
            ControlState::Csi => {
                if ('@'..='~').contains(&character) {
                    state = ControlState::Normal;
                }
            }
            ControlState::String => match character {
                '\u{7}' => state = ControlState::Normal,
                '\u{1b}' => state = ControlState::StringEsc,
                _ => {}
            },
            ControlState::StringEsc => {
                state = if character == '\\' {
                    ControlState::Normal
                } else {
                    ControlState::String
                };
            }
        }
    }
    sanitized
}

pub(crate) fn normalize_sender(sender: &str) -> String {
    let trimmed = sender.to_lowercase();
    let trimmed = trimmed.trim();
    if trimmed.starts_with("deleted account") || trimmed.starts_with("deleted") {
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
    let (text_lower, source_ranges) = lowercase_with_ranges(text);
    let mut ranges = Vec::new();
    for term in &terms {
        let mut last_idx = 0;
        while let Some(start_idx) = text_lower[last_idx..].find(term) {
            let abs_start = last_idx + start_idx;
            let abs_end = abs_start + term.len();
            if abs_end <= source_ranges.len() {
                ranges.push((source_ranges[abs_start].0, source_ranges[abs_end - 1].1));
            }
            last_idx = abs_end;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_preserves_unicode_boundaries() {
        assert_eq!(
            replace_case_insensitive("Älex and ÄLEX", "äLEX", "redacted"),
            "redacted and redacted"
        );
    }

    #[test]
    fn sanitisation_removes_terminal_controls_but_keeps_layout() {
        let value = apply_sanitisation("name\u{1b}[31m\tline\nnext\u{9b}x", &[]);
        assert_eq!(value, "name\tline\nnext");
    }

    #[test]
    fn timestamps_are_rendered_in_utc_and_invalid_values_are_preserved() {
        assert_eq!(format_ts("2024-01-01T00:00:00Z"), "2024-01-01 00:00:00");
        assert_eq!(format_ts("not-a-timestamp"), "not-a-timestamp");
    }
}
