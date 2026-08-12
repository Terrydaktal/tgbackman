//! EgUI rendering primitives.

use chrono::{TimeZone, Utc};
use eframe::egui;

use crate::model::{BackupInfo, BackupMessage};
use std::ops::Range;

pub(crate) fn get_color_by_idx(idx: usize) -> egui::Color32 {
    let colors = [
        egui::Color32::from_rgb(99, 102, 241), // Premium Indigo
        egui::Color32::from_rgb(16, 185, 129), // Premium Emerald Green
        egui::Color32::from_rgb(168, 85, 247), // Premium Vibrant Purple
        egui::Color32::from_rgb(245, 158, 11), // Premium Amber/Gold
        egui::Color32::from_rgb(244, 63, 94),  // Premium Rose/Coral
        egui::Color32::from_rgb(6, 182, 212),  // Premium Cyan
    ];
    colors[idx % colors.len()]
}

pub(crate) fn draw_gantt_chart(ui: &mut egui::Ui, backups: &[BackupInfo]) {
    let mut min_time = i64::MAX;
    let mut max_time = i64::MIN;
    for b in backups {
        if let Some(min_t) = b.min_unix {
            if min_t < min_time {
                min_time = min_t;
            }
        }
        if let Some(max_t) = b.max_unix {
            if max_t > max_time {
                max_time = max_t;
            }
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
    painter.rect_filled(rect, 5.0, egui::Color32::from_rgb(30, 30, 35));

    // Timeline boundaries within the canvas
    let chart_left = rect.left() + 90.0;
    let chart_right = rect.right() - 15.0;
    let chart_width = chart_right - chart_left;

    if chart_width <= 0.0 {
        return;
    }

    // Grid ticks (Years or Months)
    let min_dt = Utc
        .timestamp_opt(timeline_min, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
    let max_dt = Utc
        .timestamp_opt(timeline_max, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());

    let start_year = min_dt
        .format("%Y")
        .to_string()
        .parse::<i32>()
        .unwrap_or(2015);
    let end_year = max_dt
        .format("%Y")
        .to_string()
        .parse::<i32>()
        .unwrap_or(2026);

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
                        [
                            egui::pos2(x_pos, rect.top() + 5.0),
                            egui::pos2(x_pos, rect.bottom() - 25.0),
                        ],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 65)),
                    );

                    let month_name = match m {
                        1 => "Jan",
                        2 => "Feb",
                        3 => "Mar",
                        4 => "Apr",
                        5 => "May",
                        6 => "Jun",
                        7 => "Jul",
                        8 => "Aug",
                        9 => "Sep",
                        10 => "Oct",
                        11 => "Nov",
                        12 => "Dec",
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
                        [
                            egui::pos2(x_pos, rect.top() + 5.0),
                            egui::pos2(x_pos, rect.bottom() - 25.0),
                        ],
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

                egui::show_tooltip(
                    ui.ctx(),
                    egui::Id::new(format!("gantt_tooltip_{}", idx)),
                    |ui| {
                        ui.style_mut().spacing.item_spacing.y = 4.0;
                        ui.colored_label(color, format!("Backup {}", letter));
                        ui.label(format!("Path: {}", b.path));
                        ui.label(format!("Range: {} to {}", b.min_ts, b.max_ts));
                        ui.label(format!("Messages: {} msgs", b.count));
                        if let Some(ref stats) = b.media_stats {
                            ui.label(format!(
                                "Media: 📷 {}/{} | 🎥 {}/{} | 🎤 {}/{} | 📂 {}/{}",
                                stats.photos_resolved,
                                stats.photos_count,
                                stats.videos_resolved,
                                stats.videos_count,
                                stats.voice_resolved,
                                stats.voice_count,
                                stats.files_resolved,
                                stats.files_count
                            ));
                        }
                    },
                );
            }
        }
    }
}

pub(crate) fn is_outgoing_sender(sender: &str) -> bool {
    let s = sender.to_lowercase();
    matches!(s.as_str(), "me" | "self" | "you" | "outgoing")
}

fn lowercase_prefix_end(text: &str, lowercase_needle: &str) -> Option<usize> {
    let mut lowercase = String::new();
    for (index, character) in text.char_indices() {
        lowercase.extend(character.to_lowercase());
        if lowercase.len() >= lowercase_needle.len() {
            return lowercase
                .starts_with(lowercase_needle)
                .then_some(index + character.len_utf8());
        }
    }
    None
}

pub(crate) fn match_ranges(text: &str, query: &str) -> Vec<Range<usize>> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(str::to_lowercase)
        .filter(|term| !term.is_empty())
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut position = 0;
    while position < text.len() {
        let best_end = terms
            .iter()
            .filter_map(|term| lowercase_prefix_end(&text[position..], term))
            .max();
        if let Some(relative_end) = best_end {
            let end = position + relative_end;
            ranges.push(position..end);
            position = end;
        } else {
            position += text[position..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(1);
        }
    }
    ranges
}

pub(crate) fn highlighted_text_job(
    text: &str,
    query: &str,
    size: f32,
    color: egui::Color32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let normal = egui::TextFormat {
        font_id: egui::FontId::proportional(size),
        color,
        ..Default::default()
    };
    let highlighted = egui::TextFormat {
        font_id: egui::FontId::proportional(size),
        color: egui::Color32::WHITE,
        background: egui::Color32::from_rgb(180, 122, 20),
        ..Default::default()
    };
    let ranges = match_ranges(text, query);
    let mut cursor = 0;
    for range in ranges {
        if cursor < range.start {
            job.append(&text[cursor..range.start], 0.0, normal.clone());
        }
        job.append(&text[range.clone()], 0.0, highlighted.clone());
        cursor = range.end;
    }
    if cursor < text.len() {
        job.append(&text[cursor..], 0.0, normal);
    }
    job
}

pub(crate) fn match_preview(text: &str, query: &str, context_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let Some(first_match) = match_ranges(&compact, query).into_iter().next() else {
        let mut characters = compact.chars();
        let preview: String = characters.by_ref().take(context_chars * 2).collect();
        return if characters.next().is_some() {
            format!("{preview}…")
        } else {
            preview
        };
    };
    let prefix_chars = compact[..first_match.start].chars().count();
    let start_char = prefix_chars.saturating_sub(context_chars);
    let start = compact
        .char_indices()
        .nth(start_char)
        .map(|(index, _)| index)
        .unwrap_or(0);
    let match_end_chars = compact[..first_match.end].chars().count();
    let end_char = match_end_chars.saturating_add(context_chars);
    let end = compact
        .char_indices()
        .nth(end_char)
        .map(|(index, _)| index)
        .unwrap_or(compact.len());
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        &compact[start..end],
        if end < compact.len() { "…" } else { "" }
    )
}

pub(crate) fn get_sender_color(sender: &str) -> egui::Color32 {
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

pub(crate) fn render_telegram_media_box(ui: &mut egui::Ui, msg: &BackupMessage) {
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
                    ui.strong(
                        egui::RichText::new(file_name)
                            .color(egui::Color32::from_rgb(98, 172, 232))
                            .size(11.0),
                    );
                    ui.colored_label(
                        egui::Color32::from_rgb(150, 170, 190),
                        format!("Type: {}", mt),
                    );
                });
            });
        });
}

fn reply_preview_text(text: &str, media_type: Option<&str>) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !compact.is_empty() {
        let mut chars = compact.chars();
        let preview: String = chars.by_ref().take(140).collect();
        return if chars.next().is_some() {
            format!("{preview}…")
        } else {
            preview
        };
    }
    media_type
        .map(|kind| format!("[{kind}]"))
        .unwrap_or_else(|| "No preserved preview".to_string())
}

fn render_reply_preview(ui: &mut egui::Ui, reply: &crate::model::ReplyPreview) {
    let accent = if reply.missing {
        egui::Color32::from_rgb(150, 160, 170)
    } else {
        egui::Color32::from_rgb(82, 171, 238)
    };
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(18, 31, 43))
        .stroke(egui::Stroke::new(1.0, accent))
        .rounding(4.0)
        .inner_margin(egui::Margin::symmetric(7.0, 4.0))
        .show(ui, |ui| {
            let mut heading = if let Some(story_id) = reply.story_id {
                format!("↩ Story #{story_id}")
            } else if reply.missing {
                match reply.message_id {
                    Some(message_id) => format!("↩ Original unavailable · ID {message_id}"),
                    None => "↩ Reply context unavailable".to_string(),
                }
            } else {
                format!(
                    "↩ {}",
                    reply.sender.as_deref().unwrap_or("Referenced message")
                )
            };
            if let Some(chat_name) = reply.chat_name.as_deref() {
                heading.push_str(&format!(" · {chat_name}"));
            } else if reply.missing
                && let (Some(kind), Some(peer_id)) = (reply.peer_kind.as_deref(), reply.peer_id)
            {
                heading.push_str(&format!(" · {kind}:{peer_id}"));
            }
            if let Some(topic_id) = reply.topic_id {
                heading.push_str(&format!(" · topic #{topic_id}"));
            }
            ui.colored_label(accent, egui::RichText::new(heading).size(11.0).strong());
            ui.colored_label(
                egui::Color32::from_rgb(190, 205, 218),
                egui::RichText::new(reply_preview_text(&reply.text, reply.media_type.as_deref()))
                    .size(11.0),
            );
        });
}

fn spaced_type_name(value: &str) -> String {
    let value = value.strip_prefix("MessageAction").unwrap_or(value);
    let mut result = String::new();
    for character in value.chars() {
        if character.is_uppercase() && !result.is_empty() {
            result.push(' ');
        }
        result.push(character);
    }
    result
}

fn service_action_text(msg: &BackupMessage) -> String {
    if !msg.text.trim().is_empty() && msg.text.trim() != "[service]" {
        return msg.text.clone();
    }
    let Some(action) = msg.action_json.as_deref() else {
        return "Service message".to_string();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(action) else {
        return "Service message".to_string();
    };
    let kind = value
        .get("_")
        .and_then(serde_json::Value::as_str)
        .map(spaced_type_name)
        .unwrap_or_else(|| "Service message".to_string());
    if let Some(seconds) = value.get("duration").and_then(serde_json::Value::as_i64) {
        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;
        let seconds = seconds % 60;
        let duration = if hours > 0 {
            format!("{hours}h {minutes}m")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        };
        format!("{kind} · {duration}")
    } else {
        kind
    }
}

fn reaction_badges(reactions_json: Option<&str>) -> Vec<String> {
    let Some(reactions_json) = reactions_json else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(reactions_json) else {
        return Vec::new();
    };
    value
        .get("results")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|result| {
            let count = result
                .get("count")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            if count <= 0 {
                return None;
            }
            let reaction = result.get("reaction").unwrap_or(result);
            let glyph = reaction
                .get("emoticon")
                .or_else(|| reaction.get("emoji"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| {
                    if reaction.get("_").and_then(serde_json::Value::as_str) == Some("ReactionPaid")
                    {
                        "⭐"
                    } else {
                        "✨"
                    }
                });
            Some(if count == 1 {
                glyph.to_string()
            } else {
                format!("{glyph} {count}")
            })
        })
        .collect()
}

pub(crate) fn render_message_bubble(
    ui: &mut egui::Ui,
    msg: &BackupMessage,
    is_discrepancy: bool,
    is_left: bool,
    is_single_chat: bool,
    highlight_query: Option<&str>,
) {
    if is_single_chat && msg.message_type.as_deref() == Some("service") {
        ui.vertical_centered(|ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(16, 30, 47, 220))
                .rounding(12.0)
                .inner_margin(egui::Margin::symmetric(12.0, 5.0))
                .show(ui, |ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(178, 196, 211),
                        service_action_text(msg),
                    );
                });
        });
        return;
    }
    let is_outgoing = is_single_chat && (msg.is_outgoing || is_outgoing_sender(&msg.sender));
    let (bubble_color, border_color, border_width) = if is_discrepancy {
        (
            egui::Color32::from_rgb(45, 25, 25),
            egui::Color32::from_rgb(231, 76, 60),
            1.0,
        )
    } else if (is_single_chat && !is_outgoing) || (!is_single_chat && is_left) {
        (
            egui::Color32::from_rgb(24, 37, 51),
            egui::Color32::from_rgb(33, 47, 61),
            0.0,
        )
    } else {
        (
            egui::Color32::from_rgb(43, 82, 120),
            egui::Color32::from_rgb(52, 101, 145),
            0.0,
        )
    };

    let rounding = if is_single_chat {
        if is_outgoing {
            egui::Rounding {
                nw: 10.0,
                ne: 10.0,
                sw: 10.0,
                se: 2.0,
            }
        } else {
            egui::Rounding {
                nw: 10.0,
                ne: 10.0,
                sw: 2.0,
                se: 10.0,
            }
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
                        ui.strong(
                            egui::RichText::new(&msg.sender)
                                .color(sender_color)
                                .size(12.5),
                        );
                        if is_discrepancy && !is_single_chat {
                            ui.add_space(4.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(231, 76, 60),
                                "⚠️ Missing opposite",
                            );
                        }
                        ui.add_space(2.0);
                    } else if is_discrepancy {
                        ui.colored_label(
                            egui::Color32::from_rgb(231, 76, 60),
                            "⚠️ Missing opposite",
                        );
                    }

                    if let Some(forwarded_from) = msg
                        .forwarded_from
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        ui.colored_label(
                            egui::Color32::from_rgb(82, 171, 238),
                            egui::RichText::new(format!("Forwarded from {forwarded_from}"))
                                .size(11.5),
                        );
                        ui.add_space(3.0);
                    }

                    if let Some(reply) = msg.reply.as_ref() {
                        render_reply_preview(ui, reply);
                        ui.add_space(5.0);
                    }

                    let query = highlight_query.unwrap_or_default();
                    if query.trim().is_empty() {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&msg.text)
                                    .color(egui::Color32::WHITE)
                                    .size(13.0),
                            )
                            .wrap(true),
                        );
                    } else {
                        ui.add(
                            egui::Label::new(highlighted_text_job(
                                &msg.text,
                                query,
                                13.0,
                                egui::Color32::WHITE,
                            ))
                            .wrap(true),
                        );
                    }

                    if msg.media_type.is_some() || msg.media_path.is_some() {
                        ui.add_space(6.0);
                        render_telegram_media_box(ui, msg);
                    }

                    let reaction_badges = reaction_badges(msg.reactions_json.as_deref());
                    if !reaction_badges.is_empty() {
                        ui.add_space(4.0);
                        ui.horizontal_wrapped(|ui| {
                            for reaction in reaction_badges {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(31, 55, 73))
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        egui::Color32::from_rgb(58, 102, 132),
                                    ))
                                    .rounding(12.0)
                                    .inner_margin(egui::Margin::symmetric(8.0, 3.0))
                                    .show(ui, |ui| {
                                        ui.label(reaction);
                                    });
                            }
                        });
                    }

                    ui.add_space(2.0);

                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !is_single_chat && is_discrepancy {
                                ui.colored_label(egui::Color32::from_rgb(110, 140, 160), "⚠️");
                                ui.add_space(4.0);
                            }

                            let time_part = msg
                                .timestamp_str
                                .get(11..19)
                                .map(str::to_string)
                                .or_else(|| {
                                    msg.timestamp_str
                                        .get(11..16)
                                        .map(|time| format!("{time}:00"))
                                })
                                .unwrap_or_else(|| msg.timestamp_str.clone());
                            ui.colored_label(egui::Color32::from_rgb(120, 145, 165), time_part);
                            if msg.edit_timestamp.is_some() {
                                ui.colored_label(egui::Color32::from_rgb(120, 145, 165), "edited");
                            }
                            ui.add_space(10.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(90, 110, 130),
                                format!("ID: {}", msg.message_id),
                            );
                        });
                    });
                });
        });

        if is_single_chat && !is_outgoing {
            ui.add_space(width * 0.22);
        }
    });
}

pub(crate) fn render_missing_placeholder(ui: &mut egui::Ui, text: &str) {
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

#[cfg(test)]
mod metadata_tests {
    use super::{match_preview, match_ranges, reaction_badges, spaced_type_name};

    #[test]
    fn telegram_metadata_is_presented_without_raw_type_names() {
        assert_eq!(spaced_type_name("MessageActionPhoneCall"), "Phone Call");
        let reactions = reaction_badges(Some(
            r#"{"results":[{"count":3,"reaction":{"_":"ReactionEmoji","emoticon":"👍"}},{"count":1,"reaction":{"_":"ReactionPaid"}}]}"#,
        ));
        assert_eq!(reactions, vec!["👍 3", "⭐"]);
    }

    #[test]
    fn search_highlighting_is_case_insensitive_and_unicode_safe() {
        assert_eq!(match_ranges("Hello hello", "HEL"), vec![0..3, 6..9]);
        assert_eq!(match_ranges("Straße", "stra"), vec![0..4]);
        let preview = match_preview(
            "a long prefix that should not hide the important matched phrase from view",
            "matched",
            8,
        );
        assert!(preview.contains("matched"));
        assert!(preview.starts_with('…'));
    }
}
