//! EgUI rendering primitives.

use chrono::{TimeZone, Utc};
use eframe::egui;

use crate::model::{BackupInfo, BackupMessage};

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
    let me_name = std::env::var("USER").unwrap_or_default().to_lowercase();
    let me_rev: String = me_name.chars().rev().collect();
    s == "me"
        || s == "self"
        || s == "outgoing"
        || (!me_name.is_empty() && (s == me_name || s == me_rev))
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

pub(crate) fn render_message_bubble(
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
            1.0,
        )
    } else if is_left {
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

    let is_outgoing = is_single_chat && is_outgoing_sender(&msg.sender);

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

                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&msg.text)
                                .color(egui::Color32::WHITE)
                                .size(13.0),
                        )
                        .wrap(true),
                    );

                    if msg.media_type.is_some() || msg.media_path.is_some() {
                        ui.add_space(6.0);
                        render_telegram_media_box(ui, msg);
                    }

                    ui.add_space(2.0);

                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let check_marks = if !is_single_chat && is_discrepancy {
                                "⚠️"
                            } else {
                                "✓✓"
                            };
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
