//! Telegram-style, read-only presentation for canonical database messages.

use eframe::egui;

use crate::model::{ActiveChatView, MessageSearchResult};
use crate::ui::{highlighted_text_job, match_preview, render_message_bubble};

pub(crate) const GLOBAL_SEARCH_RESULT_ROW_HEIGHT: f32 = 92.0;

pub(crate) enum ChatViewerAction {
    Close,
    LoadOlder,
    LoadNewer,
    LoadLatest,
    ClearSearch,
    Search(String),
    JumpToSearchResult(usize),
}

fn initials(name: &str) -> String {
    let mut words = name
        .split_whitespace()
        .filter_map(|word| word.chars().next());
    let first = words.next().unwrap_or('?');
    let second = words.next();
    match second {
        Some(second) => format!("{first}{second}"),
        None => first.to_string(),
    }
    .to_uppercase()
}

fn compact_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let preview: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{preview}…")
    } else {
        preview
    }
}

pub(crate) fn render_global_result(
    ui: &mut egui::Ui,
    result: &MessageSearchResult,
    query: &str,
) -> egui::Response {
    let text = if result.text.trim().is_empty() {
        result
            .media_type
            .as_deref()
            .map(|kind| format!("[{kind}]"))
            .unwrap_or_else(|| "[empty message]".to_string())
    } else {
        match_preview(&result.text, query, 45)
    };
    let time = result
        .timestamp_str
        .split('T')
        .next()
        .unwrap_or(&result.timestamp_str);

    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), GLOBAL_SEARCH_RESULT_ROW_HEIGHT),
        egui::Sense::click(),
    );
    if ui.is_rect_visible(rect) {
        if response.hovered() {
            ui.painter()
                .rect_filled(rect, 5.0, egui::Color32::from_rgb(25, 38, 51));
        }
        ui.painter().line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 49, 61)),
        );
        ui.allocate_ui_at_rect(rect.shrink2(egui::vec2(7.0, 7.0)), |ui| {
            ui.horizontal(|ui| {
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(43, 106, 153))
                    .rounding(20.0)
                    .inner_margin(7.0)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(initials(&result.chat_name))
                                .strong()
                                .color(egui::Color32::WHITE),
                        );
                    });
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.strong(compact_text(&result.chat_name, 34));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.colored_label(egui::Color32::from_rgb(130, 151, 170), time);
                        });
                    });
                    ui.colored_label(
                        egui::Color32::from_rgb(82, 171, 238),
                        compact_text(&result.sender, 34),
                    );
                    ui.add(
                        egui::Label::new(highlighted_text_job(
                            &text,
                            query,
                            12.0,
                            egui::Color32::from_rgb(184, 199, 211),
                        ))
                        .wrap(true),
                    );
                });
            });
        });
    }
    response
}

pub(crate) fn render_global_result_placeholder(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), GLOBAL_SEARCH_RESULT_ROW_HEIGHT),
        egui::Sense::hover(),
    );
    if ui.is_rect_visible(rect) {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Loading match…",
            egui::FontId::proportional(12.0),
            egui::Color32::from_rgb(130, 151, 170),
        );
        ui.painter().line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 49, 61)),
        );
    }
}

pub(crate) fn render_chat_view(
    ui: &mut egui::Ui,
    chat: &mut ActiveChatView,
    loading: bool,
) -> Option<ChatViewerAction> {
    let mut action = None;
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(23, 33, 43))
        .inner_margin(egui::Margin::symmetric(12.0, 9.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button("←")
                    .on_hover_text("Return to backup management")
                    .clicked()
                {
                    action = Some(ChatViewerAction::Close);
                }
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(43, 106, 153))
                    .rounding(24.0)
                    .inner_margin(9.0)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(initials(&chat.backup_name))
                                .strong()
                                .color(egui::Color32::WHITE),
                        );
                    });
                ui.vertical(|ui| {
                    ui.heading(&chat.backup_name);
                    ui.colored_label(
                        egui::Color32::from_rgb(137, 157, 176),
                        format!(
                            "{} saved messages · {} on this page · read-only backup",
                            chat.total_messages,
                            chat.messages.len()
                        ),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if loading {
                        ui.add(egui::Spinner::new());
                    }
                    let search_response = ui.add_sized(
                        [240.0, 28.0],
                        egui::TextEdit::singleline(&mut chat.search_query)
                            .hint_text("Search in this chat"),
                    );
                    let enter_pressed = search_response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    if search_response.changed() {
                        chat.highlight_query.clone_from(&chat.search_query);
                        chat.search_results.clear();
                        chat.current_search_match_idx = None;
                        chat.search_error = None;
                        action = Some(ChatViewerAction::ClearSearch);
                    }
                    if (ui
                        .button("🔍")
                        .on_hover_text("Search this entire chat")
                        .clicked()
                        || enter_pressed)
                        && !chat.search_query.trim().is_empty()
                    {
                        action = Some(ChatViewerAction::Search(chat.search_query.clone()));
                    }
                });
            });

            if chat.searching
                || !chat.search_results.is_empty()
                || chat.search_error.is_some()
                || !chat.search_query.is_empty()
            {
                ui.separator();
                ui.horizontal(|ui| {
                    if chat.searching {
                        ui.add(egui::Spinner::new());
                        ui.label("Searching every saved message in this chat…");
                    } else if let Some(error) = chat.search_error.as_deref() {
                        ui.colored_label(
                            egui::Color32::from_rgb(235, 112, 102),
                            format!("Search failed: {error}"),
                        );
                    } else if chat.search_query.trim().is_empty() {
                        ui.label("Enter text to search this chat.");
                    } else if chat.search_results.is_empty() {
                        ui.label("No saved messages matched.");
                    } else {
                        let current = chat.current_search_match_idx.unwrap_or(0);
                        ui.label(format!(
                            "{} of {} matches",
                            current + 1,
                            chat.total_search_matches
                        ));
                        if ui.button("↑").on_hover_text("Previous match").clicked() {
                            let next = if current == 0 {
                                chat.search_results.len() - 1
                            } else {
                                current - 1
                            };
                            chat.current_search_match_idx = Some(next);
                            action = Some(ChatViewerAction::JumpToSearchResult(next));
                        }
                        if ui.button("↓").on_hover_text("Next match").clicked() {
                            let next = (current + 1) % chat.search_results.len();
                            chat.current_search_match_idx = Some(next);
                            action = Some(ChatViewerAction::JumpToSearchResult(next));
                        }
                        if ui.button("Jump to match").clicked() {
                            action = Some(ChatViewerAction::JumpToSearchResult(current));
                        }
                        ui.colored_label(
                            egui::Color32::from_rgb(137, 157, 176),
                            "All matches loaded · newest first",
                        );
                    }
                });
            }
        });

    egui::Frame::none()
        .fill(egui::Color32::from_rgb(14, 22, 33))
        .inner_margin(egui::Margin::symmetric(14.0, 8.0))
        .show(ui, |ui| {
            if chat.has_older {
                ui.vertical_centered(|ui| {
                    if ui
                        .add_enabled(!loading, egui::Button::new("Load older messages"))
                        .clicked()
                    {
                        action = Some(ChatViewerAction::LoadOlder);
                    }
                });
                ui.add_space(6.0);
            }

            let should_stick_to_bottom = chat.scroll_to_bottom;
            let focus_message_id = chat.focus_message_id;
            let reserved_navigation_height = if chat.has_newer { 36.0 } else { 0.0 };
            egui::ScrollArea::vertical()
                .id_source(format!("chat_messages_{}", chat.chat_id))
                .auto_shrink([false; 2])
                .stick_to_bottom(should_stick_to_bottom)
                .max_height((ui.available_height() - reserved_navigation_height).max(120.0))
                .show(ui, |ui| {
                    for (index, message) in chat.messages.iter().enumerate() {
                        let date = message
                            .timestamp_str
                            .split('T')
                            .next()
                            .unwrap_or(&message.timestamp_str);
                        let previous_date = index.checked_sub(1).and_then(|previous| {
                            chat.messages[previous].timestamp_str.split('T').next()
                        });
                        if !date.is_empty() && previous_date != Some(date) {
                            ui.add_space(6.0);
                            ui.vertical_centered(|ui| {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgba_unmultiplied(16, 30, 47, 220))
                                    .rounding(12.0)
                                    .inner_margin(egui::Margin::symmetric(13.0, 4.0))
                                    .show(ui, |ui| {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(178, 196, 211),
                                            date,
                                        );
                                    });
                            });
                            ui.add_space(6.0);
                        }

                        let focused = focus_message_id == Some(message.message_id);
                        let response = egui::Frame::none()
                            .stroke(if focused {
                                egui::Stroke::new(2.0, egui::Color32::from_rgb(82, 171, 238))
                            } else {
                                egui::Stroke::NONE
                            })
                            .rounding(8.0)
                            .inner_margin(if focused { 4.0 } else { 0.0 })
                            .show(ui, |ui| {
                                render_message_bubble(
                                    ui,
                                    message,
                                    false,
                                    false,
                                    true,
                                    Some(&chat.highlight_query),
                                );
                            })
                            .response;
                        if focused {
                            response.scroll_to_me(Some(egui::Align::Center));
                        }
                        ui.add_space(5.0);
                    }
                });
            chat.scroll_to_bottom = false;
            chat.focus_message_id = None;

            if chat.has_newer {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!loading, egui::Button::new("Load newer messages"))
                        .clicked()
                    {
                        action = Some(ChatViewerAction::LoadNewer);
                    }
                    if ui
                        .add_enabled(!loading, egui::Button::new("Jump to latest"))
                        .clicked()
                    {
                        action = Some(ChatViewerAction::LoadLatest);
                    }
                });
            }
        });
    action
}

#[cfg(test)]
mod tests {
    use super::{compact_text, initials};

    #[test]
    fn presentation_helpers_are_unicode_safe() {
        assert_eq!(initials("Example Conversation"), "EC");
        assert_eq!(initials("🦸 Super"), "🦸S");
        assert_eq!(compact_text("one   two three", 7), "one two…");
    }
}
