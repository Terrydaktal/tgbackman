#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // Hide console window on Windows in release

use chrono::Utc;
use eframe::egui;

mod app;
mod cache;
mod database;
mod inventory;
mod matching;
mod model;
mod ui;
mod viewer;

use app::OverlapApp;
use database::{MESSAGE_SEARCH_PAGE_SIZE, set_chat_ids_blacklisted};
use matching::format_unix_to_ts;
use model::{
    ActiveChatView, CalcMessage, ChatPageRequest, CompareMessage, LoadMessage, MediaCalcMessage,
    MessageSearchMessage, SingleChatMessage,
};
use ui::{draw_gantt_chart, get_color_by_idx, render_message_bubble, render_missing_placeholder};
use viewer::{
    ChatViewerAction, GLOBAL_SEARCH_RESULT_ROW_HEIGHT, render_chat_view, render_global_result,
    render_global_result_placeholder,
};

#[cfg(test)]
use cache::cache_is_fresh;
#[cfg(test)]
use database::{
    apply_authoritative_target_links, blacklisted_chat_ids, run_inventory,
    zero_message_migrated_predecessors,
};
#[cfg(test)]
use inventory::UnionFind;
#[cfg(test)]
use std::collections::HashSet;

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
                    self.loaded_db_path = self.loading_db_path.take();
                    self.loading_data = false;
                    self.filtered_groups.clear();
                    self.load_cache();
                    self.load_media_cache();
                    self.filter_groups();
                    self.load_rx = None;
                }
                Ok(LoadMessage::Error(err)) => {
                    self.status_msg = err;
                    self.loading_db_path = None;
                    self.loading_data = false;
                    self.filtered_groups.clear();
                    self.load_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.loading_data = false;
                    self.loading_db_path = None;
                    self.filtered_groups.clear();
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
                    self.status_msg =
                        "Overlaps calculation completed & cached successfully.".to_string();
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

        // Poll background media stats calculations
        if let Some(ref rx) = self.media_rx {
            match rx.try_recv() {
                Ok(MediaCalcMessage::Progress(msg)) => {
                    self.status_msg = msg;
                }
                Ok(MediaCalcMessage::Finished(results)) => {
                    for group in &mut self.groups {
                        for b in &mut group.backups {
                            let key = format!("{}:{}", b.chat_id, b.path);
                            if let Some(stats) = results.get(&key) {
                                b.media_stats = Some(stats.clone());
                            }
                        }
                    }
                    self.calculating_media = false;
                    self.status_msg =
                        "Media stats calculation completed & cached successfully.".to_string();
                    self.media_rx = None;
                }
                Ok(MediaCalcMessage::Error(err)) => {
                    self.status_msg = format!("Media stats calculation failed: {}", err);
                    self.calculating_media = false;
                    self.media_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.calculating_media = false;
                    self.media_rx = None;
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
                    self.status_msg =
                        "Comparison messages aligned and loaded successfully.".to_string();
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
                Ok(SingleChatMessage::Finished(page)) => {
                    let pending_highlight = self.pending_chat_highlight_query.take();
                    let mut active = self.active_chat_view.lock().unwrap();
                    if let Some(chat_view) =
                        active.as_mut().filter(|view| view.chat_id == page.chat_id)
                    {
                        chat_view.apply_page(page);
                        if let Some(query) = pending_highlight.clone() {
                            chat_view.highlight_query = query;
                        }
                    } else {
                        *active = Some(ActiveChatView {
                            backup_name: page.backup_name,
                            chat_id: page.chat_id,
                            messages: page.messages,
                            total_messages: page.total_messages,
                            has_older: page.has_older,
                            has_newer: page.has_newer,
                            scroll_to_bottom: page.focus_message_id.is_none() && !page.has_newer,
                            search_query: String::new(),
                            highlight_query: pending_highlight.unwrap_or_default(),
                            search_results: Vec::new(),
                            total_search_matches: 0,
                            searching: false,
                            search_error: None,
                            current_search_match_idx: None,
                            focus_message_id: page.focus_message_id,
                            self_sender_aliases: page.self_sender_aliases,
                        });
                    }
                    self.loading_chat_view = false;
                    self.status_msg = "Chat messages loaded successfully.".to_string();
                    self.chat_view_rx = None;
                }
                Ok(SingleChatMessage::Error(err)) => {
                    self.pending_chat_highlight_query = None;
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

        if let Some(ref rx) = self.global_search_rx {
            match rx.try_recv() {
                Ok(MessageSearchMessage::IndexReady {
                    request_id,
                    row_ids,
                    results,
                }) if request_id == self.global_search_request_id => {
                    self.global_search_results.clear();
                    for (position, result) in results.into_iter().enumerate() {
                        self.global_search_results.insert(position, result);
                    }
                    self.global_search_total_matches = row_ids.len();
                    self.global_search_row_ids = row_ids;
                    self.global_searching = false;
                    self.global_search_error = None;
                    self.global_search_rx = None;
                }
                Ok(MessageSearchMessage::Finished {
                    request_id,
                    offset,
                    total_matches,
                    results,
                    done: _,
                }) if request_id == self.global_search_request_id => {
                    if offset == 0 {
                        self.global_search_results.clear();
                    }
                    let page_size = MESSAGE_SEARCH_PAGE_SIZE as usize;
                    let keep_start = offset.saturating_sub(page_size * 4);
                    let keep_end = offset.saturating_add(page_size * 5);
                    self.global_search_results
                        .retain(|position, _| *position >= keep_start && *position < keep_end);
                    for (position, result) in results.into_iter().enumerate() {
                        self.global_search_results.insert(offset + position, result);
                    }
                    self.global_search_total_matches = total_matches;
                    self.global_searching = false;
                    self.global_search_error = None;
                    self.global_search_rx = None;
                }
                Ok(MessageSearchMessage::Error {
                    request_id,
                    offset,
                    message,
                }) if request_id == self.global_search_request_id => {
                    if offset == 0 {
                        self.global_search_results.clear();
                        self.global_search_row_ids.clear();
                        self.global_search_total_matches = 0;
                    }
                    self.global_searching = false;
                    self.global_search_error = Some(message);
                    self.global_search_rx = None;
                }
                Ok(_) | Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.global_searching = false;
                    self.global_search_error =
                        Some("Search worker stopped unexpectedly".to_string());
                    self.global_search_rx = None;
                }
            }
        }

        if let Some(ref rx) = self.chat_search_rx {
            match rx.try_recv() {
                Ok(MessageSearchMessage::Finished {
                    request_id,
                    offset,
                    total_matches,
                    results,
                    done,
                }) if request_id == self.chat_search_request_id => {
                    if let Ok(mut active) = self.active_chat_view.lock()
                        && let Some(chat) = active.as_mut()
                    {
                        let previous_current = chat.current_search_match_idx;
                        if offset == 0 {
                            chat.search_results = results;
                        } else if offset == chat.search_results.len() {
                            chat.search_results.extend(results);
                        }
                        chat.total_search_matches = total_matches;
                        chat.searching = !done;
                        chat.search_error = None;
                        chat.current_search_match_idx = if offset > 0 {
                            previous_current
                        } else if chat.search_results.is_empty() {
                            None
                        } else {
                            Some(0)
                        };
                    }
                    if done {
                        self.chat_search_rx = None;
                    }
                }
                Ok(MessageSearchMessage::Error {
                    request_id,
                    offset,
                    message,
                }) if request_id == self.chat_search_request_id => {
                    if let Ok(mut active) = self.active_chat_view.lock()
                        && let Some(chat) = active.as_mut()
                    {
                        if offset == 0 {
                            chat.search_results.clear();
                            chat.total_search_matches = 0;
                            chat.current_search_match_idx = None;
                        }
                        chat.searching = false;
                        chat.search_error = Some(message);
                    }
                    self.chat_search_rx = None;
                }
                Ok(_) | Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if let Ok(mut active) = self.active_chat_view.lock()
                        && let Some(chat) = active.as_mut()
                    {
                        chat.searching = false;
                        chat.search_error = Some("Search worker stopped unexpectedly".to_string());
                    }
                    self.chat_search_rx = None;
                }
            }
        }

        if let Some(started) = self.global_search_scheduled_at {
            let delay = std::time::Duration::from_millis(300);
            if started.elapsed() >= delay {
                self.trigger_global_message_search(self.search_query.clone(), 0, ctx.clone());
            } else {
                ctx.request_repaint_after(delay.saturating_sub(started.elapsed()));
            }
        }

        // Dark theme adjustments
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(20, 20, 25);
        ctx.set_visuals(visuals);
        let chat_is_open = self
            .active_chat_view
            .lock()
            .map(|view| view.is_some())
            .unwrap_or(false);
        let mut close_chat_view = false;
        let mut open_selected_chat = false;
        let mut open_group_chat = None;
        let mut open_global_result = None;
        let mut global_search_page_request = None;

        // Top Control Panel
        egui::TopBottomPanel::top("control_panel")
            .frame(
                egui::Frame::none()
                    .inner_margin(12.0)
                    .fill(egui::Color32::from_rgb(25, 25, 30)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("📊 tgbackman");
                    if ui
                        .selectable_label(!chat_is_open, "Backup manager")
                        .clicked()
                    {
                        close_chat_view = true;
                    }
                    if ui.selectable_label(chat_is_open, "💬 Messages").clicked() {
                        open_selected_chat = true;
                    }
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
                        ui.label("Computing Overlaps...");
                    } else if ui.button("🔄 Recompute All Overlaps").clicked() {
                        self.recompute_all_overlaps(ctx.clone());
                    }
                    ui.add_space(10.0);
                    if self.calculating_media {
                        ui.add(egui::Spinner::new());
                        ui.label("Computing Media...");
                    } else if ui.button("📊 Recompute Media Counts").clicked() {
                        self.recompute_all_media_stats(ctx.clone());
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(&self.status_msg);
                });
            });

        if close_chat_view {
            if let Ok(mut view) = self.active_chat_view.lock() {
                *view = None;
            }
            self.chat_view_rx = None;
            self.chat_search_rx = None;
            self.loading_chat_view = false;
        } else if open_selected_chat && let Some(group_idx) = self.selected_group_idx {
            self.trigger_load_preferred_chat(group_idx, ctx.clone());
        }

        // Left Side Chat List Panel
        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(320.0)
            .frame(
                egui::Frame::none()
                    .inner_margin(12.0)
                    .fill(egui::Color32::from_rgb(15, 15, 20)),
            )
            .show(ctx, |ui| {
                ui.label("🔍 Search chats and all saved messages:");
                if ui.text_edit_singleline(&mut self.search_query).changed() {
                    self.filter_groups();
                    if self.search_query.trim().is_empty() {
                        self.trigger_global_message_search(String::new(), 0, ctx.clone());
                    } else {
                        self.global_search_scheduled_at = Some(std::time::Instant::now());
                        ctx.request_repaint_after(std::time::Duration::from_millis(300));
                    }
                }
                ui.add_space(8.0);

                let search_active = !self.search_query.trim().is_empty();
                ui.heading("Conversations");
                ui.separator();
                let mut next_selected_idx = None;
                let now = Utc::now().timestamp();
                let conversation_row_height = 22.0;
                let conversation_height = if search_active {
                    (self.filtered_groups.len().min(6) as f32 * conversation_row_height)
                        .min(ui.available_height() * 0.3)
                } else {
                    ui.available_height()
                };
                if self.filtered_groups.is_empty() {
                    ui.colored_label(
                        egui::Color32::from_rgb(130, 151, 170),
                        "No matching conversations.",
                    );
                } else {
                    egui::ScrollArea::vertical()
                        .id_source("conversation_list")
                        .auto_shrink([false, true])
                        .max_height(conversation_height.max(conversation_row_height))
                        .show_rows(
                            ui,
                            conversation_row_height,
                            self.filtered_groups.len(),
                            |ui, visible_rows| {
                                for position in visible_rows {
                                    // A refresh replaces `groups` asynchronously. Keep the
                                    // render loop defensive as well as clearing the index list.
                                    let Some(&idx) = self.filtered_groups.get(position) else {
                                        continue;
                                    };
                                    let Some(group) = self.groups.get(idx) else {
                                        continue;
                                    };
                                    let selected = self.selected_group_idx == Some(idx);
                                    let latest_backup_unix = group
                                        .backups
                                        .iter()
                                        .filter_map(|backup| {
                                            backup.last_backup_run_unix.or(backup.last_backup_unix)
                                        })
                                        .max();
                                    let latest_message_unix =
                                        group.backups.iter().filter_map(|b| b.max_unix).max();
                                    let format_age = |timestamp: Option<i64>| match timestamp {
                                        Some(ts) => {
                                            let days = (now - ts) / 86400;
                                            format!("{}d ago", days.max(0))
                                        }
                                        None => "never".to_string(),
                                    };
                                    let group_blacklisted = group.is_blacklisted();
                                    let item_color = if group_blacklisted {
                                        egui::Color32::from_rgb(72, 74, 82)
                                    } else if group.is_active() {
                                        egui::Color32::from_rgb(46, 204, 113)
                                    } else if group.max_count == 0 {
                                        egui::Color32::from_rgb(130, 135, 145)
                                    } else {
                                        egui::Color32::from_rgb(231, 76, 60)
                                    };
                                    let mut label_text = egui::RichText::new(format!(
                                        "{} ({} msgs) - {} - {}",
                                        group.name,
                                        group.max_count,
                                        format_age(latest_message_unix),
                                        format_age(latest_backup_unix),
                                    ))
                                    .color(item_color);
                                    if group_blacklisted {
                                        label_text = label_text.strikethrough();
                                    }
                                    let response = ui.selectable_label(selected, label_text);
                                    if response.clicked() {
                                        next_selected_idx = Some(idx);
                                    }
                                    if response.double_clicked() && group.max_count > 0 {
                                        next_selected_idx = Some(idx);
                                        open_group_chat = Some(idx);
                                    }
                                }
                            },
                        );
                }
                if let Some(idx) = next_selected_idx {
                    self.select_group(idx);
                    if chat_is_open {
                        open_group_chat = Some(idx);
                    }
                }

                if search_active {
                    ui.add_space(8.0);
                    ui.heading("Messages");
                    ui.separator();
                    let submitted_query_matches =
                        self.global_search_last_submitted == self.search_query.trim();
                    if !submitted_query_matches {
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new());
                            ui.label("Preparing full-archive search…");
                        });
                    } else if let Some(error) = self.global_search_error.as_deref() {
                        ui.colored_label(
                            egui::Color32::from_rgb(235, 112, 102),
                            format!("Search failed: {error}"),
                        );
                    } else if self.global_search_total_matches == 0 {
                        if self.global_searching {
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new());
                                ui.label("Searching the full archive…");
                            });
                        } else {
                            ui.label("No saved messages matched.");
                        }
                    } else {
                        ui.horizontal(|ui| {
                            if self.global_searching {
                                ui.add(egui::Spinner::new());
                                ui.label("Loading visible matches…");
                            } else {
                                ui.colored_label(
                                    egui::Color32::from_rgb(130, 151, 170),
                                    format!(
                                        "{} matches · scroll anywhere",
                                        self.global_search_total_matches
                                    ),
                                );
                            }
                        });
                        let result_height = (ui.available_height() - 22.0).max(100.0);
                        egui::ScrollArea::vertical()
                            .id_source(format!(
                                "global_message_results_{}",
                                self.global_search_request_id
                            ))
                            .auto_shrink([false; 2])
                            .max_height(result_height)
                            .show_rows(
                                ui,
                                GLOBAL_SEARCH_RESULT_ROW_HEIGHT,
                                self.global_search_total_matches,
                                |ui, visible_rows| {
                                    let first_visible = visible_rows.start;
                                    let last_visible = visible_rows.end;
                                    for position in visible_rows {
                                        if let Some(result) =
                                            self.global_search_results.get(&position)
                                        {
                                            if render_global_result(ui, result, &self.search_query)
                                                .clicked()
                                            {
                                                open_global_result = Some(result.clone());
                                            }
                                        } else {
                                            render_global_result_placeholder(ui);
                                            global_search_page_request.get_or_insert(
                                                position / MESSAGE_SEARCH_PAGE_SIZE as usize
                                                    * MESSAGE_SEARCH_PAGE_SIZE as usize,
                                            );
                                        }
                                    }
                                    if last_visible > first_visible {
                                        let page_size = MESSAGE_SEARCH_PAGE_SIZE as usize;
                                        let current_page = (last_visible - 1) / page_size;
                                        let next_offset = (current_page + 1) * page_size;
                                        if last_visible.saturating_add(40) >= next_offset
                                            && next_offset < self.global_search_total_matches
                                            && !self
                                                .global_search_results
                                                .contains_key(&next_offset)
                                        {
                                            global_search_page_request.get_or_insert(next_offset);
                                        }
                                    }
                                },
                            );
                    }
                }
            });

        if let Some(offset) = global_search_page_request
            && !self.global_searching
            && self.global_search_last_submitted == self.search_query.trim()
        {
            self.trigger_global_message_search(self.search_query.clone(), offset, ctx.clone());
        }

        if let Some(group_idx) = open_group_chat {
            self.trigger_load_preferred_chat(group_idx, ctx.clone());
        }
        if let Some(result) = open_global_result {
            if let Some((group_idx, _)) = self.groups.iter().enumerate().find(|(_, group)| {
                group
                    .backups
                    .iter()
                    .any(|backup| backup.chat_id == result.chat_id)
            }) {
                self.select_group(group_idx);
            }
            self.pending_chat_highlight_query = Some(self.search_query.clone());
            self.trigger_load_chat_page(
                result.chat_id,
                result.chat_name,
                ChatPageRequest::Around {
                    message_id: result.message_id,
                },
                ctx.clone(),
            );
        }

        // Central Panel (Gantt and Detail View)
        let mut compare_pair = None;
        let mut open_chat_idx = None;
        let mut chat_view_action = None;
        let active_chat_clone = self.active_chat_view.clone();
        let chat_loading = self.loading_chat_view;
        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(16.0).fill(egui::Color32::from_rgb(20, 20, 25)))
            .show(ctx, |ui| {
                if let Ok(mut active_chat) = active_chat_clone.lock()
                    && let Some(chat) = active_chat.as_mut()
                {
                    chat_view_action = render_chat_view(ui, chat, chat_loading);
                    return;
                }
                if let Some(idx) = self.selected_group_idx {
                    let mut toggle_group_active = None;
                    let mut toggle_group_blacklisted = None;

                    egui::ScrollArea::vertical().id_source("central_scroll").show(ui, |ui| {
                        let group = &self.groups[idx];

                        ui.horizontal(|ui| {
                            let group_active = group.is_active();
                            let group_blacklisted = group.is_blacklisted();
                            let status_color = if group_blacklisted {
                                egui::Color32::from_rgb(72, 74, 82) // dark grey
                            } else if group_active {
                                egui::Color32::from_rgb(46, 204, 113) // green
                            } else if group.max_count == 0 {
                                egui::Color32::from_rgb(130, 135, 145) // discovered only
                            } else {
                                egui::Color32::from_rgb(231, 76, 60)  // red
                            };
                            let status_text = if group_blacklisted {
                                "Blacklisted · never backed up"
                            } else if group_active && group.max_count == 0 {
                                "Active · queued for first backup"
                            } else if group_active {
                                "Active"
                            } else if group.max_count == 0 {
                                "Discovered · not backed up"
                            } else {
                                "Inactive"
                            };
                            let mut label_text = egui::RichText::new(format!("Selected: {} ({})", group.name, status_text)).strong().color(status_color);
                            if group_blacklisted {
                                label_text = label_text.strikethrough();
                            }
                            if ui.selectable_label(false, label_text).on_hover_text(
                                if group_blacklisted {
                                    "Remove the blacklist rule before making this conversation active"
                                } else {
                                    "Click to toggle conversation Active status"
                                }
                            ).clicked() && !group_blacklisted {
                                toggle_group_active = Some((idx, !group_active));
                            }

                            let blacklist_label = if group_blacklisted {
                                "Remove from blacklist"
                            } else {
                                "🚫 Never back up"
                            };
                            if ui.button(blacklist_label).on_hover_text(
                                if group_blacklisted {
                                    "Remove the permanent exclusion; this does not activate the chat"
                                } else {
                                    "Deactivate this chat and exclude it even from --all"
                                }
                            ).clicked() {
                                toggle_group_blacklisted = Some((idx, !group_blacklisted));
                            }

                            let now = Utc::now().timestamp();
                            let latest_backup_unix = group
                                .backups
                                .iter()
                                .filter_map(|b| b.last_backup_run_unix.or(b.last_backup_unix))
                                .max();
                            let latest_msg_unix = group.backups.iter().filter_map(|b| b.max_unix).max();

                            let backup_ago = match latest_backup_unix {
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

                            let msg_ago = match latest_msg_unix {
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

                            ui.add_space(20.0);
                            ui.colored_label(egui::Color32::from_rgb(150, 150, 160), "Last Backup:");
                            ui.label(egui::RichText::new(&backup_ago).strong().color(egui::Color32::from_rgb(100, 180, 240)));
                            ui.label("  |  ");
                            ui.colored_label(egui::Color32::from_rgb(150, 150, 160), "Last Message:");
                            ui.label(egui::RichText::new(&msg_ago).strong().color(egui::Color32::from_rgb(100, 180, 240)));
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
                                    if ui
                                        .add_enabled(b.count > 0, egui::Button::new("💬 Open Chat").small())
                                        .on_hover_text(if b.count > 0 {
                                            "Open message history in a Telegram-styled window"
                                        } else {
                                            "This Telegram chat has not been backed up yet"
                                        })
                                        .clicked()
                                    {
                                        open_chat_idx = Some(b_idx);
                                    }
                                    let path_label = if b.count == 0 { "Planned path" } else { "Path" };
                                    ui.label(format!("| {}: {}", path_label, b.path));
                                });
                                let run_status = if b.last_backup_run_status.is_empty() {
                                    "unknown".to_string()
                                } else {
                                    b.last_backup_run_status.replace('_', " ")
                                };
                                let backup_attempt_ts = match b.last_backup_run_unix {
                                    Some(ts) => format_unix_to_ts(ts),
                                    None => "Unknown".to_string(),
                                };
                                ui.label(format!(
                                    "   Last Backup Run: {} [{}]",
                                    backup_attempt_ts, run_status
                                ));
                                let date_source = if b.last_backup_source.is_empty() {
                                    "unclassified".to_string()
                                } else {
                                    b.last_backup_source.replace('_', " ")
                                };
                                ui.label(format!(
                                    "   Last Content-Modifying Backup: {} [{}; {}]",
                                    backup_run_ts,
                                    b.last_backup_confidence,
                                    date_source
                                ))
                                .on_hover_text(if b.last_backup_evidence.is_empty() {
                                    "No backup-date evidence recorded"
                                } else {
                                    &b.last_backup_evidence
                                });
                                ui.label(format!("   Message IDs:     {} to {} (Total: {} messages)", b.min_id.unwrap_or(0), b.max_id.unwrap_or(0), b.count));
                                ui.label(format!("   Time span:       {} to {}", b.min_ts, b.max_ts));
                                if b.count == 0 {
                                    ui.horizontal(|ui| {
                                        ui.label("   Media Assets:    ");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(140, 150, 160),
                                            "None yet — activate this chat to include it in the next backup.",
                                        );
                                    });
                                } else if let Some(ref stats) = b.media_stats {
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

                        ui.add_space(15.0);
                        ui.heading("⚖️ Containment & Overlaps Analysis");
                        ui.separator();
                        ui.add_space(4.0);

                        for line in &self.comparison_results {
                            if line.trim().is_empty() {
                                ui.add_space(4.0);
                            } else {
                                ui.label(line);
                            }
                        }
                    });

                    if let Some((g_idx, blacklisted)) = toggle_group_blacklisted {
                        let chat_ids: Vec<String> = self.groups[g_idx]
                            .backups
                            .iter()
                            .map(|backup| backup.chat_id.clone())
                            .collect();
                        let active_db_path = self.active_db_path();
                        match rusqlite::Connection::open(&active_db_path) {
                            Ok(mut conn) => match set_chat_ids_blacklisted(
                                &mut conn,
                                &chat_ids,
                                blacklisted,
                            ) {
                                Ok(affected) if affected > 0 => {
                                    if let Some(group) = self.groups.get_mut(g_idx) {
                                        for backup in &mut group.backups {
                                            backup.is_blacklisted = blacklisted;
                                            if blacklisted {
                                                backup.is_active = false;
                                            }
                                        }
                                        self.status_msg = if blacklisted {
                                            format!(
                                                "Blacklisted {}. It is excluded from every backup mode.",
                                                group.name
                                            )
                                        } else {
                                            format!(
                                                "Removed {} from the blacklist. It remains inactive.",
                                                group.name
                                            )
                                        };
                                    }
                                }
                                Ok(_) => {
                                    self.status_msg = "No mapped Telegram target was found for this chat; refresh mappings before blacklisting it.".to_string();
                                }
                                Err(error) => {
                                    self.status_msg = format!("Failed to update blacklist: {}", error);
                                }
                            },
                            Err(error) => {
                                self.status_msg = format!("Failed to open database: {}", error);
                            }
                        }
                    }

                    if let Some((g_idx, active)) = toggle_group_active {
                        let chat_ids: Vec<String> = self.groups[g_idx]
                            .backups
                            .iter()
                            .map(|backup| backup.chat_id.clone())
                            .collect();
                        let result = (|| -> rusqlite::Result<()> {
                            let mut conn = rusqlite::Connection::open(self.active_db_path())?;
                            let tx = conn.transaction()?;
                            let val = if active { 1 } else { 0 };
                            for chat_id in &chat_ids {
                                tx.execute(
                                    "UPDATE chats SET is_active = ? WHERE chat_id = ?",
                                    rusqlite::params![val, chat_id],
                                )?;
                            }
                            tx.commit()
                        })();
                        match result {
                            Ok(()) => {
                                if let Some(g) = self.groups.get_mut(g_idx) {
                                    for b in &mut g.backups {
                                        b.is_active = active;
                                    }
                                }
                            }
                            Err(error) => {
                                self.status_msg = format!("Failed to update Active state: {error}");
                            }
                        }
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Select a conversation from the sidebar list to view timelines, Gantt spans, and overlaps analysis.");
                    });
                }
            });

        if let Some(action) = chat_view_action {
            match action {
                ChatViewerAction::Close => {
                    if let Ok(mut view) = self.active_chat_view.lock() {
                        *view = None;
                    }
                    self.chat_view_rx = None;
                    self.chat_search_rx = None;
                    self.loading_chat_view = false;
                }
                ChatViewerAction::ClearSearch => {
                    self.chat_search_request_id = self.chat_search_request_id.wrapping_add(1);
                    self.chat_search_rx = None;
                    if let Ok(mut view) = self.active_chat_view.lock()
                        && let Some(chat) = view.as_mut()
                    {
                        chat.searching = false;
                        chat.total_search_matches = 0;
                    }
                }
                ChatViewerAction::Search(query) => {
                    let chat_id = self
                        .active_chat_view
                        .lock()
                        .ok()
                        .and_then(|view| view.as_ref().map(|chat| chat.chat_id.clone()));
                    if let Some(chat_id) = chat_id {
                        self.trigger_chat_message_search(chat_id, query, 0, ctx.clone());
                    }
                }
                ChatViewerAction::LoadOlder => {
                    let request = self.active_chat_view.lock().ok().and_then(|view| {
                        let chat = view.as_ref()?;
                        let message = chat.messages.first()?;
                        Some((
                            chat.chat_id.clone(),
                            chat.backup_name.clone(),
                            ChatPageRequest::Before {
                                timestamp_unix: message.timestamp_unix,
                                message_id: message.message_id,
                            },
                        ))
                    });
                    if let Some((chat_id, name, request)) = request {
                        self.trigger_load_chat_page(chat_id, name, request, ctx.clone());
                    }
                }
                ChatViewerAction::LoadNewer => {
                    let request = self.active_chat_view.lock().ok().and_then(|view| {
                        let chat = view.as_ref()?;
                        let message = chat.messages.last()?;
                        Some((
                            chat.chat_id.clone(),
                            chat.backup_name.clone(),
                            ChatPageRequest::After {
                                timestamp_unix: message.timestamp_unix,
                                message_id: message.message_id,
                            },
                        ))
                    });
                    if let Some((chat_id, name, request)) = request {
                        self.trigger_load_chat_page(chat_id, name, request, ctx.clone());
                    }
                }
                ChatViewerAction::LoadLatest => {
                    let identity = self.active_chat_view.lock().ok().and_then(|view| {
                        view.as_ref()
                            .map(|chat| (chat.chat_id.clone(), chat.backup_name.clone()))
                    });
                    if let Some((chat_id, name)) = identity {
                        self.trigger_load_chat_page(
                            chat_id,
                            name,
                            ChatPageRequest::Latest,
                            ctx.clone(),
                        );
                    }
                }
                ChatViewerAction::JumpToSearchResult(index) => {
                    let target = self.active_chat_view.lock().ok().and_then(|view| {
                        let chat = view.as_ref()?;
                        let result = chat.search_results.get(index)?;
                        Some((
                            chat.chat_id.clone(),
                            chat.backup_name.clone(),
                            result.message_id,
                        ))
                    });
                    if let Some((chat_id, name, message_id)) = target {
                        self.trigger_load_chat_page(
                            chat_id,
                            name,
                            ChatPageRequest::Around { message_id },
                            ctx.clone(),
                        );
                    }
                }
            }
        }

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

        if self.loading_chat_view
            && self
                .active_chat_view
                .lock()
                .map(|view| view.is_none())
                .unwrap_or(true)
        {
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

        let active_comp_clone = self.active_comparison.clone();
        let is_comp_active = active_comp_clone.lock().unwrap().is_some();
        if is_comp_active {
            let title = {
                let lock = active_comp_clone.lock().unwrap();
                let comp = lock.as_ref().unwrap();
                format!(
                    "⚖️ Side-by-Side Comparison: Backup {} vs {}",
                    comp.backup_a_letter, comp.backup_b_letter
                )
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
                            .frame(
                                egui::Frame::none()
                                    .inner_margin(16.0)
                                    .fill(egui::Color32::from_rgb(20, 20, 25)),
                            )
                            .show(ctx, |ui| {
                                let mut comp_lock = active_comp_clone.lock().unwrap();
                                if let Some(ref mut comp) = *comp_lock {
                                    // Header controls
                                    ui.horizontal(|ui| {
                                        ui.label(format!(
                                            "Comparing overlapping messages. Total messages: {}.",
                                            comp.rows.len()
                                        ));
                                        ui.add_space(20.0);
                                        if !comp.discrepancies.is_empty() {
                                            let curr = comp.current_discrepancy_idx.unwrap_or(0);
                                            ui.label(format!(
                                                "⚠️ Discrepancy {} of {}",
                                                curr + 1,
                                                comp.discrepancies.len()
                                            ));

                                            if ui.button("⬅️ Previous Missing").clicked() {
                                                let prev_idx = if curr == 0 {
                                                    comp.discrepancies.len() - 1
                                                } else {
                                                    curr - 1
                                                };
                                                comp.current_discrepancy_idx = Some(prev_idx);
                                                comp.scroll_to_row_idx =
                                                    Some(comp.discrepancies[prev_idx]);
                                            }

                                            if ui.button("Next Missing ➡️").clicked() {
                                                let next_idx =
                                                    (curr + 1) % comp.discrepancies.len();
                                                comp.current_discrepancy_idx = Some(next_idx);
                                                comp.scroll_to_row_idx =
                                                    Some(comp.discrepancies[next_idx]);
                                            }
                                        } else {
                                            ui.colored_label(
                                                egui::Color32::from_rgb(46, 204, 113),
                                                "✅ Perfect alignment! No discrepancies found.",
                                            );
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
                                            let mut scroll_area = egui::ScrollArea::vertical()
                                                .id_source("compare_scroll_area");
                                            if let Some(target_idx) = comp.scroll_to_row_idx {
                                                let spacing_y = ui.spacing().item_spacing.y;
                                                let target_y = (target_idx as f32
                                                    * (row_height + spacing_y)
                                                    - 200.0)
                                                    .max(0.0);
                                                scroll_area = scroll_area
                                                    .scroll_offset(egui::vec2(0.0, target_y));
                                                comp.scroll_to_row_idx = None; // Reset it!
                                            }

                                            scroll_area.show_rows(
                                                ui,
                                                row_height,
                                                num_rows,
                                                |ui, row_range| {
                                                    for idx in row_range {
                                                        let row = &comp.rows[idx];
                                                        ui.columns(2, |cols| {
                                                            // Column A
                                                            cols[0].vertical(|ui| {
                                                                if let Some(ref msg) = row.msg_a {
                                                                    render_message_bubble(
                                                                        ui,
                                                                        msg,
                                                                        row.is_discrepancy,
                                                                        true,
                                                                        false,
                                                                        None,
                                                                    );
                                                                } else {
                                                                    render_missing_placeholder(
                                                                        ui,
                                                                        "Missing in Backup A",
                                                                    );
                                                                }
                                                            });

                                                            // Column B
                                                            cols[1].vertical(|ui| {
                                                                if let Some(ref msg) = row.msg_b {
                                                                    render_message_bubble(
                                                                        ui,
                                                                        msg,
                                                                        row.is_discrepancy,
                                                                        false,
                                                                        false,
                                                                        None,
                                                                    );
                                                                } else {
                                                                    render_missing_placeholder(
                                                                        ui,
                                                                        "Missing in Backup B",
                                                                    );
                                                                }
                                                            });
                                                        });
                                                        ui.add_space(6.0);
                                                    }
                                                },
                                            );
                                        });

                                    if ctx.input(|i| i.viewport().close_requested()) {
                                        *comp_lock = None;
                                    }
                                }
                            });
                    }
                },
            );
        }
    }
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
    fn same_title_distinct_peers_remain_separate_and_are_disambiguated() {
        let root = std::env::temp_dir().join(format!(
            "tgbackman-same-title-peer-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let db = root.join("backup.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
             "CREATE TABLE chats (
                 chat_id TEXT PRIMARY KEY,
                 chat_name TEXT,
                 chat_type TEXT,
                 backup_path TEXT,
                 is_active INTEGER DEFAULT 0
             );
             CREATE TABLE messages (
                 message_id INTEGER NOT NULL,
                 chat_id TEXT NOT NULL,
                 timestamp TEXT,
                 timestamp_unix INTEGER,
                 text TEXT
             );
             CREATE TABLE telegram_backup_targets (
                 target_key TEXT PRIMARY KEY,
                 chat_id TEXT NOT NULL UNIQUE,
                 peer_kind TEXT NOT NULL,
                 peer_id INTEGER NOT NULL,
                 title TEXT NOT NULL,
                 enabled INTEGER NOT NULL,
                 output_dir TEXT
             );
             CREATE TABLE telegram_backup_target_chats (
                 target_key TEXT NOT NULL,
                 chat_id TEXT NOT NULL,
                 match_method TEXT NOT NULL,
                 linked_unix INTEGER NOT NULL,
                 PRIMARY KEY(target_key, chat_id),
                 UNIQUE(chat_id)
             );
             INSERT INTO chats(chat_id, chat_name, backup_path) VALUES
                 ('channel_1001', 'Example Chat', '/backup/example-chat'),
                 ('dialog_1002', 'Example Chat', '/backup/example-chat');
             INSERT INTO messages(message_id, chat_id, timestamp, timestamp_unix, text) VALUES
                 (1, 'channel_1001', '1970-01-01T00:01:40Z', 100, 'same message one'),
                 (2, 'channel_1001', '1970-01-01T00:01:41Z', 101, 'same message two'),
                 (3, 'channel_1001', '1970-01-01T00:01:42Z', 102, 'same message three'),
                 (1, 'dialog_1002', '1970-01-01T00:01:40Z', 100, 'same message one'),
                 (2, 'dialog_1002', '1970-01-01T00:01:41Z', 101, 'same message two'),
                 (3, 'dialog_1002', '1970-01-01T00:01:42Z', 102, 'same message three');
             INSERT INTO telegram_backup_targets
                 (target_key, chat_id, peer_kind, peer_id, title, enabled, output_dir) VALUES
                 ('example-channel', 'channel_1001', 'channel', 1001, 'Example Chat', 1, '/backup/example-chat'),
                 ('example-user', 'dialog_1002', 'user', 1002, 'Example Chat', 1, '/backup/example-chat');
             INSERT INTO telegram_backup_target_chats
                 (target_key, chat_id, match_method, linked_unix) VALUES
                 ('example-channel', 'channel_1001', 'telegram-discovered', 1),
                 ('example-user', 'dialog_1002', 'telegram-discovered', 1);",
        )
        .unwrap();

        let groups = run_inventory(&conn, db.to_str().unwrap()).unwrap();
        assert_eq!(groups.len(), 2);
        let mut names: Vec<String> = groups.iter().map(|group| group.name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "Example Chat [channel:1001]".to_string(),
                "Example Chat [user:1002]".to_string()
            ]
        );

        drop(conn);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discovered_target_is_visible_as_zero_message_chat() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tgbackman-discovered-chat-test-{}-{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let db = root.join("backup.db");
        let mut conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE chats (
                 chat_id TEXT PRIMARY KEY,
                 chat_name TEXT,
                 chat_type TEXT,
                 backup_path TEXT,
                 is_active INTEGER DEFAULT 0,
                 last_backup_unix INTEGER
             );
             CREATE TABLE messages (
                 message_id INTEGER NOT NULL,
                 chat_id TEXT NOT NULL,
                 timestamp TEXT,
                 timestamp_unix INTEGER,
                 text TEXT
             );
             CREATE TABLE telegram_backup_targets (
                 target_key TEXT PRIMARY KEY,
                 chat_id TEXT NOT NULL UNIQUE,
                 peer_kind TEXT NOT NULL,
                 peer_id INTEGER NOT NULL,
                 title TEXT NOT NULL,
                 enabled INTEGER NOT NULL,
                 output_dir TEXT
             );
             CREATE TABLE telegram_backup_target_chats (
                 target_key TEXT NOT NULL,
                 chat_id TEXT NOT NULL,
                 match_method TEXT NOT NULL,
                 linked_unix INTEGER NOT NULL,
                 PRIMARY KEY(target_key, chat_id),
                 UNIQUE(chat_id)
             );
             INSERT INTO telegram_backup_targets
                 (target_key, chat_id, peer_kind, peer_id, title, enabled, output_dir)
             VALUES
                 ('new-chat-key', 'dialog_42', 'user', 42, 'New Telegram Chat', 1,
                  '/backup/New Telegram Chat'),
                 ('disabled-key', 'group_99', 'group', 99, 'Migrated predecessor', 0,
                  '/backup/Migrated predecessor');",
        )
        .unwrap();

        let groups = run_inventory(&conn, db.to_str().unwrap()).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].name, "New Telegram Chat");
        assert_eq!(groups[0].max_count, 0);
        assert!(!groups[0].is_active());
        assert!(!groups[0].is_blacklisted());
        assert_eq!(groups[0].backups[0].chat_id, "dialog_42");
        assert_eq!(groups[0].backups[0].path, "/backup/New Telegram Chat");
        assert_eq!(
            conn.query_row(
                "SELECT match_method FROM telegram_backup_target_chats WHERE target_key='new-chat-key'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
            "telegram-discovered"
        );
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM chats WHERE chat_id='group_99'",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
            0
        );

        assert_eq!(
            set_chat_ids_blacklisted(&mut conn, &["group_99".to_string()], true).unwrap(),
            1
        );
        let groups = run_inventory(&conn, db.to_str().unwrap()).unwrap();
        let blocked = groups
            .iter()
            .find(|group| group.name == "Migrated predecessor")
            .unwrap();
        assert_eq!(blocked.max_count, 0);
        assert!(blocked.is_blacklisted());
        assert!(!blocked.is_active());

        drop(conn);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn viewer_blacklist_deactivates_aliases_and_can_be_removed() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE chats (
                 chat_id TEXT PRIMARY KEY,
                 chat_name TEXT,
                 is_active INTEGER DEFAULT 0
             );
             CREATE TABLE telegram_backup_targets (
                 target_key TEXT PRIMARY KEY,
                 chat_id TEXT NOT NULL UNIQUE,
                 peer_kind TEXT NOT NULL,
                 peer_id INTEGER NOT NULL,
                 title TEXT NOT NULL
             );
             CREATE TABLE telegram_backup_target_chats (
                 target_key TEXT NOT NULL,
                 chat_id TEXT NOT NULL,
                 match_method TEXT NOT NULL,
                 linked_unix INTEGER NOT NULL,
                 PRIMARY KEY(target_key, chat_id),
                 UNIQUE(chat_id)
             );
             INSERT INTO chats(chat_id, chat_name, is_active)
                 VALUES ('current', 'Current', 1), ('historical', 'Old name', 1);
             INSERT INTO telegram_backup_targets
                 (target_key, chat_id, peer_kind, peer_id, title)
                 VALUES ('stable-key', 'current', 'channel', 123, 'Current');
             INSERT INTO telegram_backup_target_chats
                 (target_key, chat_id, match_method, linked_unix)
                 VALUES ('stable-key', 'current', 'canonical', 1),
                        ('stable-key', 'historical', 'telegram-migrated-from', 1);",
        )
        .unwrap();

        let chat_ids = vec!["current".to_string(), "historical".to_string()];
        assert_eq!(
            set_chat_ids_blacklisted(&mut conn, &chat_ids, true).unwrap(),
            1
        );
        assert_eq!(
            blacklisted_chat_ids(&conn).unwrap(),
            HashSet::from(["current".to_string(), "historical".to_string()])
        );
        assert_eq!(
            conn.query_row("SELECT SUM(is_active) FROM chats", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );

        assert_eq!(
            set_chat_ids_blacklisted(&mut conn, &chat_ids, false).unwrap(),
            1
        );
        assert!(blacklisted_chat_ids(&conn).unwrap().is_empty());
        assert_eq!(
            conn.query_row("SELECT SUM(is_active) FROM chats", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn inventory_preserves_repaired_backup_date_and_evidence() {
        let root = std::env::temp_dir().join(format!(
            "tgbackman-date-evidence-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let db = root.join("backup.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE chats (
                 chat_id TEXT PRIMARY KEY,
                 chat_name TEXT,
                 backup_path TEXT,
                 is_active INTEGER DEFAULT 0,
                 last_backup_unix INTEGER,
                 last_backup_source TEXT,
                 last_backup_confidence TEXT,
                 last_backup_evidence TEXT
             );
             CREATE TABLE messages (
                 message_id INTEGER NOT NULL,
                 chat_id TEXT NOT NULL,
                 timestamp TEXT,
                 timestamp_unix INTEGER,
                 text TEXT
             );
             INSERT INTO chats(
                 chat_id, chat_name, backup_path, last_backup_unix,
                 last_backup_source, last_backup_confidence, last_backup_evidence
             ) VALUES (
                 'legacy', 'Legacy', '/rewritten/path', 300,
                 'converted_desktop_export_asset_batch', 'high', 'preserved assets'
             );
             INSERT INTO messages(message_id, chat_id, timestamp, timestamp_unix, text)
                 VALUES (1, 'legacy', '1970-01-01T00:03:20Z', 200, 'message');",
        )
        .unwrap();

        let groups = run_inventory(&conn, db.to_str().unwrap()).unwrap();
        let backup = &groups[0].backups[0];
        assert_eq!(backup.last_backup_unix, Some(300));
        assert_eq!(
            backup.last_backup_source,
            "converted_desktop_export_asset_batch"
        );
        assert_eq!(backup.last_backup_confidence, "high");
        assert_eq!(backup.last_backup_evidence, "preserved assets");

        drop(conn);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn migrated_predecessors_are_hidden_or_grouped_by_message_presence() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE chats (chat_id TEXT PRIMARY KEY);
             CREATE TABLE messages (
                 message_id INTEGER NOT NULL,
                 chat_id TEXT NOT NULL
             );
             CREATE TABLE telegram_backup_targets (
                 target_key TEXT PRIMARY KEY,
                 title TEXT NOT NULL,
                 enabled INTEGER NOT NULL
             );
             CREATE TABLE telegram_backup_target_chats (
                 target_key TEXT NOT NULL,
                 chat_id TEXT NOT NULL,
                 match_method TEXT NOT NULL
             );
             INSERT INTO chats(chat_id) VALUES
                 ('channel_2001'),
                 ('group_2002'),
                 ('historical_group');
             INSERT INTO messages(message_id, chat_id)
                 VALUES (1, 'historical_group');
             INSERT INTO telegram_backup_targets(target_key, title, enabled)
                 VALUES ('example-migration-target', 'Example Group', 1);
             INSERT INTO telegram_backup_target_chats
                 (target_key, chat_id, match_method)
             VALUES
                 ('example-migration-target', 'channel_2001', 'telegram-discovered'),
                 ('example-migration-target', 'group_2002', 'telegram-migrated-from'),
                 ('example-migration-target', 'historical_group', 'telegram-migrated-from');",
        )
        .unwrap();

        let hidden = zero_message_migrated_predecessors(&conn).unwrap();
        assert_eq!(hidden, HashSet::from(["group_2002".to_string()]));

        let mut uf = UnionFind::new();
        let preferred_names = apply_authoritative_target_links(&conn, &mut uf).unwrap();
        let current_root = uf.find("channel_2001");
        assert_eq!(uf.find("group_2002"), current_root);
        assert_eq!(uf.find("historical_group"), current_root);
        assert_eq!(preferred_names.get(&current_root).unwrap(), "Example Group");
    }

    #[test]
    fn cache_freshness_rejects_newer_database_and_wal() {
        let root =
            std::env::temp_dir().join(format!("tgbackman-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let db = root.join("backup.db");
        let cache = root.join("backup_clusters.json");
        std::fs::write(&cache, b"{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        std::fs::write(&db, b"sqlite").unwrap();
        assert!(!cache_is_fresh(
            cache.to_str().unwrap(),
            db.to_str().unwrap()
        ));

        std::thread::sleep(std::time::Duration::from_millis(2));
        std::fs::write(&cache, b"{}").unwrap();
        assert!(cache_is_fresh(
            cache.to_str().unwrap(),
            db.to_str().unwrap()
        ));

        std::thread::sleep(std::time::Duration::from_millis(2));
        std::fs::write(format!("{}-wal", db.display()), b"uncheckpointed").unwrap();
        assert!(!cache_is_fresh(
            cache.to_str().unwrap(),
            db.to_str().unwrap()
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "requires a live database via TGBACKMAN_DB"]
    fn test_run_inventory_performance() {
        let db_path = std::env::var("TGBACKMAN_DB").unwrap_or_else(|_| {
            let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
            let volume = std::env::var("TGBACKMAN_REMOVABLE_VOLUME")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "backup-volume".to_string());
            format!("/media/{}/{}/sqlitedb/telegram_backup.db", user, volume)
        });
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let start = std::time::Instant::now();
        let result = run_inventory(&conn, &db_path);
        let duration = start.elapsed();
        println!("run_inventory took: {:?}", duration);
        assert!(result.is_ok());
        let groups = result.unwrap();
        println!("Loaded {} groups", groups.len());
    }

    #[test]
    #[ignore = "requires a mounted legacy media archive"]
    fn test_compute_media_stats_split_and_unofficial() {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let volume = std::env::var("TGBACKMAN_REMOVABLE_VOLUME")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "backup-volume".to_string());
        let db_path = format!("/media/{}/{}/sqlitedb/telegram_backup.db", user, volume);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let groups = run_inventory(&conn, &db_path).unwrap();

        // Find split chat
        let mut found_split = false;
        for g in &groups {
            for b in &g.backups {
                if b.chat_id == "2015-02-27T19-06-48Z__2015-06-14T00-52-25Z" {
                    let stats = b.compute_media_stats(&db_path);
                    println!(
                        "Split stats: photos_resolved={}/{}, videos_resolved={}/{}, voice_resolved={}/{}, files_resolved={}/{}",
                        stats.photos_resolved,
                        stats.photos_count,
                        stats.videos_resolved,
                        stats.videos_count,
                        stats.voice_resolved,
                        stats.voice_count,
                        stats.files_resolved,
                        stats.files_count
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
                if b.chat_id == "group_3001" {
                    let stats = b.compute_media_stats(&db_path);
                    println!(
                        "Unofficial stats: photos_resolved={}/{}, videos_resolved={}/{}, voice_resolved={}/{}, files_resolved={}/{}",
                        stats.photos_resolved,
                        stats.photos_count,
                        stats.videos_resolved,
                        stats.videos_count,
                        stats.voice_resolved,
                        stats.voice_count,
                        stats.files_resolved,
                        stats.files_count
                    );
                    assert!(stats.photos_resolved > 0);
                    assert!(stats.videos_resolved > 0);
                    assert!(stats.voice_resolved > 0);
                    assert!(stats.files_resolved > 0);
                    found_unofficial = true;
                }
            }
        }
        assert!(
            found_unofficial,
            "Should have found the unofficial backup chat"
        );
    }
}
