//! Database inventory mutations and target identity helpers.

use chrono::Utc;
use rusqlite::OptionalExtension;
use std::collections::{HashMap, HashSet};

use crate::cache::{get_clusters_cache_path, secure_cache_file};
use crate::inventory::UnionFind;
use crate::matching::clean_text_for_match;
use crate::model::{
    BackupInfo, BackupMessage, ChatGroup, ChatMessagePage, ChatPageRequest, MessageSearchResult,
    ReplyPreview,
};

pub(crate) const CHAT_VIEW_PAGE_SIZE: i64 = 400;
pub(crate) const MESSAGE_SEARCH_PAGE_SIZE: i64 = 250;
const CHAT_MESSAGE_COLUMNS: &str = "message_id, chat_id, sender, sender_id, timestamp_unix, timestamp, text, media_type, \
     media_path, reply_to_id, reply_to_chat_id, reply_to_peer_kind, reply_to_peer_id, \
     reply_to_top_id, reply_to_story_id, reply_quote_text, reply_media_json, forwarded_from, \
     edit_timestamp, reactions_json, message_type, action_json";

pub(crate) struct ChatRow {
    pub(crate) chat_id: String,
    pub(crate) name: String,
    pub(crate) path: Option<String>,
    pub(crate) is_active: bool,
    pub(crate) is_blacklisted: bool,
    pub(crate) last_backup_unix: Option<i64>,
    pub(crate) last_backup_run_unix: Option<i64>,
    pub(crate) last_backup_run_status: String,
    pub(crate) last_backup_source: String,
    pub(crate) last_backup_confidence: String,
    pub(crate) last_backup_evidence: String,
    pub(crate) min_msg_id: Option<i64>,
    pub(crate) max_msg_id: Option<i64>,
    pub(crate) msg_count: Option<i64>,
    pub(crate) min_timestamp: Option<String>,
    pub(crate) max_timestamp: Option<String>,
    pub(crate) min_timestamp_unix: Option<i64>,
    pub(crate) max_timestamp_unix: Option<i64>,
}

pub(crate) fn table_exists(
    conn: &rusqlite::Connection,
    table: &str,
) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?)",
        rusqlite::params![table],
        |row| row.get(0),
    )
}

struct ResolvedReplyParent {
    message_id: i64,
    sender: Option<String>,
    text: String,
    media_type: Option<String>,
    chat_id: String,
    chat_name: Option<String>,
}

fn resolve_cross_reply_parent(
    conn: &rusqlite::Connection,
    peer_kind: &str,
    peer_id: i64,
    message_id: i64,
) -> Result<Option<ResolvedReplyParent>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")?
        || !table_exists(conn, "telegram_backup_target_chats")?
    {
        return Ok(None);
    }
    let mut stmt = conn.prepare(
        "WITH candidate_chats(chat_id, priority) AS (
             SELECT chat_id, 0 FROM telegram_backup_targets
              WHERE peer_kind=? AND peer_id=?
             UNION
             SELECT links.chat_id, 1
               FROM telegram_backup_targets AS targets
               JOIN telegram_backup_target_chats AS links
                 ON links.target_key=targets.target_key
              WHERE targets.peer_kind=? AND targets.peer_id=?
         )
         SELECT parent.message_id, parent.sender, parent.text, parent.media_type,
                parent.chat_id, chats.chat_name
           FROM candidate_chats
           JOIN messages AS parent
             ON parent.chat_id=candidate_chats.chat_id AND parent.message_id=?
           LEFT JOIN chats ON chats.chat_id=parent.chat_id
          WHERE COALESCE(parent.is_deleted, 0)=0
          ORDER BY candidate_chats.priority
          LIMIT 1",
    )?;
    let mut rows = stmt.query(rusqlite::params![
        peer_kind, peer_id, peer_kind, peer_id, message_id
    ])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    Ok(Some(ResolvedReplyParent {
        message_id: row.get(0)?,
        sender: row.get(1)?,
        text: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
        media_type: row.get(3)?,
        chat_id: row.get(4)?,
        chat_name: row.get(5)?,
    }))
}

const IDENTITY_SAMPLE_SIZE: i64 = 400;

#[derive(Clone)]
struct IdentityRecord {
    chat_id: String,
    message_id: i64,
    sender_alias: String,
    sender_id: Option<String>,
    timestamp_unix: Option<i64>,
    text: String,
}

fn normalized_sender_alias(sender: &str) -> String {
    sender
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase()
}

fn identity_record(row: &rusqlite::Row<'_>) -> Result<IdentityRecord, rusqlite::Error> {
    let sender = row.get::<_, Option<String>>(2)?.unwrap_or_default();
    Ok(IdentityRecord {
        chat_id: row.get(0)?,
        message_id: row.get(1)?,
        sender_alias: normalized_sender_alias(&sender),
        sender_id: row.get(3)?,
        timestamp_unix: row.get(4)?,
        text: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
    })
}

fn linked_identity_chats(
    conn: &rusqlite::Connection,
    chat_id: &str,
) -> Result<Vec<String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_target_chats")? {
        return Ok(vec![chat_id.to_string()]);
    }
    let mut stmt = conn.prepare(
        "SELECT DISTINCT sibling.chat_id
           FROM telegram_backup_target_chats AS selected
           JOIN telegram_backup_target_chats AS sibling
             ON sibling.target_key=selected.target_key
          WHERE selected.chat_id=?
          ORDER BY sibling.chat_id",
    )?;
    let mut chats = stmt
        .query_map(rusqlite::params![chat_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    if !chats.iter().any(|candidate| candidate == chat_id) {
        chats.push(chat_id.to_string());
    }
    chats.sort_unstable();
    chats.dedup();
    Ok(chats)
}

fn resolve_self_sender_aliases(
    conn: &rusqlite::Connection,
    chat_id: &str,
) -> Result<HashSet<String>, rusqlite::Error> {
    let mut self_id_stmt = conn.prepare(
        "SELECT DISTINCT sender_id
           FROM messages
          WHERE source_format='telegram_api' AND sender='Me'
            AND sender_id IS NOT NULL AND trim(sender_id)!=''",
    )?;
    let self_sender_ids = self_id_stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<HashSet<_>, _>>()?;
    let linked_chats = linked_identity_chats(conn, chat_id)?;

    let mut sample_stmt = conn.prepare(
        "SELECT chat_id, message_id, sender, sender_id, timestamp_unix, text
           FROM messages
          WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
          ORDER BY timestamp_unix DESC, message_id DESC
          LIMIT ?",
    )?;
    let mut samples: HashMap<String, Vec<IdentityRecord>> = HashMap::new();
    for linked_chat in &linked_chats {
        let rows = sample_stmt.query_map(
            rusqlite::params![linked_chat, IDENTITY_SAMPLE_SIZE],
            identity_record,
        )?;
        samples.insert(linked_chat.clone(), rows.collect::<Result<_, _>>()?);
    }

    type AliasNode = (String, String);
    type AliasEdge = (AliasNode, AliasNode);
    let mut known_self = HashSet::<AliasNode>::new();
    for records in samples.values() {
        for record in records {
            if matches!(
                record.sender_alias.as_str(),
                "me" | "self" | "you" | "outgoing"
            ) || record
                .sender_id
                .as_ref()
                .is_some_and(|sender_id| self_sender_ids.contains(sender_id))
            {
                known_self.insert((record.chat_id.clone(), record.sender_alias.clone()));
            }
        }
    }

    let mut timestamp_candidates = conn.prepare(
        "SELECT chat_id, message_id, sender, sender_id, timestamp_unix, text
           FROM messages
          WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
            AND timestamp_unix BETWEEN ? AND ?",
    )?;
    let mut id_candidates = conn.prepare(
        "SELECT chat_id, message_id, sender, sender_id, timestamp_unix, text
           FROM messages
          WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
            AND message_id BETWEEN ? AND ?",
    )?;
    let mut evidence: HashMap<AliasEdge, HashSet<i64>> = HashMap::new();
    for (source_chat, source_records) in &samples {
        if source_records.is_empty() {
            continue;
        }
        let timestamp_bounds = source_records
            .iter()
            .filter_map(|record| record.timestamp_unix)
            .fold(None::<(i64, i64)>, |bounds, value| match bounds {
                Some((min, max)) => Some((min.min(value), max.max(value))),
                None => Some((value, value)),
            });
        let id_bounds =
            source_records
                .iter()
                .fold(None::<(i64, i64)>, |bounds, record| match bounds {
                    Some((min, max)) => {
                        Some((min.min(record.message_id), max.max(record.message_id)))
                    }
                    None => Some((record.message_id, record.message_id)),
                });

        for sibling_chat in linked_chats
            .iter()
            .filter(|candidate| *candidate != source_chat)
        {
            let mut candidates = HashMap::<i64, IdentityRecord>::new();
            if let Some((min, max)) = timestamp_bounds {
                let rows = timestamp_candidates
                    .query_map(rusqlite::params![sibling_chat, min, max], identity_record)?;
                for record in rows {
                    let record = record?;
                    candidates.insert(record.message_id, record);
                }
            }
            if let Some((min, max)) = id_bounds {
                let rows = id_candidates
                    .query_map(rusqlite::params![sibling_chat, min, max], identity_record)?;
                for record in rows {
                    let record = record?;
                    candidates.insert(record.message_id, record);
                }
            }

            for source in source_records {
                if source.sender_alias.is_empty() {
                    continue;
                }
                for candidate in candidates.values() {
                    if candidate.sender_alias.is_empty() {
                        continue;
                    }
                    let same_timestamp_and_text = source.timestamp_unix.is_some()
                        && source.timestamp_unix == candidate.timestamp_unix
                        && !source.text.is_empty()
                        && source.text == candidate.text;
                    let same_id_and_content = source.message_id == candidate.message_id
                        && (source.timestamp_unix == candidate.timestamp_unix
                            || (!source.text.is_empty() && source.text == candidate.text));
                    if same_timestamp_and_text || same_id_and_content {
                        evidence
                            .entry((
                                (source.chat_id.clone(), source.sender_alias.clone()),
                                (candidate.chat_id.clone(), candidate.sender_alias.clone()),
                            ))
                            .or_default()
                            .insert(source.message_id);
                    }
                }
            }
        }
    }

    loop {
        let mut changed = false;
        for ((from, to), matched_messages) in &evidence {
            if matched_messages.len() < 2 {
                continue;
            }
            if known_self.contains(from) {
                changed |= known_self.insert(to.clone());
            }
            if known_self.contains(to) {
                changed |= known_self.insert(from.clone());
            }
        }
        if !changed {
            break;
        }
    }

    Ok(known_self
        .into_iter()
        .filter_map(|(identity_chat, alias)| (identity_chat == chat_id).then_some(alias))
        .collect())
}

fn query_chat_messages<P: rusqlite::Params>(
    conn: &rusqlite::Connection,
    selection_sql: &str,
    params: P,
    self_sender_aliases: &HashSet<String>,
) -> Result<Vec<BackupMessage>, rusqlite::Error> {
    let has_target_links = table_exists(conn, "telegram_backup_target_chats")?;
    let linked_identity_expression = if has_target_links {
        "OR EXISTS (
             SELECT 1
               FROM telegram_backup_target_chats AS selected_link
               JOIN telegram_backup_target_chats AS sibling_link
                 ON sibling_link.target_key=selected_link.target_key
               JOIN messages AS sibling
                 ON sibling.chat_id=sibling_link.chat_id
                AND sibling.message_id=selected.message_id
              WHERE selected_link.chat_id=selected.chat_id
                AND sibling.sender_id IN (SELECT sender_id FROM self_sender_ids)
                AND COALESCE(sibling.is_deleted, 0)=0
                AND (
                    (sibling.timestamp_unix IS NOT NULL
                     AND selected.timestamp_unix IS NOT NULL
                     AND sibling.timestamp_unix=selected.timestamp_unix)
                    OR COALESCE(sibling.text, '')=COALESCE(selected.text, '')
                )
         )"
    } else {
        ""
    };
    let sql = format!(
        "WITH selected AS ({selection_sql}),
         self_sender_ids(sender_id) AS (
             SELECT DISTINCT sender_id
              FROM messages
              WHERE source_format='telegram_api'
                AND sender='Me'
                AND sender_id IS NOT NULL AND trim(sender_id)!=''
         )
         SELECT selected.message_id, selected.sender, selected.timestamp_unix,
                selected.timestamp, selected.text, selected.media_type, selected.media_path,
                selected.reply_to_id, selected.reply_to_chat_id,
                selected.reply_to_peer_kind, selected.reply_to_peer_id,
                selected.reply_to_top_id, selected.reply_to_story_id,
                selected.reply_quote_text, selected.reply_media_json,
                parent.message_id, parent.sender, parent.text, parent.media_type,
                parent.chat_id,
                CASE WHEN parent.chat_id != selected.chat_id THEN parent_chat.chat_name END,
                selected.forwarded_from, selected.edit_timestamp, selected.reactions_json,
                selected.message_type, selected.action_json,
                CASE
                    WHEN lower(trim(COALESCE(selected.sender, '')))
                         IN ('me', 'self', 'you', 'outgoing') THEN 1
                    WHEN selected.sender_id IN (SELECT sender_id FROM self_sender_ids)
                    {linked_identity_expression} THEN 1
                    ELSE 0
                END
           FROM selected
           LEFT JOIN messages AS parent
             ON parent.message_id=selected.reply_to_id
            AND parent.chat_id=COALESCE(
                    selected.reply_to_chat_id,
                    CASE WHEN selected.reply_to_peer_kind IS NULL THEN selected.chat_id END
                )
            AND COALESCE(parent.is_deleted, 0)=0
           LEFT JOIN chats AS parent_chat ON parent_chat.chat_id=parent.chat_id
          ORDER BY selected.timestamp_unix ASC, selected.message_id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mapped = stmt.query_map(params, |row| {
        let text = row.get::<_, Option<String>>(4)?.unwrap_or_default();
        let reply_message_id: Option<i64> = row.get(7)?;
        let reply_story_id: Option<i64> = row.get(12)?;
        let reply_topic_id: Option<i64> = row.get(11)?;
        let quote_text: Option<String> = row.get(13)?;
        let reply_media: Option<String> = row.get(14)?;
        let parent_message_id: Option<i64> = row.get(15)?;
        let parent_text = row.get::<_, Option<String>>(17)?.unwrap_or_default();
        let reply = if reply_message_id.is_some()
            || reply_story_id.is_some()
            || reply_topic_id.is_some()
            || quote_text.is_some()
        {
            let preview_text = quote_text
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| {
                    if !parent_text.is_empty() {
                        parent_text
                    } else if reply_media.is_some() {
                        "[media attachment]".to_string()
                    } else {
                        String::new()
                    }
                });
            Some(ReplyPreview {
                message_id: reply_message_id,
                story_id: reply_story_id,
                topic_id: reply_topic_id,
                peer_kind: row.get(9)?,
                peer_id: row.get(10)?,
                target_chat_id: row.get::<_, Option<String>>(19)?.or(row.get(8)?),
                chat_name: row.get(20)?,
                sender: row.get(16)?,
                text: preview_text,
                media_type: row.get(18)?,
                missing: parent_message_id.is_none(),
            })
        } else {
            None
        };
        Ok(BackupMessage {
            message_id: row.get(0)?,
            sender: row
                .get::<_, Option<String>>(1)?
                .unwrap_or_else(|| "Unknown".to_string()),
            timestamp_unix: row.get::<_, Option<i64>>(2)?.unwrap_or_default(),
            timestamp_str: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            clean_text: clean_text_for_match(&text),
            text,
            media_type: row.get(5)?,
            media_path: row.get(6)?,
            reply,
            forwarded_from: row.get(21)?,
            edit_timestamp: row.get(22)?,
            reactions_json: row.get(23)?,
            message_type: row.get(24)?,
            action_json: row.get(25)?,
            is_outgoing: row.get(26)?,
        })
    })?;
    let mut messages: Vec<BackupMessage> = mapped.collect::<Result<_, _>>()?;
    for message in &mut messages {
        message.is_outgoing |=
            self_sender_aliases.contains(&normalized_sender_alias(&message.sender));
    }
    resolve_reply_parents(conn, &mut messages)?;
    Ok(messages)
}

fn resolve_reply_parents(
    conn: &rusqlite::Connection,
    messages: &mut [BackupMessage],
) -> Result<(), rusqlite::Error> {
    let mut resolved = HashMap::new();
    for message in messages {
        let Some(reply) = message.reply.as_mut() else {
            continue;
        };
        let (Some(message_id), Some(peer_kind), Some(peer_id)) =
            (reply.message_id, reply.peer_kind.as_deref(), reply.peer_id)
        else {
            continue;
        };
        if !reply.missing || reply.target_chat_id.is_some() {
            continue;
        }
        let key = (peer_kind.to_string(), peer_id, message_id);
        let parent = if let Some(cached) = resolved.get(&key) {
            cached
        } else {
            let found = resolve_cross_reply_parent(conn, peer_kind, peer_id, message_id)?;
            resolved.insert(key.clone(), found);
            resolved.get(&key).expect("just inserted reply lookup")
        };
        if let Some(parent) = parent {
            if reply.text.is_empty() {
                reply.text.clone_from(&parent.text);
            }
            reply.sender.clone_from(&parent.sender);
            if reply.media_type.is_none() {
                reply.media_type.clone_from(&parent.media_type);
            }
            reply.target_chat_id = Some(parent.chat_id.clone());
            reply.chat_name.clone_from(&parent.chat_name);
            reply.message_id = Some(parent.message_id);
            reply.missing = false;
        }
    }
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn load_chat_messages(
    conn: &rusqlite::Connection,
    chat_id: &str,
    limit: i64,
) -> Result<Vec<BackupMessage>, rusqlite::Error> {
    let self_sender_aliases = resolve_self_sender_aliases(conn, chat_id)?;
    load_latest_chat_messages(conn, chat_id, limit, &self_sender_aliases)
}

fn load_latest_chat_messages(
    conn: &rusqlite::Connection,
    chat_id: &str,
    limit: i64,
    self_sender_aliases: &HashSet<String>,
) -> Result<Vec<BackupMessage>, rusqlite::Error> {
    query_chat_messages(
        conn,
        &format!(
            "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
          WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
          ORDER BY timestamp_unix DESC, message_id DESC
          LIMIT ?"
        ),
        rusqlite::params![chat_id, limit],
        self_sender_aliases,
    )
}

fn load_chat_messages_before(
    conn: &rusqlite::Connection,
    chat_id: &str,
    timestamp_unix: i64,
    message_id: i64,
    inclusive: bool,
    limit: i64,
    self_sender_aliases: &HashSet<String>,
) -> Result<Vec<BackupMessage>, rusqlite::Error> {
    let boundary_operator = if inclusive { "<=" } else { "<" };
    if timestamp_unix <= 0 {
        return query_chat_messages(
            conn,
            &format!(
                "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
                  WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                    AND timestamp_unix IS NULL AND message_id {boundary_operator} ?
                  ORDER BY message_id DESC
                  LIMIT ?"
            ),
            rusqlite::params![chat_id, message_id, limit],
            self_sender_aliases,
        );
    }

    let mut messages = query_chat_messages(
        conn,
        &format!(
            "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
              WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                AND (timestamp_unix, message_id) {boundary_operator} (?, ?)
              ORDER BY timestamp_unix DESC, message_id DESC
              LIMIT ?"
        ),
        rusqlite::params![chat_id, timestamp_unix, message_id, limit],
        self_sender_aliases,
    )?;
    let remaining = limit.saturating_sub(messages.len() as i64);
    if remaining > 0 {
        messages.extend(query_chat_messages(
            conn,
            &format!(
                "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
                  WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                    AND timestamp_unix IS NULL
                  ORDER BY message_id DESC
                  LIMIT ?"
            ),
            rusqlite::params![chat_id, remaining],
            self_sender_aliases,
        )?);
    }
    Ok(messages)
}

fn load_chat_messages_after(
    conn: &rusqlite::Connection,
    chat_id: &str,
    timestamp_unix: i64,
    message_id: i64,
    limit: i64,
    self_sender_aliases: &HashSet<String>,
) -> Result<Vec<BackupMessage>, rusqlite::Error> {
    if timestamp_unix <= 0 {
        let mut messages = query_chat_messages(
            conn,
            &format!(
                "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
                  WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                    AND timestamp_unix IS NULL AND message_id > ?
                  ORDER BY message_id ASC
                  LIMIT ?"
            ),
            rusqlite::params![chat_id, message_id, limit],
            self_sender_aliases,
        )?;
        let remaining = limit.saturating_sub(messages.len() as i64);
        if remaining > 0 {
            messages.extend(query_chat_messages(
                conn,
                &format!(
                    "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
                      WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                        AND timestamp_unix IS NOT NULL
                      ORDER BY timestamp_unix ASC, message_id ASC
                      LIMIT ?"
                ),
                rusqlite::params![chat_id, remaining],
                self_sender_aliases,
            )?);
        }
        return Ok(messages);
    }

    query_chat_messages(
        conn,
        &format!(
            "SELECT {CHAT_MESSAGE_COLUMNS} FROM messages
          WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
            AND (timestamp_unix, message_id) > (?, ?)
          ORDER BY timestamp_unix ASC, message_id ASC
          LIMIT ?"
        ),
        rusqlite::params![chat_id, timestamp_unix, message_id, limit],
        self_sender_aliases,
    )
}

fn has_message_before(
    conn: &rusqlite::Connection,
    chat_id: &str,
    message: &BackupMessage,
) -> Result<bool, rusqlite::Error> {
    if message.timestamp_unix <= 0 {
        return conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM messages
                  WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                    AND timestamp_unix IS NULL AND message_id < ?
             )",
            rusqlite::params![chat_id, message.message_id],
            |row| row.get(0),
        );
    }
    let has_timed_message = conn.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM messages
              WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                AND (timestamp_unix, message_id) < (?, ?)
         )",
        rusqlite::params![chat_id, message.timestamp_unix, message.message_id],
        |row| row.get(0),
    )?;
    if has_timed_message {
        return Ok(true);
    }
    conn.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM messages
              WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                AND timestamp_unix IS NULL
         )",
        rusqlite::params![chat_id],
        |row| row.get(0),
    )
}

fn has_message_after(
    conn: &rusqlite::Connection,
    chat_id: &str,
    message: &BackupMessage,
) -> Result<bool, rusqlite::Error> {
    if message.timestamp_unix <= 0 {
        return conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM messages
                  WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                    AND (
                        (timestamp_unix IS NULL AND message_id > ?)
                        OR timestamp_unix IS NOT NULL
                    )
             )",
            rusqlite::params![chat_id, message.message_id],
            |row| row.get(0),
        );
    }
    conn.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM messages
              WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                AND (timestamp_unix, message_id) > (?, ?)
         )",
        rusqlite::params![chat_id, message.timestamp_unix, message.message_id],
        |row| row.get(0),
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn load_chat_page(
    conn: &rusqlite::Connection,
    chat_id: &str,
    backup_name: String,
    request: ChatPageRequest,
) -> Result<ChatMessagePage, rusqlite::Error> {
    load_chat_page_with_aliases(conn, chat_id, backup_name, request, None)
}

pub(crate) fn load_chat_page_with_aliases(
    conn: &rusqlite::Connection,
    chat_id: &str,
    backup_name: String,
    request: ChatPageRequest,
    cached_self_sender_aliases: Option<&HashSet<String>>,
) -> Result<ChatMessagePage, rusqlite::Error> {
    let self_sender_aliases = cached_self_sender_aliases
        .cloned()
        .map(Ok)
        .unwrap_or_else(|| resolve_self_sender_aliases(conn, chat_id))?;
    let cached_count = conn
        .query_row(
            "SELECT msg_count FROM chats WHERE chat_id=?",
            rusqlite::params![chat_id],
            |row| row.get::<_, Option<i64>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten();
    let total_messages = if let Some(count) = cached_count {
        count
    } else {
        conn.query_row(
            "SELECT count(*) FROM messages WHERE chat_id=? AND COALESCE(is_deleted, 0)=0",
            rusqlite::params![chat_id],
            |row| row.get(0),
        )?
    };
    let (mut messages, mut focus_message_id) = match request {
        ChatPageRequest::Latest => (
            load_latest_chat_messages(conn, chat_id, CHAT_VIEW_PAGE_SIZE, &self_sender_aliases)?,
            None,
        ),
        ChatPageRequest::Before {
            timestamp_unix,
            message_id,
        } => (
            load_chat_messages_before(
                conn,
                chat_id,
                timestamp_unix,
                message_id,
                false,
                CHAT_VIEW_PAGE_SIZE,
                &self_sender_aliases,
            )?,
            None,
        ),
        ChatPageRequest::After {
            timestamp_unix,
            message_id,
        } => (
            load_chat_messages_after(
                conn,
                chat_id,
                timestamp_unix,
                message_id,
                CHAT_VIEW_PAGE_SIZE,
                &self_sender_aliases,
            )?,
            None,
        ),
        ChatPageRequest::Around { message_id } => {
            let anchor = conn.query_row(
                "SELECT COALESCE(timestamp_unix, 0), message_id
                   FROM messages
                  WHERE chat_id=? AND message_id=? AND COALESCE(is_deleted, 0)=0",
                rusqlite::params![chat_id, message_id],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )?;
            let half = CHAT_VIEW_PAGE_SIZE / 2;
            let mut page = load_chat_messages_before(
                conn,
                chat_id,
                anchor.0,
                anchor.1,
                true,
                half,
                &self_sender_aliases,
            )?;
            page.extend(load_chat_messages_after(
                conn,
                chat_id,
                anchor.0,
                anchor.1,
                half,
                &self_sender_aliases,
            )?);
            page.sort_by_key(|message| (message.timestamp_unix, message.message_id));
            page.dedup_by_key(|message| message.message_id);
            (page, Some(message_id))
        }
    };
    messages.sort_by_key(|message| (message.timestamp_unix, message.message_id));
    if focus_message_id.is_none() {
        focus_message_id = match request {
            ChatPageRequest::Before { .. } => messages.last().map(|message| message.message_id),
            ChatPageRequest::After { .. } => messages.first().map(|message| message.message_id),
            ChatPageRequest::Latest | ChatPageRequest::Around { .. } => focus_message_id,
        };
    }
    let has_older = messages
        .first()
        .map(|message| has_message_before(conn, chat_id, message))
        .transpose()?
        .unwrap_or(false);
    let has_newer = messages
        .last()
        .map(|message| has_message_after(conn, chat_id, message))
        .transpose()?
        .unwrap_or(false);
    Ok(ChatMessagePage {
        backup_name,
        chat_id: chat_id.to_string(),
        messages,
        total_messages,
        has_older,
        has_newer,
        focus_message_id,
        self_sender_aliases,
    })
}

fn fts_prefix_query(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(|token| format!("\"{}\"*", token.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn count_search_messages(
    conn: &rusqlite::Connection,
    query: &str,
    chat_id: Option<&str>,
) -> Result<usize, rusqlite::Error> {
    let fts_query = fts_prefix_query(query);
    if fts_query.is_empty() {
        return Ok(0);
    }
    let mut predicate = String::from(
        " FROM messages_fts
          JOIN messages ON messages.id=messages_fts.rowid
         WHERE messages_fts MATCH ? AND COALESCE(messages.is_deleted, 0)=0",
    );
    if chat_id.is_some() {
        predicate.push_str(" AND messages.chat_id=?");
    }
    let sql = format!("SELECT COUNT(*){predicate}");
    let total_matches: i64 = if let Some(chat_id) = chat_id {
        conn.query_row(&sql, rusqlite::params![fts_query, chat_id], |row| {
            row.get(0)
        })?
    } else {
        conn.query_row(&sql, rusqlite::params![fts_query], |row| row.get(0))?
    };
    Ok(total_matches.max(0) as usize)
}

pub(crate) fn search_message_page(
    conn: &rusqlite::Connection,
    query: &str,
    chat_id: Option<&str>,
    limit: i64,
    offset: usize,
) -> Result<Vec<MessageSearchResult>, rusqlite::Error> {
    let fts_query = fts_prefix_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let mut sql = String::from(
        "SELECT messages.chat_id, COALESCE(chats.chat_name, messages.chat_id),
                messages.message_id, messages.sender, messages.timestamp,
                messages.text, messages.media_type
           FROM messages_fts
           JOIN messages ON messages.id=messages_fts.rowid
           LEFT JOIN chats ON chats.chat_id=messages.chat_id
          WHERE messages_fts MATCH ? AND COALESCE(messages.is_deleted, 0)=0",
    );
    if chat_id.is_some() {
        sql.push_str(" AND messages.chat_id=?");
    }
    sql.push_str(
        " ORDER BY COALESCE(messages.timestamp_unix, 0) DESC, messages.message_id DESC
          LIMIT ? OFFSET ?",
    );
    let mut stmt = conn.prepare(&sql)?;
    let map_row = |row: &rusqlite::Row<'_>| {
        Ok(MessageSearchResult {
            chat_id: row.get(0)?,
            chat_name: row.get(1)?,
            message_id: row.get(2)?,
            sender: row
                .get::<_, Option<String>>(3)?
                .unwrap_or_else(|| "Unknown".to_string()),
            timestamp_str: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            text: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            media_type: row.get(6)?,
        })
    };
    let results = if let Some(chat_id) = chat_id {
        stmt.query_map(
            rusqlite::params![fts_query, chat_id, limit, offset as i64],
            map_row,
        )?
        .collect::<Result<Vec<_>, _>>()?
    } else {
        stmt.query_map(rusqlite::params![fts_query, limit, offset as i64], map_row)?
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(results)
}

pub(crate) fn search_message_rowids(
    conn: &rusqlite::Connection,
    query: &str,
) -> Result<Vec<i64>, rusqlite::Error> {
    let fts_query = fts_prefix_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT messages.id
           FROM messages_fts
           JOIN messages ON messages.id=messages_fts.rowid
          WHERE messages_fts MATCH ? AND COALESCE(messages.is_deleted, 0)=0
          ORDER BY COALESCE(messages.timestamp_unix, 0) DESC, messages.message_id DESC",
    )?;
    stmt.query_map(rusqlite::params![fts_query], |row| row.get(0))?
        .collect()
}

pub(crate) fn load_search_results_by_rowids(
    conn: &rusqlite::Connection,
    row_ids: &[i64],
) -> Result<Vec<MessageSearchResult>, rusqlite::Error> {
    if row_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = std::iter::repeat_n("?", row_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT messages.id, messages.chat_id,
                COALESCE(chats.chat_name, messages.chat_id), messages.message_id,
                messages.sender, messages.timestamp, messages.text, messages.media_type
           FROM messages
           LEFT JOIN chats ON chats.chat_id=messages.chat_id
          WHERE messages.id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut by_row_id = HashMap::with_capacity(row_ids.len());
    let rows = stmt.query_map(rusqlite::params_from_iter(row_ids), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            MessageSearchResult {
                chat_id: row.get(1)?,
                chat_name: row.get(2)?,
                message_id: row.get(3)?,
                sender: row
                    .get::<_, Option<String>>(4)?
                    .unwrap_or_else(|| "Unknown".to_string()),
                timestamp_str: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                text: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
                media_type: row.get(7)?,
            },
        ))
    })?;
    for row in rows {
        let (row_id, result) = row?;
        by_row_id.insert(row_id, result);
    }
    let ordered = row_ids
        .iter()
        .filter_map(|row_id| by_row_id.remove(row_id))
        .collect::<Vec<_>>();
    if ordered.len() != row_ids.len() {
        return Err(rusqlite::Error::QueryReturnedNoRows);
    }
    Ok(ordered)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn search_messages(
    conn: &rusqlite::Connection,
    query: &str,
    chat_id: Option<&str>,
    limit: i64,
    offset: usize,
) -> Result<(Vec<MessageSearchResult>, usize), rusqlite::Error> {
    let total_matches = count_search_messages(conn, query, chat_id)?;
    let results = search_message_page(conn, query, chat_id, limit, offset)?;
    Ok((results, total_matches))
}

pub(crate) fn ensure_blacklist_schema(conn: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS telegram_backup_blacklist (
             target_key TEXT PRIMARY KEY,
             peer_kind TEXT NOT NULL,
             peer_id INTEGER NOT NULL,
             title TEXT NOT NULL,
             reason TEXT,
             created_unix INTEGER NOT NULL,
             UNIQUE(peer_kind, peer_id)
         )",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS telegram_backup_diagnostic_events (
             event_id TEXT PRIMARY KEY,
             event_unix INTEGER NOT NULL,
             event_type TEXT NOT NULL,
             component TEXT NOT NULL,
             level TEXT NOT NULL DEFAULT 'info',
             operation_id TEXT,
             run_key TEXT,
             target_key TEXT,
             status TEXT,
             details_json TEXT NOT NULL,
             build_revision TEXT NOT NULL,
             actor TEXT NOT NULL DEFAULT 'local-user',
             writer_role TEXT NOT NULL DEFAULT 'tgbackman-gui',
             reason TEXT,
             outcome TEXT,
             host_name TEXT NOT NULL DEFAULT '',
             process_id INTEGER NOT NULL DEFAULT 0,
             previous_hash TEXT,
             integrity_sha256 TEXT
         )",
        [],
    )?;
    for (name, definition) in [
        ("actor", "TEXT NOT NULL DEFAULT 'local-user'"),
        ("writer_role", "TEXT NOT NULL DEFAULT 'tgbackman-gui'"),
        ("reason", "TEXT"),
        ("outcome", "TEXT"),
        ("host_name", "TEXT NOT NULL DEFAULT ''"),
        ("process_id", "INTEGER NOT NULL DEFAULT 0"),
        ("previous_hash", "TEXT"),
        ("integrity_sha256", "TEXT"),
    ] {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pragma_table_info('telegram_backup_diagnostic_events')
                 WHERE name = ?1
             )",
            [name],
            |row| row.get(0),
        )?;
        if !exists {
            conn.execute(
                &format!(
                    "ALTER TABLE telegram_backup_diagnostic_events ADD COLUMN {name} {definition}"
                ),
                [],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn record_gui_event(
    conn: &rusqlite::Connection,
    event_type: &str,
    status: &str,
    chat_ids: &[String],
) -> Result<(), rusqlite::Error> {
    record_gui_event_with_reason(
        conn,
        event_type,
        status,
        chat_ids,
        "user-requested GUI mutation",
        None,
    )
}

pub(crate) fn record_gui_event_with_reason(
    conn: &rusqlite::Connection,
    event_type: &str,
    status: &str,
    chat_ids: &[String],
    reason: &str,
    error: Option<&str>,
) -> Result<(), rusqlite::Error> {
    ensure_blacklist_schema(conn)?;
    let now = Utc::now().timestamp();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let event_id = format!("gui-{}-{}", unique, std::process::id());
    let details = serde_json::json!({
        "chat_count": chat_ids.len(),
        "chat_ids": chat_ids.iter().take(1000).collect::<Vec<_>>(),
        "error": error.map(|value| value.chars().take(512).collect::<String>()),
    });
    let actor = std::env::var("TGBACKMAN_ACTOR")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "local-user".to_string());
    let host_name = std::env::var("HOSTNAME").unwrap_or_default();
    conn.execute(
        "INSERT OR IGNORE INTO telegram_backup_diagnostic_events (
             event_id, event_unix, event_type, component, level, operation_id,
             run_key, target_key, status, details_json, build_revision,
             actor, writer_role, reason, outcome, host_name, process_id
         ) VALUES (?, ?, ?, 'tgbackman-gui', ?, NULL, NULL, NULL, ?, ?, ?,
                   ?, 'tgbackman-gui', ?, ?, ?, ?)",
        rusqlite::params![
            event_id,
            now,
            event_type,
            if status == "failed" { "error" } else { "info" },
            status,
            details.to_string(),
            option_env!("TGBACKMAN_BUILD_REVISION").unwrap_or("unknown"),
            actor,
            reason,
            status,
            host_name,
            std::process::id(),
        ],
    )?;
    Ok(())
}

pub(crate) fn record_gui_failure(
    db_path: &str,
    event_type: &str,
    chat_ids: &[String],
    error: &str,
) {
    if let Ok(conn) = rusqlite::Connection::open(db_path) {
        let _ = record_gui_event_with_reason(
            &conn,
            event_type,
            "failed",
            chat_ids,
            "GUI mutation failed",
            Some(error),
        );
    }
}

pub(crate) fn blacklisted_chat_ids(
    conn: &rusqlite::Connection,
) -> Result<HashSet<String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")?
        || !table_exists(conn, "telegram_backup_target_chats")?
    {
        return Ok(HashSet::new());
    }
    let mut stmt = conn.prepare(
        "SELECT DISTINCT targets.chat_id
         FROM telegram_backup_targets AS targets
         JOIN telegram_backup_blacklist AS blacklist
           ON blacklist.target_key = targets.target_key
           OR (blacklist.peer_kind = targets.peer_kind AND blacklist.peer_id = targets.peer_id)
         UNION
         SELECT DISTINCT links.chat_id
         FROM telegram_backup_targets AS targets
         JOIN telegram_backup_blacklist AS blacklist
           ON blacklist.target_key = targets.target_key
           OR (blacklist.peer_kind = targets.peer_kind AND blacklist.peer_id = targets.peer_id)
         JOIN telegram_backup_target_chats AS links ON links.target_key = targets.target_key",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect()
}

pub(crate) fn set_chat_ids_blacklisted(
    conn: &mut rusqlite::Connection,
    chat_ids: &[String],
    blacklisted: bool,
) -> Result<usize, rusqlite::Error> {
    ensure_blacklist_schema(conn)?;
    let tx = conn.transaction()?;
    let mut affected = 0;
    for chat_id in chat_ids {
        if blacklisted {
            affected += tx.execute(
                "INSERT OR IGNORE INTO telegram_backup_blacklist (
                     target_key, peer_kind, peer_id, title, reason, created_unix
                 )
                 SELECT targets.target_key, targets.peer_kind, targets.peer_id,
                        targets.title, 'Added in tgbackman', ?
                 FROM telegram_backup_targets AS targets
                 WHERE targets.chat_id = ? OR EXISTS (
                     SELECT 1 FROM telegram_backup_target_chats AS links
                     WHERE links.target_key = targets.target_key AND links.chat_id = ?
                 )",
                rusqlite::params![Utc::now().timestamp(), chat_id, chat_id],
            )?;
            tx.execute(
                "UPDATE chats SET is_active=0 WHERE chat_id=?",
                rusqlite::params![chat_id],
            )?;
        } else {
            affected += tx.execute(
                "DELETE FROM telegram_backup_blacklist
                 WHERE target_key IN (
                     SELECT targets.target_key
                     FROM telegram_backup_targets AS targets
                     WHERE targets.chat_id = ? OR EXISTS (
                         SELECT 1 FROM telegram_backup_target_chats AS links
                         WHERE links.target_key = targets.target_key AND links.chat_id = ?
                     )
                 )",
                rusqlite::params![chat_id, chat_id],
            )?;
        }
    }
    record_gui_event(
        &tx,
        if blacklisted {
            "chat_blacklisted"
        } else {
            "chat_blacklist_removed"
        },
        "completed",
        chat_ids,
    )?;
    tx.commit()?;
    Ok(affected)
}

pub(crate) fn materialize_discovered_chats(
    conn: &rusqlite::Connection,
) -> Result<usize, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")?
        || !table_exists(conn, "telegram_backup_target_chats")?
    {
        return Ok(0);
    }

    // Older `map --all` runs recorded unmatched Telegram dialogs as targets
    // without adding them to `chats`, which is the GUI inventory and active
    // selection table. Materialize enabled targets and disabled blacklist
    // tombstones with no archive link. The latter must remain visible so the
    // user can remove their never-back-up rule after purging their contents.
    let inserted = conn.execute(
        "INSERT OR IGNORE INTO chats (
             chat_id, chat_name, chat_type, backup_path, is_active,
             last_backup_unix, msg_count
         )
         SELECT targets.chat_id,
                targets.title,
                CASE targets.peer_kind
                    WHEN 'user' THEN 'personal_chat'
                    ELSE targets.peer_kind
                END,
                targets.output_dir,
                0,
                NULL,
                0
         FROM telegram_backup_targets AS targets
         WHERE (
                 COALESCE(targets.enabled, 1) = 1
                 OR EXISTS (
                     SELECT 1 FROM telegram_backup_blacklist AS blacklist
                     WHERE blacklist.target_key = targets.target_key
                        OR (blacklist.peer_kind = targets.peer_kind AND blacklist.peer_id = targets.peer_id)
                 )
               )
           AND NOT EXISTS (
               SELECT 1 FROM telegram_backup_target_chats AS links
               WHERE links.target_key = targets.target_key
           )",
        [],
    )?;

    conn.execute(
        "INSERT OR IGNORE INTO telegram_backup_target_chats (
             target_key, chat_id, match_method, linked_unix
         )
         SELECT targets.target_key,
                targets.chat_id,
                'telegram-discovered',
                ?
         FROM telegram_backup_targets AS targets
         JOIN chats ON chats.chat_id = targets.chat_id
         WHERE (
                 COALESCE(targets.enabled, 1) = 1
                 OR EXISTS (
                     SELECT 1 FROM telegram_backup_blacklist AS blacklist
                     WHERE blacklist.target_key = targets.target_key
                        OR (blacklist.peer_kind = targets.peer_kind AND blacklist.peer_id = targets.peer_id)
                 )
               )
           AND NOT EXISTS (
               SELECT 1 FROM telegram_backup_target_chats AS links
               WHERE links.target_key = targets.target_key
           )",
        rusqlite::params![Utc::now().timestamp()],
    )?;

    Ok(inserted)
}

pub(crate) fn zero_message_migrated_predecessors(
    conn: &rusqlite::Connection,
) -> Result<HashSet<String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_target_chats")? {
        return Ok(HashSet::new());
    }

    let mut stmt = conn.prepare(
        "SELECT links.chat_id
         FROM telegram_backup_target_chats AS links
         WHERE links.match_method = 'telegram-migrated-from'
           AND NOT EXISTS (
               SELECT 1 FROM messages
               WHERE messages.chat_id = links.chat_id
           )",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect()
}

pub(crate) fn apply_authoritative_target_links(
    conn: &rusqlite::Connection,
    uf: &mut UnionFind,
) -> Result<HashMap<String, String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")?
        || !table_exists(conn, "telegram_backup_target_chats")?
    {
        return Ok(HashMap::new());
    }

    let mut stmt = conn.prepare(
        "SELECT targets.target_key, targets.title, links.chat_id
         FROM telegram_backup_targets AS targets
         JOIN telegram_backup_target_chats AS links
           ON links.target_key = targets.target_key
         WHERE COALESCE(targets.enabled, 1) = 1
         ORDER BY targets.target_key, links.chat_id",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let links: Vec<(String, String, String)> = rows.collect::<Result<_, _>>()?;

    let mut first_chat_by_target: HashMap<String, String> = HashMap::new();
    for (target_key, _, chat_id) in &links {
        if let Some(first_chat) = first_chat_by_target.get(target_key) {
            uf.union(first_chat, chat_id);
        } else {
            first_chat_by_target.insert(target_key.clone(), chat_id.clone());
        }
    }

    // Prefer the current Telegram title only when a logical group maps to one
    // unambiguous enabled target. This keeps renamed predecessor archives under
    // the current conversation name without guessing across unrelated targets.
    let mut titles_by_root: HashMap<String, HashSet<String>> = HashMap::new();
    for (_, title, chat_id) in links {
        let root = uf.find(&chat_id);
        titles_by_root.entry(root).or_default().insert(title);
    }
    Ok(titles_by_root
        .into_iter()
        .filter_map(|(root, titles)| {
            if titles.len() == 1 {
                Some((root, titles.into_iter().next().unwrap()))
            } else {
                None
            }
        })
        .collect())
}

/// Return the stable Telegram peer identity attached to each indexed chat.
///
/// Display names are mutable (and are not unique), while the peer kind/id is
/// stable.  Keeping this mapping separate lets clustering avoid joining two
/// unrelated conversations that happen to share a title.
pub(crate) fn target_peer_identities(
    conn: &rusqlite::Connection,
) -> Result<HashMap<String, String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")?
        || !table_exists(conn, "telegram_backup_target_chats")?
    {
        return Ok(HashMap::new());
    }

    let mut stmt = conn.prepare(
        "SELECT links.chat_id, targets.peer_kind, targets.peer_id
         FROM telegram_backup_targets AS targets
         JOIN telegram_backup_target_chats AS links
           ON links.target_key = targets.target_key
         WHERE COALESCE(targets.enabled, 1) = 1
         UNION
         SELECT targets.chat_id, targets.peer_kind, targets.peer_id
         FROM telegram_backup_targets AS targets
         WHERE COALESCE(targets.enabled, 1) = 1",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    let mut identities = HashMap::new();
    for row in rows {
        let (chat_id, peer_kind, peer_id) = row?;
        identities.insert(chat_id, format!("{}:{}", peer_kind, peer_id));
    }
    Ok(identities)
}

/// Return normalized target titles which are used by more than one stable
/// Telegram peer.  These names must be shown with an identity suffix because
/// a title alone cannot tell the conversations apart.
pub(crate) fn ambiguous_target_title_norms(
    conn: &rusqlite::Connection,
) -> Result<HashSet<String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")? {
        return Ok(HashSet::new());
    }

    let mut stmt = conn.prepare(
        "SELECT title, peer_kind, peer_id
         FROM telegram_backup_targets",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    let mut identities_by_title: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows {
        let (title, peer_kind, peer_id) = row?;
        let norm = title.trim().to_lowercase();
        if !norm.is_empty() {
            identities_by_title
                .entry(norm)
                .or_default()
                .insert(format!("{}:{}", peer_kind, peer_id));
        }
    }

    Ok(identities_by_title
        .into_iter()
        .filter_map(|(title, identities)| (identities.len() > 1).then_some(title))
        .collect())
}

pub(crate) fn ambiguous_target_roots(
    conn: &rusqlite::Connection,
    uf: &mut UnionFind,
) -> Result<HashSet<String>, rusqlite::Error> {
    if !table_exists(conn, "telegram_backup_targets")? {
        return Ok(HashSet::new());
    }

    let mut stmt = conn.prepare(
        "SELECT title, peer_kind, peer_id, chat_id
         FROM telegram_backup_targets",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    let mut identities_by_title: HashMap<String, HashSet<String>> = HashMap::new();
    let mut roots_by_title: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows {
        let (title, peer_kind, peer_id, chat_id) = row?;
        let norm = title.trim().to_lowercase();
        if norm.is_empty() {
            continue;
        }
        identities_by_title
            .entry(norm.clone())
            .or_default()
            .insert(format!("{}:{}", peer_kind, peer_id));
        roots_by_title
            .entry(norm)
            .or_default()
            .insert(uf.find(&chat_id));
    }

    let mut ambiguous_roots = HashSet::new();
    for (title, identities) in identities_by_title {
        if identities.len() > 1 {
            if let Some(roots) = roots_by_title.remove(&title) {
                ambiguous_roots.extend(roots);
            }
        }
    }
    Ok(ambiguous_roots)
}

pub(crate) fn incompatible_peer_identities(
    first: &str,
    second: &str,
    identities: &HashMap<String, String>,
) -> bool {
    matches!((identities.get(first), identities.get(second)), (Some(a), Some(b)) if a != b)
}

fn ensure_column(
    conn: &rusqlite::Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), rusqlite::Error> {
    let present: i64 = conn.query_row(
        &format!(
            "SELECT count(*) FROM pragma_table_info('{}') WHERE name = ?1",
            table.replace('\'', "''")
        ),
        [column],
        |row| row.get(0),
    )?;
    if present == 0 {
        conn.execute(definition, [])?;
    }
    Ok(())
}

pub(crate) fn run_inventory(
    conn: &rusqlite::Connection,
    db_path: &str,
) -> Result<Vec<ChatGroup>, rusqlite::Error> {
    let start_total = std::time::Instant::now();

    for (table, column, definition) in [
        (
            "chats",
            "is_active",
            "ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0",
        ),
        (
            "chats",
            "last_backup_unix",
            "ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER",
        ),
        (
            "chats",
            "last_backup_run_unix",
            "ALTER TABLE chats ADD COLUMN last_backup_run_unix INTEGER",
        ),
        (
            "chats",
            "last_backup_run_status",
            "ALTER TABLE chats ADD COLUMN last_backup_run_status TEXT",
        ),
        (
            "chats",
            "last_backup_source",
            "ALTER TABLE chats ADD COLUMN last_backup_source TEXT",
        ),
        (
            "chats",
            "last_backup_confidence",
            "ALTER TABLE chats ADD COLUMN last_backup_confidence TEXT",
        ),
        (
            "chats",
            "last_backup_evidence",
            "ALTER TABLE chats ADD COLUMN last_backup_evidence TEXT",
        ),
        (
            "chats",
            "min_msg_id",
            "ALTER TABLE chats ADD COLUMN min_msg_id INTEGER",
        ),
        (
            "chats",
            "max_msg_id",
            "ALTER TABLE chats ADD COLUMN max_msg_id INTEGER",
        ),
        (
            "chats",
            "msg_count",
            "ALTER TABLE chats ADD COLUMN msg_count INTEGER",
        ),
        (
            "chats",
            "min_timestamp",
            "ALTER TABLE chats ADD COLUMN min_timestamp TEXT",
        ),
        (
            "chats",
            "max_timestamp",
            "ALTER TABLE chats ADD COLUMN max_timestamp TEXT",
        ),
        (
            "chats",
            "min_timestamp_unix",
            "ALTER TABLE chats ADD COLUMN min_timestamp_unix INTEGER",
        ),
        (
            "chats",
            "max_timestamp_unix",
            "ALTER TABLE chats ADD COLUMN max_timestamp_unix INTEGER",
        ),
        (
            "messages",
            "is_deleted",
            "ALTER TABLE messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "messages",
            "deleted_unix",
            "ALTER TABLE messages ADD COLUMN deleted_unix INTEGER",
        ),
    ] {
        ensure_column(conn, table, column, definition)?;
    }
    // Message tombstones were added after the original archive schema.  Keep
    // inventory usable on older databases while the GUI performs its full
    // checked migration on load.
    let _ = conn.execute(
        "ALTER TABLE messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0;",
        [],
    );
    let _ = conn.execute("ALTER TABLE messages ADD COLUMN deleted_unix INTEGER;", []);

    ensure_blacklist_schema(conn)?;
    let discovered = materialize_discovered_chats(conn)?;
    if discovered > 0 {
        println!(
            "Added {} Telegram-discovered chat(s) with no backup to the inventory.",
            discovered
        );
    }
    let blacklisted_chat_ids = blacklisted_chat_ids(conn)?;
    let hidden_migrated_chats = zero_message_migrated_predecessors(conn)?;
    if !hidden_migrated_chats.is_empty() {
        println!(
            "Hiding {} zero-message migrated predecessor(s) from the inventory.",
            hidden_migrated_chats.len()
        );
    }

    let mut stmt = conn.prepare(
        "SELECT chat_id, chat_name, backup_path, COALESCE(is_active, 0), last_backup_unix, \
         last_backup_run_unix, COALESCE(last_backup_run_status, ''), \
         min_msg_id, max_msg_id, msg_count, min_timestamp, max_timestamp, min_timestamp_unix, max_timestamp_unix, \
         COALESCE(last_backup_source, ''), COALESCE(last_backup_confidence, ''), \
         COALESCE(last_backup_evidence, '') FROM chats"
    )?;
    let mut rows = stmt.query([])?;
    let mut chats = Vec::new();

    while let Some(row) = rows.next()? {
        let chat_id: String = row.get(0)?;
        if hidden_migrated_chats.contains(&chat_id) {
            continue;
        }
        let name: Option<String> = row.get(1)?;
        let path: Option<String> = row.get(2)?;
        let active: i32 = row.get(3)?;
        let last_backup: Option<i64> = row.get(4)?;
        let last_backup_run: Option<i64> = row.get(5)?;
        let last_backup_run_status: String = row.get(6)?;
        let min_msg_id: Option<i64> = row.get(7)?;
        let max_msg_id: Option<i64> = row.get(8)?;
        let msg_count: Option<i64> = row.get(9)?;
        let min_timestamp: Option<String> = row.get(10)?;
        let max_timestamp: Option<String> = row.get(11)?;
        let min_timestamp_unix: Option<i64> = row.get(12)?;
        let max_timestamp_unix: Option<i64> = row.get(13)?;
        let last_backup_source: String = row.get(14)?;
        let last_backup_confidence: String = row.get(15)?;
        let last_backup_evidence: String = row.get(16)?;

        chats.push(ChatRow {
            is_blacklisted: blacklisted_chat_ids.contains(&chat_id),
            chat_id,
            name: name.unwrap_or_default(),
            path,
            is_active: active != 0,
            last_backup_unix: last_backup,
            last_backup_run_unix: last_backup_run,
            last_backup_run_status,
            last_backup_source,
            last_backup_confidence,
            last_backup_evidence,
            min_msg_id,
            max_msg_id,
            msg_count,
            min_timestamp,
            max_timestamp,
            min_timestamp_unix,
            max_timestamp_unix,
        });
    }

    println!("Phase 1 (Chats fetch): {:?}", start_total.elapsed());

    let target_peer_identities = target_peer_identities(conn)?;
    let ambiguous_target_titles = ambiguous_target_title_norms(conn)?;

    let start_fuzzy = std::time::Instant::now();
    let mut uf = UnionFind::new();
    let clusters_cache_path = get_clusters_cache_path(db_path);

    let mut loaded_cache = false;
    // Message-only API increments do not change legacy clustering. Importers
    // explicitly remove this cache when they add or rewrite legacy sources;
    // current target links are always applied below. Reusing the structural
    // cache across ordinary backups avoids an unnecessary archive-wide join.
    if std::path::Path::new(&clusters_cache_path).is_file() {
        if let Ok(file) = std::fs::File::open(&clusters_cache_path) {
            if let Ok(parent_map) = serde_json::from_reader::<_, HashMap<String, String>>(file) {
                uf.parent = parent_map;
                loaded_cache = true;
                println!("Loaded cached chat clusters from {}", clusters_cache_path);

                // A cache written before stable peer identities were enforced
                // may have merged two unrelated same-title peers.  Reject it
                // rather than displaying a stale duplicate collapse.
                let mut identities_by_root: HashMap<String, HashSet<String>> = HashMap::new();
                for chat in &chats {
                    if let Some(identity) = target_peer_identities.get(&chat.chat_id) {
                        let root = uf.find(&chat.chat_id);
                        identities_by_root
                            .entry(root)
                            .or_default()
                            .insert(identity.clone());
                    }
                }
                if identities_by_root.values().any(|set| set.len() > 1) {
                    println!("Ignoring chat-cluster cache containing unrelated Telegram peers.");
                    uf = UnionFind::new();
                    loaded_cache = false;
                }
            }
        }
    }

    if !loaded_cache {
        // 1. FUZZY ALIAS LINKING via oldest signatures
        let mut exact_signatures: HashMap<(i64, String), Vec<String>> = HashMap::new();
        let mut stmt_msgs = conn.prepare(
            "SELECT timestamp_unix, text FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0 AND text != '' AND timestamp_unix IS NOT NULL ORDER BY timestamp_unix ASC LIMIT 50"
        )?;
        for c in &chats {
            let mut rows_msgs = stmt_msgs.query(rusqlite::params![c.chat_id])?;
            while let Some(row) = rows_msgs.next()? {
                let ts: i64 = row.get(0)?;
                let text: String = row.get(1)?;
                let clean_text = text.trim();
                if clean_text.len() >= 6 {
                    exact_signatures
                        .entry((ts, clean_text.to_string()))
                        .or_default()
                        .push(c.chat_id.clone());
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
                    if incompatible_peer_identities(&c1, &c2, &target_peer_identities) {
                        continue;
                    }
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
        for c in &chats {
            if !c.name.is_empty() {
                let norm = c.name.trim().to_lowercase();
                if norm != "deleted account"
                    && norm != "telegram"
                    && norm != "group"
                    && norm != "unknown"
                {
                    chats_by_norm_name
                        .entry(norm)
                        .or_default()
                        .push(c.chat_id.clone());
                }
            }
        }

        let mut stmt_join = conn.prepare(
            "SELECT COUNT(*) FROM messages a JOIN messages b ON a.timestamp_unix = b.timestamp_unix WHERE COALESCE(a.is_deleted, 0)=0 AND COALESCE(b.is_deleted, 0)=0 AND a.chat_id = ? AND b.chat_id = ? AND a.text = b.text AND a.text != '' AND length(a.text) >= 6"
        )?;

        for (_, cids) in chats_by_norm_name {
            if cids.len() < 2 {
                continue;
            }
            for i in 0..cids.len() {
                for j in i + 1..cids.len() {
                    let c1 = &cids[i];
                    let c2 = &cids[j];
                    if incompatible_peer_identities(c1, c2, &target_peer_identities) {
                        continue;
                    }
                    if uf.find(c1) == uf.find(c2) {
                        continue;
                    }
                    let count: i64 =
                        stmt_join.query_row(rusqlite::params![c1, c2], |r| r.get(0))?;
                    if count >= 3 {
                        uf.union(c1, c2);
                    }
                }
            }
        }

        println!("Phase 3 (Same-name joins): {:?}", start_joins.elapsed());
    } else {
        println!(
            "Phase 2 & 3 (Clustering): skipped (loaded from cache in {:?})",
            start_fuzzy.elapsed()
        );
    }

    let preferred_group_names = apply_authoritative_target_links(conn, &mut uf)?;
    let ambiguous_target_roots = ambiguous_target_roots(conn, &mut uf)?;

    let start_stats = std::time::Instant::now();

    // `msg_count IS NULL` is the canonical invalidation marker used by bulk
    // importers. Incremental exporters refresh the affected chat atomically.
    // Re-scan only invalidated chats; scanning every multi-million-row chat on
    // every GUI launch made unchanged databases needlessly expensive to open.
    let _ = conn.execute("BEGIN TRANSACTION;", []);

    let mut stats_map = HashMap::new();
    let mut stmt_update_stats = conn.prepare(
        "UPDATE chats SET min_msg_id = ?, max_msg_id = ?, msg_count = ?, min_timestamp = ?, max_timestamp = ?, min_timestamp_unix = ?, max_timestamp_unix = ? WHERE chat_id = ?"
    )?;
    let mut stmt_calc_stats = conn.prepare(
        "SELECT MIN(message_id), MAX(message_id), COUNT(*), MIN(timestamp), MAX(timestamp), MIN(timestamp_unix), MAX(timestamp_unix) FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0"
    )?;

    for c in &chats {
        if let Some(count) = c.msg_count {
            stats_map.insert(
                c.chat_id.clone(),
                (
                    c.min_msg_id,
                    c.max_msg_id,
                    count,
                    c.min_timestamp.clone(),
                    c.max_timestamp.clone(),
                    c.min_timestamp_unix,
                    c.max_timestamp_unix,
                ),
            );
            continue;
        }
        let mut rows_calc = stmt_calc_stats.query(rusqlite::params![c.chat_id])?;
        if let Some(row) = rows_calc.next()? {
            let min_id: Option<i64> = row.get(0)?;
            let max_id: Option<i64> = row.get(1)?;
            let count: i64 = row.get(2)?;
            let min_ts: Option<String> = row.get(3)?;
            let max_ts: Option<String> = row.get(4)?;
            let min_unix: Option<i64> = row.get(5)?;
            let max_unix: Option<i64> = row.get(6)?;

            stmt_update_stats.execute(rusqlite::params![
                min_id, max_id, count, &min_ts, &max_ts, min_unix, max_unix, c.chat_id
            ])?;
            stats_map.insert(
                c.chat_id.clone(),
                (min_id, max_id, count, min_ts, max_ts, min_unix, max_unix),
            );
        }
    }

    let _ = conn.execute("COMMIT TRANSACTION;", []);

    // Mappings and Stats
    let mut logical_groups: HashMap<String, Vec<BackupInfo>> = HashMap::new();
    for c in chats {
        let cid = c.chat_id;
        let name = c.name;
        let path = c.path;
        let is_active = c.is_active;
        let is_blacklisted = c.is_blacklisted;
        let last_backup_unix = c.last_backup_unix;
        let last_backup_run_unix = c.last_backup_run_unix;
        let last_backup_run_status = c.last_backup_run_status;
        let last_backup_source = c.last_backup_source;
        let last_backup_confidence = c.last_backup_confidence;
        let last_backup_evidence = c.last_backup_evidence;

        let norm_name = name.trim().to_lowercase();
        let has_messages = stats_map
            .get(&cid)
            .is_some_and(|(_, _, count, _, _, _, _)| *count > 0);
        if has_messages
            && (norm_name == "deleted account"
                || norm_name == "telegram"
                || norm_name == "group"
                || norm_name == "unknown")
            && uf.find(&cid) == cid
        {
            continue;
        }

        let root = uf.find(&cid);

        if let Some(stats) = stats_map.get(&cid) {
            let (min_id, max_id, count, min_ts, max_ts, min_unix, max_unix) = stats;
            let format_ts = |ts_str: &Option<String>| -> String {
                match ts_str {
                    Some(s) => s.replace("T", " ").replace("Z", ""),
                    None => "Unknown".to_string(),
                }
            };

            logical_groups.entry(root).or_default().push(BackupInfo {
                chat_id: cid.clone(),
                name: if name.is_empty() {
                    "Unknown".to_string()
                } else if ambiguous_target_titles.contains(&norm_name) {
                    target_peer_identities
                        .get(&cid)
                        .map(|identity| format!("{} [{}]", name, identity))
                        .unwrap_or_else(|| name.clone())
                } else {
                    name.clone()
                },
                path: path.clone().unwrap_or_default(),
                min_id: *min_id,
                max_id: *max_id,
                count: *count,
                min_ts: format_ts(min_ts),
                max_ts: format_ts(max_ts),
                min_unix: *min_unix,
                max_unix: *max_unix,
                is_active,
                is_blacklisted,
                last_backup_unix,
                last_backup_run_unix,
                last_backup_run_status,
                last_backup_source,
                last_backup_confidence,
                last_backup_evidence,
                media_stats: None,
            });
        }
    }

    println!("Phase 4 (Stats queries): {:?}", start_stats.elapsed());

    // Keep the cluster cache newer than the database writes above. This also
    // refreshes a previously loaded cache so the next load can safely reuse it.
    let temporary = format!("{}.tmp-{}", clusters_cache_path, std::process::id());
    if let Ok(file) = std::fs::File::create(&temporary) {
        if serde_json::to_writer_pretty(file, &uf.parent).is_ok() {
            let _ = std::fs::rename(&temporary, &clusters_cache_path);
            secure_cache_file(&clusters_cache_path);
        } else {
            let _ = std::fs::remove_file(&temporary);
        }
    }

    let start_groups = std::time::Instant::now();

    let mut result_groups = Vec::new();
    for (root, mut entries) in logical_groups {
        if entries.is_empty() {
            continue;
        }
        entries.sort_by_key(|e| e.count);
        let max_count = entries.iter().map(|e| e.count).max().unwrap_or(0);
        let names: HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
        let mut names_vec: Vec<String> = names.into_iter().collect();
        names_vec.sort();
        let display_name = if ambiguous_target_roots.contains(&root) {
            names_vec.join(" / ")
        } else {
            preferred_group_names
                .get(&root)
                .cloned()
                .unwrap_or_else(|| names_vec.join(" / "))
        };

        result_groups.push(ChatGroup {
            name: display_name,
            max_count,
            backups: entries,
        });
    }

    result_groups.sort_by(|a, b| b.max_count.cmp(&a.max_count));
    println!(
        "Phase 5 (Grouping + final sorting): {:?}",
        start_groups.elapsed()
    );

    Ok(result_groups)
}

#[cfg(test)]
mod reply_tests {
    use super::{
        ensure_blacklist_schema, load_chat_messages, load_chat_page, load_chat_page_with_aliases,
        load_search_results_by_rowids, record_gui_event_with_reason, search_message_rowids,
        search_messages,
    };
    use crate::model::ChatPageRequest;

    fn test_connection() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE chats(chat_id TEXT PRIMARY KEY, chat_name TEXT);
             CREATE TABLE messages(
                 id INTEGER PRIMARY KEY, message_id INTEGER NOT NULL, chat_id TEXT NOT NULL,
                 sender TEXT, sender_id TEXT, timestamp_unix INTEGER, timestamp TEXT, text TEXT,
                 media_type TEXT, media_path TEXT, is_deleted INTEGER DEFAULT 0,
                 reply_to_id INTEGER, reply_to_chat_id TEXT, reply_to_peer_kind TEXT,
                 reply_to_peer_id INTEGER, reply_to_top_id INTEGER,
                 reply_to_story_id INTEGER, reply_quote_text TEXT,
                 reply_media_json TEXT, forwarded_from TEXT, edit_timestamp TEXT,
                 reactions_json TEXT, message_type TEXT, action_json TEXT, source_format TEXT,
                 UNIQUE(chat_id, message_id)
             );
             CREATE TABLE telegram_backup_targets(
                 target_key TEXT PRIMARY KEY, chat_id TEXT, peer_kind TEXT, peer_id INTEGER
             );
             CREATE TABLE telegram_backup_target_chats(
                 target_key TEXT, chat_id TEXT
             );
             INSERT INTO chats VALUES ('child', 'Child'), ('parent', 'Parent');
             INSERT INTO telegram_backup_targets
                 VALUES ('parent-key', 'parent', 'channel', 99);
             INSERT INTO telegram_backup_target_chats VALUES ('parent-key', 'parent');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text
             ) VALUES
                 (4, 'child', 'Local parent', 4, '1970-01-01T00:00:04Z', 'local body'),
                 (7, 'parent', 'Cross parent', 7, '1970-01-01T00:00:07Z', 'cross body');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text,
                 reply_to_id, reply_to_chat_id
             ) VALUES
                 (5, 'child', 'Me', 5, '1970-01-01T00:00:05Z', 'local reply', 4, 'child');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text,
                 reply_to_id, reply_to_peer_kind, reply_to_peer_id, reply_to_top_id
             ) VALUES
                 (20, 'child', 'Me', 20, '1970-01-01T00:00:20Z', 'cross reply',
                  7, 'channel', 99, 3);
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text,
                 reply_to_id, reply_quote_text
             ) VALUES
                 (21, 'child', 'Me', 21, '1970-01-01T00:00:21Z', 'missing reply',
                  999, 'preserved quote');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text,
                 reply_to_peer_kind, reply_to_peer_id, reply_to_story_id
             ) VALUES
                 (22, 'child', 'Me', 22, '1970-01-01T00:00:22Z', 'story reply',
                  'user', 42, 11);",
        )
        .unwrap();
        conn
    }

    #[test]
    fn gui_mutation_events_record_success_and_failure_context() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        ensure_blacklist_schema(&conn).unwrap();
        let chat_ids = vec!["chat-1".to_string()];
        record_gui_event_with_reason(
            &conn,
            "chat_active_state_changed",
            "active",
            &chat_ids,
            "user-requested GUI mutation",
            None,
        )
        .unwrap();
        record_gui_event_with_reason(
            &conn,
            "chat_blacklist_mutation_failed",
            "failed",
            &chat_ids,
            "GUI mutation failed",
            Some("database busy"),
        )
        .unwrap();
        let row = conn
            .query_row(
                "SELECT level, status, outcome, details_json, actor, process_id
                 FROM telegram_backup_diagnostic_events
                 WHERE event_type='chat_blacklist_mutation_failed'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row.0, "error");
        assert_eq!(row.1, "failed");
        assert_eq!(row.2, "failed");
        assert!(row.3.contains("database busy"));
        assert!(!row.4.is_empty());
        assert!(row.5 > 0);
    }

    #[test]
    fn resolves_local_and_cross_chat_replies_with_fallbacks() {
        let conn = test_connection();
        let messages = load_chat_messages(&conn, "child", 100).unwrap();

        let local = messages
            .iter()
            .find(|message| message.message_id == 5)
            .unwrap();
        let local_reply = local.reply.as_ref().unwrap();
        assert_eq!(local_reply.sender.as_deref(), Some("Local parent"));
        assert_eq!(local_reply.text, "local body");
        assert!(!local_reply.missing);

        let cross = messages
            .iter()
            .find(|message| message.message_id == 20)
            .unwrap();
        let cross_reply = cross.reply.as_ref().unwrap();
        assert_eq!(cross_reply.sender.as_deref(), Some("Cross parent"));
        assert_eq!(cross_reply.text, "cross body");
        assert_eq!(cross_reply.chat_name.as_deref(), Some("Parent"));
        assert_eq!(cross_reply.topic_id, Some(3));
        assert!(!cross_reply.missing);

        let missing = messages
            .iter()
            .find(|message| message.message_id == 21)
            .unwrap();
        let missing_reply = missing.reply.as_ref().unwrap();
        assert_eq!(missing_reply.text, "preserved quote");
        assert!(missing_reply.missing);

        let story = messages
            .iter()
            .find(|message| message.message_id == 22)
            .unwrap();
        let story_reply = story.reply.as_ref().unwrap();
        assert_eq!(story_reply.story_id, Some(11));
        assert_eq!(story_reply.peer_kind.as_deref(), Some("user"));
        assert_eq!(story_reply.peer_id, Some(42));
        assert!(story_reply.missing);
    }

    #[test]
    fn viewer_pages_bound_memory_and_can_jump_to_an_exact_message() {
        let conn = test_connection();
        conn.execute("INSERT INTO chats VALUES ('paged', 'Paged chat')", [])
            .unwrap();
        for message_id in 1..=405 {
            conn.execute(
                "INSERT INTO messages(
                     message_id, chat_id, sender, timestamp_unix, timestamp, text
                 ) VALUES (?, 'paged', 'Me', ?, ?, ?)",
                rusqlite::params![
                    message_id,
                    message_id,
                    format!(
                        "1970-01-01T00:{:02}:{:02}Z",
                        message_id / 60,
                        message_id % 60
                    ),
                    format!("message {message_id}")
                ],
            )
            .unwrap();
        }

        let latest = load_chat_page(
            &conn,
            "paged",
            "Paged chat".to_string(),
            ChatPageRequest::Latest,
        )
        .unwrap();
        assert_eq!(latest.messages.len(), 400);
        assert_eq!(latest.messages.first().unwrap().message_id, 6);
        assert!(latest.has_older);
        assert!(!latest.has_newer);

        let around = load_chat_page(
            &conn,
            "paged",
            "Paged chat".to_string(),
            ChatPageRequest::Around { message_id: 203 },
        )
        .unwrap();
        assert_eq!(around.focus_message_id, Some(203));
        assert!(
            around
                .messages
                .iter()
                .any(|message| message.message_id == 203)
        );
        assert!(around.has_older);
        assert!(around.has_newer);
    }

    #[test]
    fn viewer_pages_reach_legacy_messages_without_timestamps() {
        let conn = test_connection();
        conn.execute("INSERT INTO chats VALUES ('legacy', 'Legacy chat')", [])
            .unwrap();
        for message_id in 1..=405 {
            conn.execute(
                "INSERT INTO messages(
                     message_id, chat_id, sender, timestamp_unix, timestamp, text
                 ) VALUES (?, 'legacy', 'Me', ?, ?, ?)",
                rusqlite::params![
                    message_id,
                    message_id,
                    format!("1970-01-01T00:00:{message_id:02}Z"),
                    format!("message {message_id}")
                ],
            )
            .unwrap();
        }
        for message_id in -3..=-1 {
            conn.execute(
                "INSERT INTO messages(message_id, chat_id, sender, text)
                 VALUES (?, 'legacy', 'Unknown', ?)",
                rusqlite::params![message_id, format!("legacy message {message_id}")],
            )
            .unwrap();
        }

        let latest = load_chat_page(
            &conn,
            "legacy",
            "Legacy chat".to_string(),
            ChatPageRequest::Latest,
        )
        .unwrap();
        assert_eq!(latest.messages.len(), 400);
        assert!(latest.has_older);

        let oldest = load_chat_page(
            &conn,
            "legacy",
            "Legacy chat".to_string(),
            ChatPageRequest::Before {
                timestamp_unix: latest.messages.first().unwrap().timestamp_unix,
                message_id: latest.messages.first().unwrap().message_id,
            },
        )
        .unwrap();
        assert_eq!(
            oldest
                .messages
                .iter()
                .map(|message| message.message_id)
                .collect::<Vec<_>>(),
            vec![-3, -2, -1, 1, 2, 3, 4, 5]
        );
        assert!(!oldest.has_older);
        assert!(oldest.has_newer);

        let around_untimestamped = load_chat_page(
            &conn,
            "legacy",
            "Legacy chat".to_string(),
            ChatPageRequest::Around { message_id: -2 },
        )
        .unwrap();
        assert_eq!(around_untimestamped.focus_message_id, Some(-2));
        assert!(
            around_untimestamped
                .messages
                .iter()
                .any(|message| message.message_id == -2)
        );
        assert!(!around_untimestamped.has_older);
        assert!(around_untimestamped.has_newer);
    }

    #[test]
    fn message_search_uses_full_text_prefixes_and_chat_scope() {
        let conn = test_connection();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(
                 text, media_path, content='messages', content_rowid='id'
             );
             INSERT INTO messages_fts(messages_fts) VALUES('rebuild');",
        )
        .unwrap();

        let (global, total) = search_messages(&conn, "missing rep", None, 20, 0).unwrap();
        assert_eq!(global.len(), 1);
        assert_eq!(total, 1);
        assert_eq!(global[0].chat_name, "Child");
        assert_eq!(global[0].message_id, 21);

        assert!(
            search_messages(&conn, "cross", Some("parent"), 20, 0)
                .unwrap()
                .0
                .iter()
                .any(|result| result.message_id == 7)
        );
        assert!(
            search_messages(&conn, "cross", Some("missing"), 20, 0)
                .unwrap()
                .0
                .is_empty()
        );

        let (first_page, reply_total) =
            search_messages(&conn, "reply", Some("child"), 1, 0).unwrap();
        let (second_page, repeated_total) =
            search_messages(&conn, "reply", Some("child"), 1, 1).unwrap();
        assert!(reply_total > 1);
        assert_eq!(reply_total, repeated_total);
        assert_eq!(first_page.len(), 1);
        assert_eq!(second_page.len(), 1);
        assert_ne!(first_page[0].message_id, second_page[0].message_id);

        let ordered_row_ids = search_message_rowids(&conn, "reply").unwrap();
        let materialized = load_search_results_by_rowids(&conn, &ordered_row_ids).unwrap();
        assert_eq!(materialized.len(), reply_total);
        assert!(
            materialized
                .windows(2)
                .all(|pair| pair[0].message_id > pair[1].message_id)
        );
    }

    #[test]
    fn outgoing_identity_follows_stable_id_into_linked_legacy_aliases() {
        let conn = test_connection();
        conn.execute_batch(
            "INSERT INTO chats VALUES ('api-copy', 'Peer'), ('legacy-copy', 'Peer');
             INSERT INTO telegram_backup_targets
                 VALUES ('peer-key', 'api-copy', 'user', 123);
             INSERT INTO telegram_backup_target_chats
                 VALUES ('peer-key', 'api-copy'), ('peer-key', 'legacy-copy');
             INSERT INTO messages(
                 message_id, chat_id, sender, sender_id, timestamp_unix, timestamp, text,
                 source_format
             ) VALUES
                 (100, 'api-copy', 'Me', '42', 100, '1970-01-01T00:01:40Z',
                  'same archived message', 'telegram_api'),
                 (101, 'api-copy', 'Me', '42', 101, '1970-01-01T00:01:41Z',
                  'second archived message', 'telegram_api');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text
             ) VALUES
                 (100, 'legacy-copy', 'historical alias', 100,
                  '1970-01-01T00:01:40Z', 'same archived message'),
                 (101, 'legacy-copy', 'historical alias', 101,
                  '1970-01-01T00:01:41Z', 'second archived message'),
                 (102, 'legacy-copy', 'historical alias', 102,
                  '1970-01-01T00:01:42Z', 'legacy-only message');",
        )
        .unwrap();

        let messages = load_chat_messages(&conn, "legacy-copy", 10).unwrap();
        assert_eq!(messages.len(), 3);
        assert!(messages.iter().all(|message| message.is_outgoing));
    }

    #[test]
    fn outgoing_identity_propagates_across_a_chain_of_renamed_backups() {
        let conn = test_connection();
        conn.execute_batch(
            "INSERT INTO chats VALUES
                 ('chain-api', 'Peer'), ('chain-middle', 'Peer'), ('chain-old', 'Peer');
             INSERT INTO telegram_backup_targets
                 VALUES ('chain-key', 'chain-api', 'user', 456);
             INSERT INTO telegram_backup_target_chats VALUES
                 ('chain-key', 'chain-api'),
                 ('chain-key', 'chain-middle'),
                 ('chain-key', 'chain-old');
             INSERT INTO messages(
                 message_id, chat_id, sender, sender_id, timestamp_unix, timestamp, text,
                 source_format
             ) VALUES
                 (1000, 'chain-api', 'Me', '42', 1000, '1970-01-01T00:16:40Z',
                  'current self one', 'telegram_api'),
                 (1001, 'chain-api', 'Me', '42', 1001, '1970-01-01T00:16:41Z',
                  'current self two', 'telegram_api');
             INSERT INTO messages(
                 message_id, chat_id, sender, timestamp_unix, timestamp, text
             ) VALUES
                 (1000, 'chain-middle', 'new self name', 1000,
                  '1970-01-01T00:16:40Z', 'current self one'),
                 (1001, 'chain-middle', 'new self name', 1001,
                  '1970-01-01T00:16:41Z', 'current self two'),
                 (10, 'chain-middle', 'new self name', 100,
                  '1970-01-01T00:01:40Z', 'historical self one'),
                 (11, 'chain-middle', 'new self name', 101,
                  '1970-01-01T00:01:41Z', 'historical self two'),
                 (12, 'chain-middle', 'Peer', 200,
                  '1970-01-01T00:03:20Z', 'historical peer one'),
                 (13, 'chain-middle', 'Peer', 201,
                  '1970-01-01T00:03:21Z', 'historical peer two'),
                 (500, 'chain-old', 'old self name', 100,
                  '1970-01-01T00:01:40Z', 'historical self one'),
                 (501, 'chain-old', 'old self name', 101,
                  '1970-01-01T00:01:41Z', 'historical self two'),
                 (502, 'chain-old', 'old self name', 102,
                  '1970-01-01T00:01:42Z', 'old-only self message'),
                 (600, 'chain-old', 'Peer', 200,
                  '1970-01-01T00:03:20Z', 'historical peer one'),
                 (601, 'chain-old', 'Peer', 201,
                  '1970-01-01T00:03:21Z', 'historical peer two');",
        )
        .unwrap();

        let messages = load_chat_messages(&conn, "chain-old", 20).unwrap();
        assert_eq!(messages.len(), 5);
        assert!(
            messages
                .iter()
                .filter(|message| message.sender == "old self name")
                .all(|message| message.is_outgoing)
        );
        assert!(
            messages
                .iter()
                .filter(|message| message.sender == "Peer")
                .all(|message| !message.is_outgoing)
        );
    }

    #[test]
    #[ignore = "requires TGBACKMAN_DB and TGBACKMAN_PERF_CHAT_ID"]
    fn live_database_chat_page_performance() {
        let db_path = std::env::var("TGBACKMAN_DB").expect("set TGBACKMAN_DB");
        let chat_id = std::env::var("TGBACKMAN_PERF_CHAT_ID").expect("set TGBACKMAN_PERF_CHAT_ID");
        let message_id = std::env::var("TGBACKMAN_PERF_MESSAGE_ID")
            .ok()
            .and_then(|value| value.parse::<i64>().ok());
        let request = message_id.map_or(ChatPageRequest::Latest, |message_id| {
            ChatPageRequest::Around { message_id }
        });
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let started = std::time::Instant::now();
        let page =
            load_chat_page(&conn, &chat_id, "Performance chat".to_string(), request).unwrap();
        println!(
            "loaded {} messages from a {}-message chat in {:?}",
            page.messages.len(),
            page.total_messages,
            started.elapsed()
        );
        let cached_aliases = page.self_sender_aliases.clone();
        let started = std::time::Instant::now();
        let cached_page = load_chat_page_with_aliases(
            &conn,
            &chat_id,
            "Performance chat".to_string(),
            request,
            Some(&cached_aliases),
        )
        .unwrap();
        println!(
            "reloaded {} messages with cached identity in {:?}",
            cached_page.messages.len(),
            started.elapsed()
        );
    }

    #[test]
    #[ignore = "requires TGBACKMAN_DB and optionally TGBACKMAN_PERF_SEARCH_QUERY"]
    fn live_database_broad_search_performance() {
        let db_path = std::env::var("TGBACKMAN_DB").expect("set TGBACKMAN_DB");
        let query =
            std::env::var("TGBACKMAN_PERF_SEARCH_QUERY").unwrap_or_else(|_| "the".to_string());
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let started = std::time::Instant::now();
        let row_ids = search_message_rowids(&conn, &query).unwrap();
        let total = row_ids.len();
        println!(
            "indexed {total} matches for {query:?} in {:?}",
            started.elapsed()
        );

        let started = std::time::Instant::now();
        let first = load_search_results_by_rowids(&conn, &row_ids[..total.min(250)]).unwrap();
        println!(
            "materialized first {} matches in {:?}",
            first.len(),
            started.elapsed()
        );
        assert_eq!(first.len(), total.min(250));

        if total > 500 {
            let offset = (total / 2 / 250) * 250;
            let started = std::time::Instant::now();
            let end = (offset + 250).min(total);
            let middle = load_search_results_by_rowids(&conn, &row_ids[offset..end]).unwrap();
            println!(
                "materialized {} matches at virtual offset {offset} in {:?}",
                middle.len(),
                started.elapsed()
            );
            assert!(!middle.is_empty());
        }
    }
}
