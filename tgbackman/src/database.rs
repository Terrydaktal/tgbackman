//! Database inventory mutations and target identity helpers.

use chrono::Utc;
use std::collections::{HashMap, HashSet};

use crate::cache::{cache_is_fresh, get_clusters_cache_path};
use crate::inventory::UnionFind;
use crate::model::{BackupInfo, ChatGroup};

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
    Ok(())
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
         UNION
         SELECT targets.chat_id, targets.peer_kind, targets.peer_id
         FROM telegram_backup_targets AS targets",
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

pub(crate) fn run_inventory(
    conn: &rusqlite::Connection,
    db_path: &str,
) -> Result<Vec<ChatGroup>, rusqlite::Error> {
    let start_total = std::time::Instant::now();

    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0;",
        [],
    );
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER;", []);
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN last_backup_run_unix INTEGER;",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN last_backup_run_status TEXT;",
        [],
    );
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN last_backup_source TEXT;", []);
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN last_backup_confidence TEXT;",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN last_backup_evidence TEXT;",
        [],
    );
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN min_msg_id INTEGER;", []);
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN max_msg_id INTEGER;", []);
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN msg_count INTEGER;", []);
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN min_timestamp TEXT;", []);
    let _ = conn.execute("ALTER TABLE chats ADD COLUMN max_timestamp TEXT;", []);
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN min_timestamp_unix INTEGER;",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE chats ADD COLUMN max_timestamp_unix INTEGER;",
        [],
    );

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
    if cache_is_fresh(&clusters_cache_path, db_path) {
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
    } else if std::path::Path::new(&clusters_cache_path).exists() {
        println!("Ignoring stale chat-cluster cache: {}", clusters_cache_path);
    }

    if !loaded_cache {
        // 1. FUZZY ALIAS LINKING via oldest signatures
        let mut exact_signatures: HashMap<(i64, String), Vec<String>> = HashMap::new();
        let mut stmt_msgs = conn.prepare(
            "SELECT timestamp_unix, text FROM messages WHERE chat_id = ? AND text != '' AND timestamp_unix IS NOT NULL ORDER BY timestamp_unix ASC LIMIT 50"
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
            "SELECT COUNT(*) FROM messages a JOIN messages b ON a.timestamp_unix = b.timestamp_unix WHERE a.chat_id = ? AND b.chat_id = ? AND a.text = b.text AND a.text != '' AND length(a.text) >= 6"
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

    // Query fresh stats for every chat. The cached columns are denormalized
    // display data and can be stale after an API exporter writes directly to
    // `messages`; trusting them hides newly imported messages and dates.
    let _ = conn.execute("BEGIN TRANSACTION;", []);

    let mut stats_map = HashMap::new();
    let mut stmt_update_stats = conn.prepare(
        "UPDATE chats SET min_msg_id = ?, max_msg_id = ?, msg_count = ?, min_timestamp = ?, max_timestamp = ?, min_timestamp_unix = ?, max_timestamp_unix = ? WHERE chat_id = ?"
    )?;
    let mut stmt_calc_stats = conn.prepare(
        "SELECT MIN(message_id), MAX(message_id), COUNT(*), MIN(timestamp), MAX(timestamp), MIN(timestamp_unix), MAX(timestamp_unix) FROM messages WHERE chat_id = ?"
    )?;

    for c in &chats {
        let mut rows_calc = stmt_calc_stats.query(rusqlite::params![c.chat_id])?;
        if let Some(row) = rows_calc.next()? {
            let min_id: Option<i64> = row.get(0)?;
            let max_id: Option<i64> = row.get(1)?;
            let count: i64 = row.get(2)?;
            let min_ts: Option<String> = row.get(3)?;
            let max_ts: Option<String> = row.get(4)?;
            let min_unix: Option<i64> = row.get(5)?;
            let max_unix: Option<i64> = row.get(6)?;

            let stats_changed = c.min_msg_id != min_id
                || c.max_msg_id != max_id
                || c.msg_count != Some(count)
                || c.min_timestamp != min_ts
                || c.max_timestamp != max_ts
                || c.min_timestamp_unix != min_unix
                || c.max_timestamp_unix != max_unix;
            if stats_changed {
                stmt_update_stats.execute(rusqlite::params![
                    min_id, max_id, count, &min_ts, &max_ts, min_unix, max_unix, c.chat_id
                ])?;
            }
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
    if let Ok(file) = std::fs::File::create(&clusters_cache_path) {
        let _ = serde_json::to_writer_pretty(file, &uf.parent);
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
