import os
import sqlite3

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def count_missing_messages(cursor, chat_a_id, chat_b_id, start_unix, end_unix):
    from collections import defaultdict
    if start_unix is None or end_unix is None:
        return 0

    def clean_text_for_match(text):
        if not text:
            return ""

        # Shortcut for pure media placeholders (e.g. [photo], [voice_message])
        trimmed = text.strip()
        if trimmed.startswith('[') and trimmed.endswith(']'):
            return trimmed

        import re
        # 1. Strip legacy forwarded headers like {{FWD: ...}}
        text = re.sub(r"^\{\{FWD:.*?\}\}\s*", "", text, flags=re.DOTALL)
        # 2. Strip legacy double bracket prefixes/blocks like [[Webpage]] or [[Voice Message, ...]]
        text = re.sub(r"^\[\[.*?\]\]\s*", "", text)

        t = text.lower()

        # 1. Strip URLs first
        t = re.sub(r'(?:https?://|tel:|mailto:|tg:)[^\s"\'“”‘’<>]+', " ", t)
        # 2. Strip Domains
        t = re.sub(r'\b[a-z0-9\-]+\.[a-z]{2,24}(?:\.[a-z]{2,24})*\b', " ", t)

        # Stage 1: Split by whitespace, and deduplicate adjacent words using their boundary-stripped normalized form
        words = t.split()
        stage1_words = []
        for w in words:
            if not stage1_words:
                stage1_words.append(w)
            else:
                # Strip leading/trailing non-alphanumeric characters to form a comparison key
                w_norm = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', w)
                last_norm = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', stage1_words[-1])
                if w_norm and w_norm == last_norm:
                    # Skip duplicate
                    continue
                else:
                    stage1_words.append(w)

        t_stage1 = " ".join(stage1_words)

        # Stage 2: Keep only alphanumeric characters and spaces
        clean_chars = []
        for c in t_stage1:
            if c.isalnum() or c.isspace():
                clean_chars.append(c)
            else:
                clean_chars.append(' ')
        t_stage2 = "".join(clean_chars)

        # Stage 3: Split, final adjacent deduplication, and join
        final_words = t_stage2.split()
        deduped = []
        for w in final_words:
            if not deduped:
                deduped.append(w)
            elif deduped[-1] != w:
                deduped.append(w)
        return " ".join(deduped).strip()

    # 1. Fetch all messages from Chat A in the range [start_unix, end_unix]
    cursor.execute("""
        SELECT timestamp_unix, text, media_path
        FROM messages
        WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ?
    """, (chat_a_id, start_unix, end_unix))
    messages_a = cursor.fetchall()
    if not messages_a:
        return 0

    # 2. Fetch all messages from Chat B in the expanded range [start_unix - 3605, end_unix + 3605]
    # to account for potential 1-hour BST/DST timezone shifts in either direction.
    cursor.execute("""
        SELECT timestamp_unix, text, media_path
        FROM messages
        WHERE chat_id = ? AND timestamp_unix BETWEEN ? AND ?
    """, (chat_b_id, start_unix - 3605, end_unix + 3605))
    messages_b = cursor.fetchall()

    # 3. Build a fast lookup structure for Chat B messages.
    b_by_ts = defaultdict(list)
    for ts, txt, media in messages_b:
        if ts is not None:
            b_by_ts[ts].append((clean_text_for_match(txt), media or ""))

    # 4. Check each message in Chat A
    missing_count = 0
    for ts, txt, media in messages_a:
        if ts is None:
            continue
        txt_clean = clean_text_for_match(txt)
        media = media or ""
        # Look in interval [ts - 5, ts + 5], and check for common DST/BST timezone shifts (0, +1 hr, -1 hr)
        found = False
        for offset in (0, 3600, -3600):
            target_ts = ts + offset
            for candidate_ts in range(target_ts - 5, target_ts + 6):
                if candidate_ts in b_by_ts:
                    # Check for match
                    candidates = b_by_ts[candidate_ts]
                    for index, (b_txt_clean, b_media) in enumerate(candidates):
                        # Match cases:
                        # 1. Cleaned texts match exactly
                        if txt_clean == b_txt_clean:
                            candidates.pop(index)
                            found = True
                            break
                        # 2. Symmetric media placeholder match (e.g. one has empty text, other has '[photo]')
                        elif (txt_clean == "" and b_txt_clean.startswith("[") and b_txt_clean.endswith("]")) or \
                             (b_txt_clean == "" and txt_clean.startswith("[") and txt_clean.endswith("]")):
                            candidates.pop(index)
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            missing_count += 1

    return missing_count

if __name__ == "__main__":
    db_path = os.environ.get("TGBACKMAN_DB")
    if not db_path:
        import getpass
        user = os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()
        volume = os.environ.get("TGBACKMAN_REMOVABLE_VOLUME", "").strip("/")
        if volume:
            candidates = [os.path.join("/media", user, volume, "sqlitedb", "telegram_backup.db")]
        else:
            media_root = os.path.join("/media", user)
            candidates = sorted(
                os.path.join(media_root, entry, "sqlitedb", "telegram_backup.db")
                for entry in os.listdir(media_root)
                if os.path.isfile(os.path.join(media_root, entry, "sqlitedb", "telegram_backup.db"))
            ) if os.path.isdir(media_root) else []
        db_path = next((candidate for candidate in candidates if os.path.exists(candidate)), "telegram_backup.db")

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Optimize SQLite performance using memory cache configurations
    cursor.execute("PRAGMA cache_size = -1048576;")  # 1 GB memory cache
    cursor.execute("PRAGMA temp_store = MEMORY;")

    # Get all chats
    cursor.execute("SELECT chat_id, chat_name, backup_path FROM chats")
    chats = cursor.fetchall()
    print(f"Loaded {len(chats)} chats from database.")

    uf = UnionFind()

    # 1. FUZZY ALIAS LINKING via oldest footprints (catches renamed aliases like ChatName / AlternateName)
    print("Harvesting oldest message signatures for fuzzy alias linking...")
    exact_signatures = {} # (timestamp_unix, text) -> list of chat_ids

    for cid, name, path in chats:
        cursor.execute("""
            SELECT timestamp_unix, text
            FROM messages
            WHERE chat_id = ?
              AND COALESCE(is_deleted, 0)=0
              AND text != ''
              AND timestamp_unix IS NOT NULL
            ORDER BY timestamp_unix ASC
            LIMIT 50
        """, (cid,))
        msgs = cursor.fetchall()

        for ts, text in msgs:
            clean_text = text.strip()
            if len(clean_text) >= 6:
                exact_sig = (ts, clean_text)
                exact_signatures.setdefault(exact_sig, []).append(cid)

    exact_shared_counts = {}
    for sig, cids in exact_signatures.items():
        if len(cids) < 2:
            continue
        unique_cids = list(set(cids))
        for i in range(len(unique_cids)):
            for j in range(i + 1, len(unique_cids)):
                c1, c2 = unique_cids[i], unique_cids[j]
                pair = tuple(sorted([c1, c2]))
                exact_shared_counts[pair] = exact_shared_counts.get(pair, 0) + 1

    linked_via_sigs = 0
    for (c1, c2), count in exact_shared_counts.items():
        if count >= 3:
            if uf.find(c1) != uf.find(c2):
                uf.union(c1, c2)
                linked_via_sigs += 1

    # 2. SAME-NAME DUPLICATES LINKING via indexed history query (handles BST timezone shifts and disparate timeframes)
    print("Verifying same-name backups via sub-millisecond B-tree index checks...")
    chats_by_norm_name = {}
    for cid, name, path in chats:
        if name:
            norm_name = name.strip().lower()
            if norm_name not in ("", "deleted account", "telegram", "group", "unknown"):
                chats_by_norm_name.setdefault(norm_name, []).append(cid)

    linked_via_same_name = 0
    for norm_name, cids in chats_by_norm_name.items():
        if len(cids) < 2:
            continue
        # Perform fast pairwise index check for all chats that share the exact same name
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                c1, c2 = cids[i], cids[j]
                if uf.find(c1) == uf.find(c2):
                    continue # Already linked

                # Perform rapid B-tree join to check for shared messages anywhere in history
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM messages a
                    JOIN messages b ON a.timestamp_unix = b.timestamp_unix
                    WHERE COALESCE(a.is_deleted, 0)=0
                      AND COALESCE(b.is_deleted, 0)=0
                      AND a.chat_id = ?
                      AND b.chat_id = ?
                      AND a.text = b.text
                      AND a.text != ''
                      AND length(a.text) >= 6
                """, (c1, c2))
                shared_count = cursor.fetchone()[0]

                if shared_count >= 3:
                    uf.union(c1, c2)
                    linked_via_same_name += 1

    print(f"Linking completed. Linked {linked_via_sigs} aliases via footprints, and {linked_via_same_name} same-name duplicates via indexed history.")

    # Group backups by their logical Union-Find root chat ID
    logical_groups = {}
    for cid, name, path in chats:
        if name:
            norm_name = name.strip().lower()
            # Skip generic placeholder chats from grouping, UNLESS they have been explicitly linked via message signatures!
            if norm_name in ("deleted account", "telegram", "group", "unknown") and uf.find(cid) == cid:
                continue

        root_cid = uf.find(cid)
        logical_groups.setdefault(root_cid, []).append((cid, name, path))

    print("Gathering backup stats...")
    reports = []

    for root_cid, entries in logical_groups.items():
        chat_stats = []
        names_seen = set()
        import datetime

        def format_unix_to_ts(unix_ts):
            if not unix_ts:
                return "Unknown"
            dt = datetime.datetime.fromtimestamp(unix_ts, datetime.timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        for cid, name, path in entries:
            cursor.execute("""
                SELECT MIN(message_id), MAX(message_id), COUNT(*),
                       MIN(timestamp), MAX(timestamp),
                       MIN(timestamp_unix), MAX(timestamp_unix)
                FROM messages
                WHERE chat_id = ?
                  AND COALESCE(is_deleted, 0)=0
            """, (cid,))
            min_id, max_id, count, min_ts, max_ts, min_unix, max_unix = cursor.fetchone()

            def format_ts(ts_str):
                if not ts_str:
                    return "Unknown"
                return ts_str.replace("T", " ").replace("Z", "")

            if min_id is not None and count > 0:
                if name:
                    names_seen.add(name.strip())
                chat_stats.append({
                    "chat_id": cid,
                    "name": name or "Unknown",
                    "path": path,
                    "min_id": min_id,
                    "max_id": max_id,
                    "count": count,
                    "min_ts": format_ts(min_ts),
                    "max_ts": format_ts(max_ts),
                    "min_unix": min_unix,
                    "max_unix": max_unix
                })

        if not chat_stats:
            continue

        # Sort backups of this chat by message count in ascending order
        chat_stats.sort(key=lambda x: x["count"])

        # Combined display name representing all aliases in this logical group
        display_name = " / ".join(sorted(list(names_seen))) if names_seen else "Unknown Chat"

        # Keep track of the maximum message count among all backups for this chat
        max_count = chat_stats[-1]["count"]

        reports.append({
            "name": display_name,
            "max_count": max_count,
            "backups": chat_stats
        })

    # Sort the entire list of reports by their maximum backup message count in ascending order
    reports.sort(key=lambda r: r["max_count"])

    print("Running backup inventory and containment analyses...")
    overlaps_found = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total_backups_listed = 0

    for r in reports:
        backups = r["backups"]
        name = r["name"]
        total_backups_listed += len(backups)

        print(f"\n==========================================")
        print(f"Chat Name: {name}")
        print(f"==========================================")

        # Print the backups in ascending order of message count (largest at the bottom)
        for idx, b in enumerate(backups):
            letter = letters[idx] if idx < len(letters) else f"B{idx+1}"
            print(f"  [Backup {letter}] Path: {b['path']}")
            print(f"             Message IDs: {b['min_id']} to {b['max_id']} (Total: {b['count']} msgs)")
            print(f"             Date Range:  {b['min_ts']} to {b['max_ts']}")

        # Only perform pairwise containment and overlap analysis if there are multiple backups
        if len(backups) >= 2:
            print(f"  ----------------------------------------")
            for i in range(len(backups)):
                for j in range(i + 1, len(backups)):
                    a = backups[i]
                    b = backups[j]

                    letter_a = letters[i] if i < len(letters) else f"B{i+1}"
                    letter_b = letters[j] if j < len(letters) else f"B{j+1}"

                    # Check for chronological span containment
                    # We use a 1-day (86400s) buffer to accommodate minor timezone or export generation shifts
                    a_contains_b = False
                    b_contains_a = False

                    if a["min_unix"] is not None and a["max_unix"] is not None and b["min_unix"] is not None and b["max_unix"] is not None:
                        if a["min_unix"] <= b["min_unix"] + 86400 and a["max_unix"] >= b["max_unix"] - 86400:
                            a_contains_b = True
                        elif b["min_unix"] <= a["min_unix"] + 86400 and b["max_unix"] >= a["max_unix"] - 86400:
                            b_contains_a = True

                    if a["min_unix"] is not None and a["max_unix"] is not None and b["min_unix"] is not None and b["max_unix"] is not None:
                        overlap_start = max(a["min_unix"], b["min_unix"])
                        overlap_end = min(a["max_unix"], b["max_unix"])

                        if overlap_start <= overlap_end:
                            overlap_days = (overlap_end - overlap_start) / 86400.0
                            overlaps_found += 1

                            if b_contains_a:
                                print(f"  -> Backup {letter_b} fully contains the chronological span of Backup {letter_a}!")
                            elif a_contains_b:
                                print(f"  -> Backup {letter_a} fully contains the chronological span of Backup {letter_b}!")
                            else:
                                print(f"  -> Backup {letter_a} and Backup {letter_b} overlap chronologically by {overlap_days:.1f} days!")

                            print(f"     Overlap span: {format_unix_to_ts(overlap_start)} to {format_unix_to_ts(overlap_end)}")

                            # Count missing in both directions within the overlap region
                            missing_a_in_b = count_missing_messages(cursor, a["chat_id"], b["chat_id"], overlap_start, overlap_end)
                            missing_b_in_a = count_missing_messages(cursor, b["chat_id"], a["chat_id"], overlap_start, overlap_end)

                            if missing_a_in_b == 0 and missing_b_in_a == 0:
                                print(f"     SUCCESS: Perfect alignment! 0 missing messages in the overlapping region for both backups.")
                            else:
                                if missing_a_in_b > 0:
                                    print(f"     WARNING: Backup {letter_b} is missing {missing_a_in_b} individual messages that exist in Backup {letter_a}'s range!")
                                if missing_b_in_a > 0:
                                    print(f"     WARNING: Backup {letter_a} is missing {missing_b_in_a} individual messages that exist in Backup {letter_b}'s range!")

    conn.close()
    print(f"\nInventory Scan completed.")
    print(f"  - Total unique chat names listed: {len(reports)}")
    print(f"  - Total individual backup directories mapped: {total_backups_listed}")
    print(f"  - Total containment/overlap cases identified: {overlaps_found}")
