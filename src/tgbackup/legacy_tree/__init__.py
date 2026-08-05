"""Tools that operate on legacy Telegram export directories and their media.

These modules are filesystem-oriented. They do not represent or update the
canonical Telegram backup database; database-aware reconciliation lives in
``tgbackup.database``. ``inspect_tree`` may read an unofficial SQLite file
embedded in a legacy export, but never opens the canonical database.
"""
