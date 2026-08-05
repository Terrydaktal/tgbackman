"""Tools that create, inspect, reconcile, or query the canonical SQLite DB.

Modules are intentionally not imported eagerly so ``python -m`` execution of
one database tool does not load a second copy of that module first.
"""
