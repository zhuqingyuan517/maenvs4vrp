"""DTOPTW full observation variant (global action space + rich features).

This environment reuses the original `dtoptw` dynamics, but exposes observations that
are compatible with pointer-network style policies over the full node set (N), i.e.
actions are global node indices (not a local K-subset).
"""

