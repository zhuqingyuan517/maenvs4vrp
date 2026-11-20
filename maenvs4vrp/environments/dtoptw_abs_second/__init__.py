"""DTOPTW variant with absolute (non-relative) observation features.

This package is intentionally thin: it reuses the original dtoptw dynamics
and only changes the observation builder (and thus the learning interface).
Compared with dtoptw_abs, this variant also exposes other agents observations.
"""
