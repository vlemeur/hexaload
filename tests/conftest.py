"""Shared fixtures for the fully offline test suite.

Fixtures are built in-memory or written to tmp_path; the suite never touches
the network. Filled as modules land — metrics first, then walk-forward.
"""
