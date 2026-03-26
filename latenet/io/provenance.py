"""Provenance metadata for generated rows.

Stamps every batch with the git commit hash and timestamp so rows
can be traced back to the exact code version that produced them.
"""

from __future__ import annotations

import subprocess
import time


def get_git_hash() -> str:
    """Return the full SHA-1 of the current HEAD commit.

    Returns 'unknown' if not in a git repo or git is unavailable.
    Uses the short hash (first 8 chars). Commits must never be squashed
    on merge so hashes remain resolvable in git history.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def get_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
