"""
Auto-update checker for yt-dlp.
Checks PyPI for latest version and offers pip-based update.
"""
import json
import subprocess
import sys
import urllib.request
from typing import Optional, Tuple


def get_current_version() -> Optional[str]:
    """Get the currently installed yt-dlp version."""
    try:
        import yt_dlp
        # Try multiple version attribute names (yt-dlp changed this over versions)
        for attr in ('__version__', 'version', 'VERSION'):
            ver = getattr(yt_dlp.version, attr, None)
            if ver:
                return ver
        # Fallback: try parsing from yt_dlp module level
        import yt_dlp
        return getattr(yt_dlp, '__version__', None)
    except (ImportError, AttributeError):
        return None


def get_latest_version() -> Optional[str]:
    """Fetch the latest yt-dlp version from PyPI."""
    try:
        url = "https://pypi.org/pypi/yt-dlp/json"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
            return data["info"]["version"]
    except Exception:
        return None


def is_outdated() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if installed yt-dlp is outdated.
    Returns (is_outdated, current_version, latest_version).
    """
    current = get_current_version()
    latest = get_latest_version()
    if current is None or latest is None:
        return False, current, latest
    is_outdated = current != latest
    return is_outdated, current, latest


def update_yt_dlp(progress_callback=None) -> Tuple[bool, str]:
    """
    Update yt-dlp to the latest version via pip.
    Returns (success, message).
    """
    try:
        args = [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"]
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode == 0:
            return True, f"Successfully updated yt-dlp"
        else:
            return False, f"Update failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Update timed out after 5 minutes"
    except Exception as e:
        return False, f"Update error: {str(e)}"
