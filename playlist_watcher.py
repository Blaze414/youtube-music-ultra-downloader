"""
Playlist watching mode: detect new tracks added since last download.
Store watched playlist URLs and check for new content.
"""
import logging
from typing import Dict, List, Optional, Tuple
import yt_dlp

logger = logging.getLogger(__name__)


def extract_playlist_videos(url: str) -> Tuple[str, List[Dict]]:
    """
    Extract current video list from a playlist URL.
    Returns (playlist_title, entries_list).
    Raises yt_dlp exceptions on failure (caller should handle them).
    """
    opts = {"quiet": True, "extract_flat": True, "socket_timeout": 30}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title", info.get("playlist_title", "Unknown"))
    entries = [
        {"id": e.get("id"), "title": e.get("title") or "Unknown"}
        for e in (info.get("entries") or [])
        if e and e.get("id")
    ]
    return title, entries


def get_new_videos(playlist_url: str, already_downloaded_ids: List[str]) -> List[Dict]:
    """
    Get videos from a playlist that are not in the already-downloaded set.
    Returns list of new entries.
    """
    _, all_entries = extract_playlist_videos(playlist_url)
    downloaded_set = set(already_downloaded_ids)
    return [e for e in all_entries if e.get("id") not in downloaded_set]


def add_watched_playlist(url: str, name: str) -> None:
    """Add a playlist to the watch list in config (idempotent)."""
    from config import get_config_value, set_config_value
    watched = get_config_value("watched_playlists", [])
    if not any(p.get("url") == url for p in watched):
        watched.append({"url": url, "name": name, "last_check": None})
        set_config_value("watched_playlists", watched)


def remove_watched_playlist(url: str) -> None:
    """Remove a playlist from the watch list."""
    from config import get_config_value, set_config_value
    watched = get_config_value("watched_playlists", [])
    watched = [p for p in watched if p.get("url") != url]
    set_config_value("watched_playlists", watched)


def get_watched_playlists() -> List[Dict]:
    """Get all watched playlists from config."""
    from config import get_config_value
    return get_config_value("watched_playlists", [])


def check_watched_playlist(url: str, already_downloaded_ids: List[str]) -> Tuple[str, List[Dict]]:
    """
    Check a single watched playlist for new videos.
    Returns (playlist_name, new_entries).
    """
    name, all_entries = extract_playlist_videos(url)
    downloaded_set = set(already_downloaded_ids)
    new_entries = [e for e in all_entries if e.get("id") not in downloaded_set]
    return name, new_entries


def check_all_watched(per_playlist_downloaded_ids_func=None) -> Dict[str, Tuple[str, List[Dict]]]:
    """
    Check all watched playlists for new content.
    per_playlist_downloaded_ids_func: optional callable(url) -> List[str] of completed IDs.
    Returns dict of {url: (playlist_name, new_entries)}.
    Errors for individual playlists are logged and skipped; partial results are returned.
    """
    results = {}
    for playlist in get_watched_playlists():
        url = playlist.get("url", "")
        if not url:
            continue

        try:
            if per_playlist_downloaded_ids_func:
                downloaded_ids = per_playlist_downloaded_ids_func(url)
            else:
                from state_manager import get_completed_video_ids
                downloaded_ids = get_completed_video_ids(url)
        except Exception as e:
            logger.warning("Failed to get downloaded IDs for %s: %s", url, e)
            continue

        try:
            name, new_entries = check_watched_playlist(url, downloaded_ids)
            if new_entries:
                results[url] = (name, new_entries)
        except Exception as e:
            logger.warning("Failed to check playlist %s: %s", url, e)
            continue

    return results