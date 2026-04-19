"""
Download state persistence for resume/crash recovery.
Tracks completed and failed videos per playlist using per-playlist JSON files.

State location: ~/.config/youtube-music-downloader/state/
"""
import json
import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import get_config_dir

# Use an RLock because several public helpers acquire the state lock and then
# call internal functions that may need the same lock again during lazy
# directory initialization on first run.
_state_lock = threading.RLock()

# Lazily computed state directory (avoids importing CONFIG_DIR at module level)
_state_dir: Optional[Path] = None


def _get_state_dir() -> Path:
    """Get the state directory, lazily initialized."""
    global _state_dir
    if _state_dir is None:
        with _state_lock:
            if _state_dir is None:
                _state_dir = get_config_dir() / "state"
    return _state_dir


def ensure_state_dir() -> Path:
    """Create state directory if it doesn't exist."""
    path = _get_state_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _playlist_hash(url: str) -> str:
    """Generate a short MD5 hash for a playlist URL to use as filename key."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _get_state_path(url: str) -> Path:
    """Get the JSON state file path for a playlist URL."""
    ensure_state_dir()
    return _get_state_dir() / f"{_playlist_hash(url)}.json"


def _empty_state(url: str) -> Dict:
    """Return an empty state dict structure for a playlist."""
    return {
        "playlist_url": url,
        "playlist_name": "",
        "last_download": None,
        "completed_videos": [],
        "failed_videos": [],
    }


def load_state(url: str) -> Dict:
    """Load state for a playlist, returns empty state if none exists."""
    path = _get_state_path(url)
    if not path.exists():
        return _empty_state(url)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _empty_state(url)


def _atomic_save(path: Path, data: Dict) -> None:
    """Atomically save JSON data using temp file + rename."""
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def save_state(state: Dict) -> None:
    """Save state for a playlist atomically."""
    path = _get_state_path(state["playlist_url"])
    with _state_lock:
        _atomic_save(path, state)


def mark_video_completed(url: str, video_id: str, title: str) -> None:
    """Mark a video as completed in the playlist's state."""
    with _state_lock:
        state = load_state(url)
        state["last_download"] = datetime.now().isoformat()
        # Avoid duplicates
        if not any(v["id"] == video_id for v in state["completed_videos"]):
            state["completed_videos"].append({
                "id": video_id,
                "title": title,
                "downloaded_at": datetime.now().isoformat(),
            })
        _atomic_save(_get_state_path(url), state)


def mark_video_failed(url: str, video_id: str, title: str, error: str) -> None:
    """Mark a video as failed in the playlist's state."""
    with _state_lock:
        state = load_state(url)
        # Remove from completed if it was there (retry scenario)
        state["completed_videos"] = [v for v in state["completed_videos"] if v["id"] != video_id]
        # Add/update to failed
        existing = next((v for v in state["failed_videos"] if v["id"] == video_id), None)
        if existing:
            existing["last_attempt"] = datetime.now().isoformat()
            existing["error"] = error
        else:
            state["failed_videos"].append({
                "id": video_id,
                "title": title,
                "error": error,
                "first_attempted_at": datetime.now().isoformat(),
                "last_attempt": datetime.now().isoformat(),
            })
        _atomic_save(_get_state_path(url), state)


def is_video_completed(url: str, video_id: str) -> bool:
    """Check if a video was already completed in a previous run."""
    state = load_state(url)
    return any(v["id"] == video_id for v in state["completed_videos"])


def get_completed_video_ids(url: str) -> List[str]:
    """Get list of completed video IDs for a playlist."""
    state = load_state(url)
    return [v["id"] for v in state["completed_videos"]]


def set_playlist_name(url: str, name: str) -> None:
    """Store the playlist name in its state file."""
    with _state_lock:
        state = load_state(url)
        state["playlist_name"] = name
        _atomic_save(_get_state_path(url), state)


def clear_state(url: str) -> None:
    """Clear all state for a playlist (fresh start)."""
    path = _get_state_path(url)
    if path.exists():
        path.unlink()


def get_all_states() -> List[Dict]:
    """Return all playlist states (for UI display of watch list)."""
    ensure_state_dir()
    states = []
    for state_file in _get_state_dir().glob("*.json"):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                states.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return states
