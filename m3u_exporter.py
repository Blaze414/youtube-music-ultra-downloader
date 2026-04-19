"""
M3U playlist export for downloaded playlists.
Generates Extended M3U files with track metadata.
"""
import os
from pathlib import Path
from typing import List, Optional

# Import shared duration helper from library_db to avoid duplication
try:
    from library_db import get_mp3_duration
except ImportError:
    # Fallback if library_db is not available
    def get_mp3_duration(mp3_path: Path) -> int:
        """Get duration in seconds from MP3 file metadata using mutagen."""
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(mp3_path))
            return int(audio.info.length)
        except Exception:
            return -1


def _escape_m3u_title(title: str) -> str:
    """Escape a string for use in an M3U #EXTINF title field."""
    # Per M3U spec, commas and backslashes must be escaped
    return title.replace("\\", "\\\\").replace(",", "\\,")


def generate_m3u_playlist(folder_path: Path, playlist_name: str) -> Optional[Path]:
    """
    Generate an M3U playlist file in the given folder.
    Returns the path to the generated M3U file, or None if no MP3 files found.
    """
    if not folder_path.exists():
        return None

    # Collect all MP3 files sorted alphabetically
    mp3_files: List[Path] = sorted(folder_path.glob("*.mp3"), key=lambda p: p.stem.lower())

    if not mp3_files:
        return None

    m3u_path = folder_path / "playlist.m3u"
    tmp_path = m3u_path.with_suffix(".tmp")

    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            f.write(f"#PLAYLIST:{_escape_m3u_title(playlist_name)}\n\n")

            for mp3_file in mp3_files:
                # Get duration from metadata; use 0 if unavailable
                duration = get_mp3_duration(mp3_file)
                if duration < 0:
                    duration = 0
                # Title = filename stem, escaped for M3U safety
                title = _escape_m3u_title(mp3_file.stem)
                f.write(f"#EXTINF:{duration},{title}\n")
                f.write(f"{mp3_file.name}\n")

            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(tmp_path, m3u_path)
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return m3u_path


def export_playlist(playlist_folder: Path, playlist_name: str) -> Optional[Path]:
    """Main entry point for exporting a playlist."""
    return generate_m3u_playlist(playlist_folder, playlist_name)