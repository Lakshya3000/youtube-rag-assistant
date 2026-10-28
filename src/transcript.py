"""Utilities for extracting a YouTube video ID and fetching captions."""
from urllib.parse import parse_qs, urlparse


def extract_video_id(value: str) -> str:
    """Return a video ID from a YouTube URL, or return a supplied plain ID."""
    value = value.strip()
    if not value:
        raise ValueError("Enter a YouTube video ID or URL.")

    if "://" not in value and "/" not in value and "?" not in value:
        return value

    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            pieces = parsed.path.strip("/").split("/")
            video_id = pieces[1] if len(pieces) > 1 else ""
        else:
            video_id = ""
    else:
        raise ValueError("That does not look like a supported YouTube URL.")

    if not video_id:
        raise ValueError("Could not extract a video ID from that URL.")
    return video_id


def fetch_youtube_transcript(video_input: str, languages: tuple[str, ...] = ("en",)) -> tuple[str, str]:
    """Fetch a transcript; returns (transcript_text, video_id).

    YouTube may block requests from cloud IPs. Callers should offer a transcript
    upload/paste fallback rather than retrying indefinitely.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_video_id(video_input)
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    text = " ".join(
        snippet.text.replace("\n", " ").strip()
        for snippet in transcript.snippets
        if snippet.text and snippet.text.strip()
    )
    if not text:
        raise ValueError("The transcript was empty. Upload or paste a transcript instead.")
    return text, video_id
