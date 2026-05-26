"""Plain-text formatting of recipe download metadata.

Shared by the dataset catalog generator and ``vkit recipes`` (which adds Rich
colour on top). Returns plain strings — no Rich markup — so it is safe for
both Markdown and terminal use.
"""

from __future__ import annotations

from urllib.parse import urlparse


def format_size_range(download_sizes: dict[str, int]) -> str:
    """Plain size column: '—' when unknown, a single size, or 'min - max'.

    Uses a hyphen-minus separator (matching the original ``vkit recipes``
    table) so that refactoring the CLI helper to delegate here produces
    identical output.
    """
    from voxkitchen.utils.download import format_bytes

    if not download_sizes:
        return "—"
    values = list(download_sizes.values())
    lo, hi = min(values), max(values)
    if lo == hi:
        return format_bytes(lo)
    return f"{format_bytes(lo)} - {format_bytes(hi)}"


def download_source_label(download_urls: dict[str, list[str]]) -> str:
    """Plain source label inferred from the first URL host; 'url' if empty.

    Matches the host-detection logic of the original ``_download_source_label``
    in ``vkit recipes`` exactly:

    - ``openslr.org`` → ``"openslr"``
    - ``keithito.com`` / ``data.keithito.com`` → ``"keithito"``
    - ``huggingface.co`` / ``hf.co`` → ``"HuggingFace"``
    - anything else → bare hostname (``www.`` prefix stripped)
    - no entries → ``"url"``
    """
    for urls in download_urls.values():
        if not urls:
            continue
        host = urlparse(urls[0]).hostname or ""
        host = host.lower()
        if "openslr" in host:
            return "openslr"
        if "keithito" in host:
            return "keithito"
        if "huggingface" in host or "hf.co" in host:
            return "HuggingFace"
        # Strip a leading "www." for a tidier label.
        return host.removeprefix("www.")
    return "url"
