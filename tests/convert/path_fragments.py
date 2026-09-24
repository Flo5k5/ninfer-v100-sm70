"""The check every converter test applies to what it writes: no local filesystem path.

An artifact's directory JSON and a converter's side report travel apart from the machine that
wrote them, so neither may hold a path to an input or output.
"""

from __future__ import annotations

from pathlib import Path
import re
import tempfile


# Converters also read inputs from the checkout itself (the draft-head ranking fixture), whose
# location is just as local as tmp_path.
CHECKOUT = Path(__file__).resolve().parents[2]


def assert_no_path_fragments(text: str, tmp_path: Path) -> None:
    """No JSON string (member name or value) is, or starts like, a local filesystem path."""

    for fragment in (str(tmp_path), str(CHECKOUT), str(Path.home()), tempfile.gettempdir(),
                     "/home/", "/tmp/", "/private/", "/Users/", "/data/", "/var/"):
        assert fragment not in text, fragment
    # A JSON string (member name or value) that starts like an absolute or home path.
    assert not re.findall(r'"(?:/|~|\\\\|[A-Za-z]:)', text)
