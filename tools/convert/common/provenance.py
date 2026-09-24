"""Provenance records that name their inputs without local filesystem paths.

An artifact's directory JSON travels with the artifact. A local path describes one machine (its
home, scratch or mount directories), not the input, so converters that record provenance name
their inputs by label or repository id, and remove the path members that an earlier artifact's
provenance carries before extending it.
"""

from __future__ import annotations

import argparse
from pathlib import PureWindowsPath


def is_local_path(value: str) -> bool:
    """Whether ``value`` is an absolute, home-relative, Windows drive/UNC, or explicitly relative
    filesystem path -- one with a literal ``.`` or ``..`` component, such as ``..``,
    ``../checkpoints/model`` or ``./model``.

    A Hugging Face repository id (``org/model``) or an artifact id has no such component and is
    not a path, even though it may contain ``/`` or ``.`` characters within a segment.
    """

    if value.startswith(("/", "~", "\\")) or bool(PureWindowsPath(value).drive):
        return True
    parts = value.replace("\\", "/").split("/")
    return "." in parts or ".." in parts


def input_label(value: str) -> str:
    """``argparse`` type of an input label: a repository id or name, never a local path."""

    if not value.strip() or value != value.strip() or is_local_path(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an input label: name the input by repository id or name "
            "(for example org/model), not by a filesystem path"
        )
    return value


def _is_path_member(key: str, value: object) -> bool:
    if key == "path" or key.endswith("_path"):
        return True
    return isinstance(value, str) and is_local_path(value)


def strip_local_paths(value: object, location: str = "") -> tuple[object, list[str]]:
    """Copy of a JSON value without its filesystem paths, and where they were (``a.b[2].path``).

    A member named ``path`` or ``*_path`` and any string that is a local path are removed; so is
    an object or array that only held removed members. Everything else is kept as it is.
    """

    removed: list[str] = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            member = f"{location}.{key}" if location else key
            if _is_path_member(key, item):
                removed.append(member)
                continue
            kept, inner = strip_local_paths(item, member)
            removed.extend(inner)
            if inner and isinstance(kept, (dict, list)) and not kept:
                continue
            result[key] = kept
        return result, removed
    if isinstance(value, list):
        items = []
        for index, item in enumerate(value):
            element = f"{location}[{index}]"
            if isinstance(item, str) and is_local_path(item):
                removed.append(element)
                continue
            kept, inner = strip_local_paths(item, element)
            removed.extend(inner)
            if inner and isinstance(kept, (dict, list)) and not kept:
                continue
            items.append(kept)
        return items, removed
    return value, removed
