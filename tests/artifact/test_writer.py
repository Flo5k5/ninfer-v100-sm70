from __future__ import annotations

import os
import stat

import pytest

from tools.artifact.framing import PAYLOAD_ALIGNMENT
from tools.artifact.schema import ResourceSpec
from tools.artifact.writer import ArtifactWriter

BLOB = "resource/blob"


def _writer(path):
    """Two alignment units per file: a two-unit payload spans the entry and one part."""
    return ArtifactWriter(
        path,
        [ResourceSpec(BLOB, 2 * PAYLOAD_ALIGNMENT)],
        components={"text": {"config": {}}},
        bindings={},
        max_file_bytes=2 * PAYLOAD_ALIGNMENT,
    )


def _mode(path):
    return oct(stat.S_IMODE(path.stat().st_mode))


# Neither umask yields mkstemp's 0600, and they differ, so no fixed mode passes both.
@pytest.mark.parametrize("umask", [0o002, 0o027], ids=oct)
def test_published_files_get_the_mode_of_a_normal_file_creation(tmp_path, umask):
    path = tmp_path / "model.ninfer"
    previous = os.umask(umask)
    try:
        with _writer(path) as writer:
            writer.write_object(BLOB, bytes(2 * PAYLOAD_ALIGNMENT))
    finally:
        os.umask(previous)

    published = [path, tmp_path / "model.ninfer.part-0001"]
    assert sorted(tmp_path.iterdir()) == published
    assert [_mode(file) for file in published] == [oct(0o666 & ~umask)] * 2


def test_finish_keeps_an_output_that_appeared_during_the_write(tmp_path):
    path = tmp_path / "model.ninfer"
    writer = _writer(path)
    writer.write_object(BLOB, bytes(2 * PAYLOAD_ALIGNMENT))
    path.write_bytes(b"foreign")

    with pytest.raises(FileExistsError):
        writer.finish()
    assert path.read_bytes() == b"foreign"
    assert list(tmp_path.iterdir()) == [path]
