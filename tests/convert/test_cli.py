from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.torch import save_file
import torch

from tools.artifact.framing import HEADER
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

from .path_fragments import assert_no_path_fragments


def test_cli_custom_sources_method_template_and_shards(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (source / name).write_text(json.dumps(value))
    template = tmp_path / "template.jinja"
    template.write_text("custom {{ messages }}")
    recipe = tmp_path / "recipe.py"
    recipe.write_text("""import torch
from tools.convert.sources.logical import array_source

def custom(request):
    return request.job(produce=lambda output: output.write_values(0, request.source.values().to(torch.bfloat16)))

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        value = torch.ones(parameter.shape, dtype=torch.bfloat16)
        recipe.assign(name, source=array_source(value, "my-importer"))
    recipe.assign(("text/token_embedding", "text/layers/0/mlp/gate", "text/layers/0/mlp/up"), method=custom)
    for role, format in zip(("query", "key", "gate", "value"), ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")):
        recipe.assign("text/layers/0/attention/" + role, format=format, method="grouped_absmax")
""")
    output = tmp_path / "custom.ninfer"
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.convert",
        "--model",
        str(source),
        "--recipe",
        str(recipe),
        "--out",
        str(output),
        "--name",
        "my-training-run",
        "--resource",
        f"chat_template.jinja={template}",
        "--source",
        f"unused={tmp_path / 'missing'}",
        "--device",
        "cpu",
        "--max-file-bytes",
        "16384",
        "--rows-per-chunk",
        "3",
    ]
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    with Artifact(output) as artifact:
        assert set(artifact.directory.components) == {"text"}
        assert artifact.directory.metadata == {"name": "my-training-run"}
        assert len(artifact.directory.files) > 1
        resource = artifact.directory.components["text"]["resources"][
            "chat_template.jinja"
        ]
        assert artifact.read_object(resource) == template.read_bytes()
        embedding = binding_parts(
            artifact.directory.bindings["text/token_embedding"], artifact.by_id
        )[0][0]
        assert artifact.read_object(embedding) == b"\x80\x3f" * (8 * 128)
        parents = []
        for role in ("query", "key", "gate", "value"):
            name = "text/layers/0/attention/" + role
            parents.append(
                binding_parts(artifact.directory.bindings[name], artifact.by_id)[0][0]
            )
        assert len(set(parents)) == 4
        assert [artifact.object(parent).format for parent in parents] == [
            "q4_g64_fp16",
            "q5_g64_fp16",
            "q6_g64_fp16",
            "q8_g32_fp16",
        ]
        gate = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/gate"], artifact.by_id
        )[0][0]
        up = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/up"], artifact.by_id
        )[0][0]
        assert gate != up


def test_cli_records_inputs_by_name_without_local_paths(tmp_path):
    source = tmp_path / "checkpoints" / "Qwen3.5-Tiny"
    source.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (source / name).write_text(json.dumps(value))
    (source / "chat_template.jinja").write_text("{{ messages }}")
    generator = torch.Generator().manual_seed(0)
    save_file(
        {
            name: torch.randn(8, 128, generator=generator).to(torch.bfloat16)
            for name in ("model.embed_tokens.weight", "lm_head.weight")
        },
        source / "model.safetensors",
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    recipe = inputs / "recipe.py"
    recipe.write_text("""import torch
from tools.convert.sources.logical import array_source

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        if name not in ("text/token_embedding", "text/output_head"):
            value = torch.ones(parameter.shape, dtype=torch.bfloat16)
            recipe.assign(name, source=array_source(value, "ones"))
""")
    override = inputs / "override.py"
    override.write_text("""def narrow_head(model, recipe, sources):
    recipe.assign("text/output_head", format="q8_g32_fp16", method="grouped_absmax")
""")
    ranking = inputs / "counts.i64"
    np.arange(8, 0, -1, dtype="<i8").tofile(ranking)
    template = inputs / "template.jinja"
    template.write_text("custom {{ messages }}")
    output = tmp_path / "out" / "tiny.ninfer"
    command = [
        sys.executable, "-B", "-m", "tools.convert",
        "--model", str(source),
        "--recipe", str(recipe),
        "--override", f"{override}:narrow_head",
        "--proposal", "--ranking", str(ranking), "--proposal-rows", "4",
        "--resource", f"chat_template.jinja={template}",
        "--source", f"unused={tmp_path / 'missing'}",
        "--device", "cpu",
        "--max-file-bytes", "16384",
        "--out", str(output),
    ]
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr

    def digest(path):
        return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    expected = {
        "converter": "ninfer-v3",
        "recipe": "recipe.py:configure",
        "recipe_file": digest(recipe),
        "override": "override.py:narrow_head",
        "override_file": digest(override),
        # A source is recorded by role and directory name, and only once it is opened.
        "sources": {"base": {"name": "Qwen3.5-Tiny"}},
        "ranking": digest(ranking),
    }
    with output.open("rb") as stream:
        _, directory_bytes, _ = HEADER.unpack(stream.read(HEADER.size))
        directory = stream.read(directory_bytes).decode("utf-8")
    assert json.loads(directory)["provenance"] == expected
    assert_no_path_fragments(directory, tmp_path)

    report_text = Path(f"{output}.conversion.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["provenance"] == expected
    assert report["output"] == "tiny.ninfer"
    names = [file["name"] for file in report["files"]]
    assert len(names) > 1
    assert names == ["tiny.ninfer"] + [
        f"tiny.ninfer.part-{index:04d}" for index in range(1, len(names))
    ]
    # Source labels name a checkpoint tensor by the source's role.
    assert {label for job in report["methods"] for label in job["sources"]} == {
        "ones",
        "base:model.embed_tokens.weight",
        "base:lm_head.weight",
        "rows(base:lm_head.weight, 4 spans, 4 rows)",
        "shortlist(counts.i64)",
    }
    assert_no_path_fragments(report_text, tmp_path)
