#!/usr/bin/env python3
"""Import a published all-NVFP4 checkpoint (QUASAR-QAT) into a NInfer artifact.

QUASAR quantized every matrix including the attention and GDN projections, with QAT
(distillation) fidelity. The stage-A rewrite tool converts MLP 56-63 and the output
head from a BF16 source against the published NVFP4 words; this wrapper adds the
checkpoint download and the artifact verification, and documents what is NOT imported
(the attention and GDN entries — they need the kernel routes of the mission's
palier 2) so the artifact is a correct G23-equivalent built from published words.

Usage (on solo, CPU only, after the palier 2 kernels are merged for the full route):
    python3 -m tools.convert.qwen3_8_27b.import_quasar \
      --base /data/llamacpp/models-hf/orca-ninfer/qwen3_8_27b_orca_nvfp4.ninfer \
      --quasar-repo QUASAR/Qwen3.8-27B-NVFP4-QAT \
      --out /data/llamacpp/models-hf/quasar-ninfer/qwen3_8_27b_nvfp4_quasar.ninfer

The base is the orca NVFP4 v1 artifact (for the bindings, Uses, and the non-converted
objects); the QUASAR checkpoint provides the calibrated words for MLP 56-63 and the
output head, and later the attention/GDN entries once palier 2 lands.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

QUASAR_HF_REPO = "QUASAR/Qwen3.8-27B-NVFP4-QAT"
# The stage-A tool accepts any HF NVFP4 checkpoint as --reference.
# QUASAR publishes ModelOpt layout (weight / weight_scale / weight_scale_2).


def download(repo: str, destination: Path) -> Path:
    """Clone the HF checkpoint to destination (git LFS or huggingface-cli)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / "config.json").exists():
        print(f"checkpoint already present at {destination}")
        return destination
    url = f"https://huggingface.co/{repo}"
    print(f"cloning {repo} to {destination} (~19 GB, git LFS)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, type=Path,
                        help="mixed NVFP4/FP8 NInfer artifact (the original v1)")
    parser.add_argument("--quasar-repo", default=QUASAR_HF_REPO,
                        help=f"HF repo of the published checkpoint (default: {QUASAR_HF_REPO})")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="existing QUASAR checkpoint directory (skips the download)")
    parser.add_argument("--out", required=True, type=Path,
                        help="output artifact path")
    parser.add_argument("--bf16-source", type=Path,
                        default=Path("/data/llamacpp/models-hf/orcarouter-Qwen3.8-27B-Uncensored-BF16"),
                        help="BF16 checkpoint for the --weights comparison (default: orca BF16)")
    parser.add_argument("--no-download", action="store_true",
                        help="never download; error if the checkpoint is missing")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path("/data/llamacpp/models-hf") / args.quasar_repo.split("/")[-1]
    if not (checkpoint / "config.json").exists():
        if args.no_download:
            parser.error(f"checkpoint not found at {checkpoint} and --no-download is set")
        download(args.quasar_repo, checkpoint)

    # The stage-A rewrite: MLP 56-63 + output head, with QUASAR as the reference
    # for the published NVFP4 words (QAT fidelity) and the BF16 source for
    # the weights comparison.
    command = [
        sys.executable, "-m", "tools.convert.qwen3_8_27b.rewrite_nvfp4",
        "--base", str(args.base),
        "--source", str(args.bf16_source),
        "--source-label", "orcarouter/Qwen3.8-27B-Uncensored-BF16",
        "--reference", str(checkpoint),
        "--stage", "a",
        "--out", str(args.out),
        "--verify",
    ]
    print("running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"stage-A rewrite failed (rc={result.returncode})")
        return result.returncode

    print(f"\nartifact written to {args.out}")
    print("note: attention and GDN entries are NOT converted by stage A;")
    print("      they need the kernel routes of palier 2 (mission-vitesse-27b.md).")
    print("      this artifact is G23-equivalent from published QAT words.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
