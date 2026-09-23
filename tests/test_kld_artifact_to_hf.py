from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.kld import artifact_to_hf


def test_nvfp4_dequantization_decodes_codes_scales_and_divisor() -> None:
    rows, columns = 128, 64
    generator = torch.Generator().manual_seed(3)
    packed = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
    # Nonnegative finite E4M3FN words (0x7F is NaN).
    scales = torch.randint(0, 0x7F, (rows, columns // 16), dtype=torch.uint8, generator=generator)
    divisor = torch.tensor(37.5, dtype=torch.float32)
    payload = encode_nvfp4(packed, scales, divisor.numpy().tobytes(), (rows, columns))

    values = artifact_to_hf.dequantize_nvfp4(payload, (rows, columns))

    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for row, column in ((0, 0), (5, 1), (77, 30), (127, 63)):
        byte = int(packed[row, column // 2])
        code = byte & 0x0F if column % 2 == 0 else byte >> 4
        magnitude = levels[code & 0x7] * (-1.0 if code & 0x8 else 1.0)
        scale = float(scales[row, column // 16].view(torch.float8_e4m3fn).float())
        assert float(values[row, column]) == pytest.approx(magnitude * scale / 37.5, rel=1e-6)


class _FakeDecoder:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def logical(self, name: str) -> tuple[torch.Tensor, list[str]]:
        return self.values[name], ["fp8_e4m3fn_row_bf16"]


def test_assemble_interleaves_selected_rows_and_transposes_the_convolution() -> None:
    source = torch.zeros(8, 2)
    query = torch.full((4, 2), 1.0)
    gate = torch.full((4, 2), 2.0)
    decoder = _FakeDecoder({"query": query, "gate": gate})
    heads = [((0, 2), (4, 6)), ((2, 4), (6, 8))]
    values, formats = artifact_to_hf.assemble(
        source, [("query", heads[0]), ("gate", heads[1])], decoder)
    assert values[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
    assert formats == ["fp8_e4m3fn_row_bf16"]

    kernel = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # [taps, channels]
    decoder = _FakeDecoder({"text/layers/0/gdn/convolution": kernel})
    values, _ = artifact_to_hf.assemble(
        torch.zeros(3, 1, 4), [("text/layers/0/gdn/convolution", None)], decoder)
    assert values[:, 0, :].tolist() == kernel.transpose(0, 1).tolist()
