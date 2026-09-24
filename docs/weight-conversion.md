# Weight conversion

NInfer's converter creates `.ninfer` artifacts from local weights and a Python recipe. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

The converter is upstream NInfer's v3 pipeline. It writes the same artifacts on this branch, apart
from the provenance and report records described [below](#resources-files-and-inspection). The
Volta Engine runs a narrower set of them than upstream's; read
[what this port runs](#what-this-port-runs) before changing a recipe.

Run the commands below from the repository root.

## Start with an official recipe

Source-weight conversion requires Python 3.10 or later, with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion. The input paths below are placeholders for your
local checkpoint directories.

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the indexed proposal head used by speculative decoding; it uses the repository's
token ranking and defaults to 131,072 rows. The ordinary full-vocabulary output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |

On this port, only the two NVFP4 recipes produce artifacts that load today; see
[what this port runs](#what-this-port-runs).

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

MTP and Vision use the main source. DFlash and DFlash2 use the corresponding named source, supplied
as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

## What this port runs

The Volta Engine keeps the registered targets and weight profiles of its kernels. At load, it takes
the model from the artifact's `name` metadata (`--name`) and the weights profile from the official
recipe name recorded in its provenance. It then binds every weight its profile needs by name, and
checks each weight's format and shape.

An artifact therefore runs on this port when:

- `--name` is the registered model: `qwen3.6-27b`, `qwen3.8-27b` or `qwen3.6-35b-a3b`;
- `--recipe` is the official recipe for that model and profile, so a change goes in an
  `--override` file rather than in a recipe file, whose own name no profile registers;
- the result keeps the official format and shape of every weight the profile binds.

Replacing values from another compatible checkpoint of the same architecture, with the same
formats, keeps an artifact runnable. Changing formats, splitting rows or regrouping projections
still produces a valid file, but the Engine refuses it when it binds the profile.

At present the profile is resolved for the two NVFP4 recipes only. `Reader::identity`
(`src/artifact/reader.cpp`) takes the text after the recipe name's last underscore as the profile.
The three `groupwise-int` recipes therefore fail at load, and so do the published `groupwise-int` v3
artifacts, until the identity fix in [#17](https://github.com/Flo5k5/ninfer-v100-sm70/pull/17)
lands. Qwen3.6-35B-A3B needs more than that fix: the v2-to-v3 shim cannot resolve the MoE expert
weights its binder addresses, so no v3 artifact of that model loads yet.

## Change part of a recipe

Save the following as `my_override.py`:

```python
def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it after the official recipe, with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --override my_override.py \
  --components text,vision,mtp \
  --proposal \
  --name qwen3.6-27b \
  --out models/my_qwen.ninfer
```

This example changes a stored format, so the file it writes is not one this port runs. The default
entry function is `configure`; `my_override.py:customize` selects another function. A complete
recipe file can also replace `--recipe`: it calls the official function first, for example
`qwen3_6_27b(model, recipe, sources)` from `tools.convert.official_recipes`, then applies its
changes. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `nvfp4` | Supply a custom quantizer | `import_encoded` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them.

The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Those references still use the v2 names:
`Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S` and `W8G32_F16S` for the grouped formats above,
`FP8_E4M3FN_ROW_BF16S`, `NVFP4`, `BF16`, `FP32` and `I32`, with the same meaning. Source format
names alone do not establish compatibility: scale direction, granularity, code meaning and axis
order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
Upstream's Engine also runs a single-parent FP8 form of the four projections, which an override
can request:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

This port binds the official groupwise parents only, so it does not run that form. The adapter
handles source Q/gate row order; the override works with logical projections. Changing their
representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in an override, keeping its
official format:

```python
def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    recipe.assign(name, source=model.source(name, sources["alternate"]))
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. For a weight that the recipe imports
encoded, request the encoded rows with its format, as the official recipes do:
`model.source(name, sources["alternate"], "nvfp4")`. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, an override can
read a logical matrix from a NumPy file stored beside it:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), path.name, read_values)
    recipe.assign(name, source=source)
```

The second argument is the source's label, which the conversion report records: give a name, not
a path. The NumPy file must already follow the logical row/column order. For an unfamiliar
quantized source, its reader performs the corresponding decoding before returning values. To
preserve existing compatible encoded words, also provide `read_encoded` returning `EncodedRows`,
and the format's required divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override: it keeps the official format of the projection it changes.
`request.values` traverses the prepared logical inputs in parent order, including explicit groups.
`output.write_codes` performs the registered packing and validates codes/scales; the method should
not duplicate that byte-layout logic. Direct output uses `output.write_values`. Keep source blocks
and temporary device tensors bounded to the method's working set. `--rows-per-chunk` defaults to
512; custom methods own how they use it.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`; `--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

This port renders the official Qwen3.6 and Qwen3.8 chat templates, which it recognizes by their
SHA-256 digest. When the stored `chat_template.jinja` is another template, such as upstream's
maintained templates, it renders the template that `tokenizer_config.json` carries instead, and
refuses to start if neither is official. `generation_config.json` is preserved; sampling presets
remain determined by the architecture and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to NInfer and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

The artifact's provenance and the report travel without the machine that wrote them, so neither
records a local path:

- each opened source appears under its role (`base`, `quantized`, `dflash2`, ...) with its
  directory or file name, and source labels name tensors by that role, as in
  `base:lm_head.weight`;
- a recipe or override file appears as `name:function`, with its name and SHA-256 digest; an
  official recipe keeps its name;
- the proposal ranking appears with its name and SHA-256 digest;
- the report names the artifact and its part files by file name.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference. Conversion rejects missing logical
coverage, invalid source geometry, unsupported encodings and invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.
