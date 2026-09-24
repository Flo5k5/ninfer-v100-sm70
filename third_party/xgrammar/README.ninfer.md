# Vendored xgrammar

NInfer vendors the C++ core of the upstream xgrammar `v0.2.7` release for grammar-constrained
decoding:

- upstream: <https://github.com/mlc-ai/xgrammar>
- tag: `v0.2.7`
- commit: `82505d0d987c36a4209fb3d8571cf6b0f28b5acd`
- license: Apache-2.0; see `LICENSE` and `NOTICE`
- bundled headers: picojson (BSD-2-Clause, license in `3rdparty/picojson/picojson.h`) and dlpack
  `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` (Apache-2.0, `3rdparty/dlpack/LICENSE`)

The committed `include/`, `cpp/` and `3rdparty/` files are unchanged upstream files. The Python,
TVM-FFI, web and Swift bindings, tests, documentation and the upstream build system are
intentionally omitted.

Local additions, outside the upstream tree layout:

- `CMakeLists.txt` builds the static C++ core with `XGRAMMAR_LOG_CUSTOMIZE=1` and without
  cpptrace, and with warnings left to the upstream sources;
- `ninfer_hooks/logging_hooks.cc` implements the two customization hooks. Nothing is printed, so
  request schemas, regex patterns and grammar text never reach the process logs. A fatal message
  becomes a `std::runtime_error` carrying only the message. A warning goes to the calling thread's
  active `WarningCapture` (`ninfer_hooks/warning_capture.h`) and is dropped otherwise: xgrammar warns
  when it relaxes a construct it cannot represent exactly, and the grammar service rejects such
  grammars rather than constrain output with a weaker one.

NInfer does not discover a system package or fetch network content during configuration. To update,
replace the upstream files from the new tag, keep the local additions, and rerun the grammar tests.
