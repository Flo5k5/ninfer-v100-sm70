#!/usr/bin/env bash
# Build every CMake target of the sm_70 configuration: libraries, apps, tests, and benchmarks.
# A default configure leaves tests and benchmarks out, so this is the check that keeps them
# compiling and linking. Nothing is run: the test executables need a V100.
#
# Environment:
#   NINFER_SOURCE_DIR  repository to build (default: the checkout that contains this script)
#   NINFER_BUILD_DIR   build directory, reused incrementally (default: build-v100-all in the
#                      source directory)
#   NINFER_BUILD_JOBS  parallel jobs (default: the Ninja default, which follows the visible CPU
#                      count; set it explicitly under a CPU quota such as docker --cpus)
# Extra arguments go to the CMake configure step, for example
# -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc or -DFETCHCONTENT_SOURCE_DIR_CUTLASS=<dir>.
#
# The build keeps going after a failure (ninja -k 0), so one run reports every broken target. The
# exit status is the verdict. failed-targets.txt in the build directory names the targets whose
# build step failed, one per line; it stays empty when configure fails or every target builds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
SOURCE_DIR="${NINFER_SOURCE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly SOURCE_DIR
readonly BUILD_DIR="${NINFER_BUILD_DIR:-${SOURCE_DIR}/build-v100-all}"
readonly BUILD_LOG="${BUILD_DIR}/full-build.log"
readonly FAILED_TARGETS="${BUILD_DIR}/failed-targets.txt"

parallel_args=(--parallel)
if [[ -n "${NINFER_BUILD_JOBS:-}" ]]; then
    parallel_args+=("${NINFER_BUILD_JOBS}")
fi

mkdir -p "${BUILD_DIR}"
: > "${FAILED_TARGETS}"
SECONDS=0

if ! cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=ON \
    -DNINFER_BUILD_BENCHMARKS=ON \
    "$@"; then
    echo "ninfer sm_70 full build: configure failed" >&2
    exit 1
fi

build_status=0
cmake --build "${BUILD_DIR}" "${parallel_args[@]}" -- -k 0 2>&1 | tee "${BUILD_LOG}" \
    || build_status=$?

# Ninja prints "FAILED: <output>" for each failed edge (1.12+ inserts "[code=N] "). Map object
# files and device links to their CMake target through the "CMakeFiles/<target>.dir/" component,
# and linked outputs through their file name.
sed -n -E 's/^FAILED: (\[code=[0-9]+\] )?([^ ]+).*/\2/p' "${BUILD_LOG}" \
    | sed -E 's#^(.*/)?CMakeFiles/([^/]+)\.dir/.*#\2#; s#.*/##; s#^lib(.+)\.a$#\1#' \
    | sort -u > "${FAILED_TARGETS}"

readonly ELAPSED="$((SECONDS / 60))m$((SECONDS % 60))s"
if (( build_status == 0 )); then
    echo "ninfer sm_70 full build: every target built in ${ELAPSED}"
    exit 0
fi
if [[ -s "${FAILED_TARGETS}" ]]; then
    echo "ninfer sm_70 full build failed after ${ELAPSED}; failed targets:" >&2
    sed 's/^/  /' "${FAILED_TARGETS}" >&2
else
    echo "ninfer sm_70 full build failed after ${ELAPSED} without a failed target; see ${BUILD_LOG}" >&2
fi
exit "${build_status}"
