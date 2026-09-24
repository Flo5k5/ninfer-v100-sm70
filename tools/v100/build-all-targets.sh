#!/usr/bin/env bash
# Build every CMake target of the sm_70 configuration: libraries, apps, tests, and benchmarks.
# A default configure leaves tests and benchmarks out, so this is the check that keeps them
# compiling and linking. Nothing is run: the test executables need a V100.
#
# Usage: build-all-targets.sh [-DVAR=VALUE | -UVAR]...
# Each argument goes to the CMake configure step, for example
# -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc or -DFETCHCONTENT_SOURCE_DIR_CUTLASS=<dir>.
# The architecture, build type, and target gates stay forced whatever the arguments.
#
# Environment:
#   NINFER_SOURCE_DIR           repository to build (default: the checkout containing this script)
#   NINFER_BUILD_DIR            build directory, reused incrementally (default: build-v100-all in
#                               the source directory)
#   CMAKE_BUILD_PARALLEL_LEVEL  parallel jobs (default: the Ninja default, which follows the visible
#                               CPU count; set it to the quota under docker --cpus)
#
# The build keeps going after a failure (ninja -k 0), so one run reports every broken target. The
# build directory receives full-build.log (configure and build output) and failed-targets.txt (the
# targets whose build step failed, one per line).
#
# Exit status:
#   0  every target built
#   1  the build failed; failed-targets.txt names the failed targets, if a step failed on its own
#   2  configure failed
#   3  usage or environment error
set -euo pipefail

usage_error() {
    echo "build-all-targets.sh: $*" >&2
    exit 3
}

for argument in "$@"; do
    case "${argument}" in
        -h | --help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -D?* | -U?*) ;;
        *) usage_error "unsupported argument '${argument}': only -DVAR=VALUE and -UVAR (see --help)" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
SOURCE_DIR="${NINFER_SOURCE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly SOURCE_DIR
readonly BUILD_DIR="${NINFER_BUILD_DIR:-${SOURCE_DIR}/build-v100-all}"
readonly BUILD_LOG="${BUILD_DIR}/full-build.log"
readonly FAILED_TARGETS="${BUILD_DIR}/failed-targets.txt"

if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" && ! "${CMAKE_BUILD_PARALLEL_LEVEL}" =~ ^[1-9][0-9]*$ ]]; then
    usage_error "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer, got '${CMAKE_BUILD_PARALLEL_LEVEL}'"
fi
[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || usage_error "no CMakeLists.txt in ${SOURCE_DIR}"
for tool in cmake ninja; do
    command -v "${tool}" > /dev/null || usage_error "${tool} is not on PATH"
done
if ! { mkdir -p "${BUILD_DIR}" && : > "${BUILD_LOG}" && : > "${FAILED_TARGETS}"; } 2> /dev/null; then
    usage_error "cannot write to ${BUILD_DIR}"
fi

SECONDS=0
# The arguments come first so that the forced settings after them always win.
if ! cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja "$@" \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=ON \
    -DNINFER_BUILD_BENCHMARKS=ON 2>&1 | tee "${BUILD_LOG}"; then
    echo "ninfer sm_70 full build: configure failed; see ${BUILD_LOG}" >&2
    exit 2
fi

build_status=0
cmake --build "${BUILD_DIR}" -- -k 0 2>&1 | tee -a "${BUILD_LOG}" || build_status=$?

# Ninja prints "FAILED: <output>" for each failed edge (1.13+ inserts "[code=N] "). Map object
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
exit 1
