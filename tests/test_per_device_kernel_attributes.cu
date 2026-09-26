// Multi-GPU regression for per-device kernel setup. One kernel that needs more than the default
// 48 KiB of dynamic shared memory is configured through PerDeviceOnce, exactly like the Op
// launchers, then launched on device 0, on device 1, and on device 0 again. A guard initialized
// once per process configures only the first device, so the launch on the second one fails.
// Needs two CUDA devices and skips otherwise.
#include "core/per_device.h"

#include <cuda_runtime.h>

#include <iostream>

namespace {

constexpr int kThreads = 256;
// Above the 48 KiB launch default and below the opt-in limit of every sm_70+ device.
constexpr int kSharedBytes = 64 * 1024;
constexpr int kWords       = kSharedBytes / static_cast<int>(sizeof(unsigned));

__global__ void large_shared_sum_kernel(unsigned long long* sum) {
    extern __shared__ unsigned words[];
    for (int index = threadIdx.x; index < kWords; index += blockDim.x) {
        words[index] = static_cast<unsigned>(index);
    }
    __syncthreads();
    // Each thread reads words written by other threads, so the whole allocation is exercised.
    unsigned long long partial = 0;
    for (int index = threadIdx.x; index < kWords; index += blockDim.x) {
        partial += words[kWords - 1 - index];
    }
    atomicAdd(sum, partial);
}

cudaError_t launch_large_shared_sum(unsigned long long* sum) {
    static ninfer::PerDeviceOnce<cudaError_t> attribute;
    const cudaError_t setup = attribute.get([] {
        return cudaFuncSetAttribute(large_shared_sum_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    });
    if (setup != cudaSuccess) { return setup; }
    large_shared_sum_kernel<<<1, kThreads, kSharedBytes>>>(sum);
    return cudaGetLastError();
}

bool succeeded(cudaError_t err, int device, const char* what) {
    if (err == cudaSuccess) { return true; }
    std::cerr << "device " << device << ": " << what << " failed: " << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << '\n';
    return false;
}

bool dynamic_shared_limit(int device, int& bytes) {
    cudaFuncAttributes attributes{};
    if (!succeeded(cudaFuncGetAttributes(&attributes, large_shared_sum_kernel), device,
                   "cudaFuncGetAttributes")) {
        return false;
    }
    bytes = attributes.maxDynamicSharedSizeBytes;
    return true;
}

struct DeviceSum {
    unsigned long long* data = nullptr;

    DeviceSum()                            = default;
    DeviceSum(const DeviceSum&)            = delete;
    DeviceSum& operator=(const DeviceSum&) = delete;

    ~DeviceSum() {
        if (data != nullptr) { cudaFree(data); }
    }
};

// Returns the number of failures observed on `device`.
int run_on(int device) {
    if (!succeeded(cudaSetDevice(device), device, "cudaSetDevice")) { return 1; }

    int limit_before = 0;
    if (!dynamic_shared_limit(device, limit_before)) { return 1; }
    std::cout << "device " << device << ": dynamic shared-memory limit before launch "
              << limit_before << " bytes\n";

    DeviceSum sum;
    if (!succeeded(cudaMalloc(&sum.data, sizeof(*sum.data)), device, "cudaMalloc") ||
        !succeeded(cudaMemset(sum.data, 0, sizeof(*sum.data)), device, "cudaMemset") ||
        !succeeded(launch_large_shared_sum(sum.data), device, "large shared-memory launch") ||
        !succeeded(cudaDeviceSynchronize(), device, "cudaDeviceSynchronize")) {
        return 1;
    }

    unsigned long long result = 0;
    if (!succeeded(cudaMemcpy(&result, sum.data, sizeof(result), cudaMemcpyDeviceToHost), device,
                   "cudaMemcpy")) {
        return 1;
    }
    int failures                      = 0;
    const unsigned long long expected = static_cast<unsigned long long>(kWords) * (kWords - 1) / 2;
    if (result != expected) {
        std::cerr << "device " << device << ": sum " << result << ", expected " << expected << '\n';
        ++failures;
    }

    int limit_after = 0;
    if (!dynamic_shared_limit(device, limit_after)) { return failures + 1; }
    if (limit_after != kSharedBytes) {
        std::cerr << "device " << device << ": dynamic shared-memory limit " << limit_after
                  << " bytes after setup, expected " << kSharedBytes << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    // The stub driver is what a CPU-only build container links against: no device either.
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver ||
        count_err == cudaErrorStubLibrary) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count < 2) {
        std::cout << "SKIP: needs two CUDA devices, found " << count << '\n';
        return 77;
    }

    int failures = 0;
    failures += run_on(0);
    failures += run_on(1);
    failures += run_on(0);
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
