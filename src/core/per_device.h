#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {

// Exclusive upper bound of the CUDA device ordinals a PerDeviceOnce can key. The slots are stored
// inline so that a lookup never allocates; a larger ordinal is rejected instead of being folded
// onto another device's slot.
inline constexpr int kMaxCudaDevices = 64;

// Ordinal of the calling thread's current CUDA device (cudaGetDevice). A failed query aborts
// through CUDA_CHECK. Defined in core/device.cu so that this header does not depend on CUDA.
struct CurrentCudaDevice {
    int operator()() const;
};

// A value computed once per CUDA device, then cached.
//
// Kernel attributes (cudaFuncSetAttribute, cudaFuncSetCacheConfig) are set on one device, and
// facts such as the SM count or an occupancy figure describe one device. A function-local `static`
// initialized once per process is therefore wrong as soon as the same launcher runs on a second
// device. Declare one function-local `static PerDeviceOnce<T>` per kernel or per fact instead:
// get() runs `init` the first time it is called while a given device is current, and returns that
// device's value on every later call. After that first call, get() costs one device-ordinal query
// and one acquire load: no lock and no allocation.
//
// `init` runs at most once per device and object, even under concurrent calls. If it throws,
// nothing is cached and the next call runs it again. DeviceOrdinal is a default-constructible
// callable returning the current ordinal; host tests substitute a simulated one.
template <typename T, typename DeviceOrdinal = CurrentCudaDevice>
class PerDeviceOnce {
public:
    template <typename Init>
    const T& get(Init&& init) {
        const int device = DeviceOrdinal{}();
        if (device < 0 || device >= kMaxCudaDevices) {
            throw std::out_of_range("PerDeviceOnce: CUDA device ordinal " + std::to_string(device) +
                                    " is outside [0, " + std::to_string(kMaxCudaDevices) + ")");
        }
        Slot& slot = slots_[static_cast<std::size_t>(device)];
        if (!slot.ready.load(std::memory_order_acquire)) {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!slot.ready.load(std::memory_order_relaxed)) {
                slot.value = std::forward<Init>(init)();
                slot.ready.store(true, std::memory_order_release);
            }
        }
        return slot.value;
    }

private:
    struct Slot {
        std::atomic<bool> ready{false};
        T value{};
    };

    std::mutex mutex_;
    std::array<Slot, kMaxCudaDevices> slots_{};
};

} // namespace ninfer
