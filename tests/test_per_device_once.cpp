// Host-only qualification of PerDeviceOnce: per-device keying, once-per-device initialization,
// retry after a throwing initializer, rejection of out-of-range ordinals, concurrent first use,
// and an allocation-free steady state. The current device is simulated, so no GPU is needed.
#include "core/per_device.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <latch>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0};

// Simulated current device of the calling thread, standing in for cudaGetDevice.
thread_local int t_current_device = 0;

struct SimulatedDevice {
    int operator()() const { return t_current_device; }
};

template <typename T>
using Cache = ninfer::PerDeviceOnce<T, SimulatedDevice>;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int once_per_device() {
    int failures = 0;
    Cache<int> cache;
    int calls   = 0;
    auto device = [&](int ordinal) {
        t_current_device = ordinal;
        return cache.get([&] {
            ++calls;
            return 100 + ordinal;
        });
    };

    failures += check(device(0) == 100, "device 0 did not receive its own value");
    failures += check(device(0) == 100 && calls == 1, "device 0 was initialized more than once");
    failures += check(device(1) == 101, "device 1 reused device 0's value");
    failures += check(calls == 2, "device 1 was not initialized on its first use");
    failures += check(device(0) == 100 && device(1) == 101 && calls == 2,
                      "switching back to an initialized device ran its initializer again");
    const int last = ninfer::kMaxCudaDevices - 1;
    failures += check(device(last) == 100 + last && calls == 3,
                      "the highest supported ordinal was not keyed separately");
    return failures;
}

int independent_objects() {
    Cache<int> first;
    Cache<int> second;
    t_current_device = 0;
    const int a      = first.get([] { return 1; });
    const int b      = second.get([] { return 2; });
    return check(a == 1 && b == 2, "two objects on one device shared a slot");
}

int throwing_initializer_is_retried() {
    int failures = 0;
    Cache<int> cache;
    t_current_device = 3;
    bool thrown      = false;
    try {
        (void)cache.get([]() -> int { throw std::runtime_error("setup failed"); });
    } catch (const std::runtime_error&) { thrown = true; }
    failures += check(thrown, "an initializer exception was not propagated");
    int calls       = 0;
    const int value = cache.get([&] {
        ++calls;
        return 7;
    });
    failures += check(value == 7 && calls == 1, "a failed initialization was cached");
    return failures;
}

int out_of_range_ordinals_are_rejected() {
    int failures = 0;
    for (const int ordinal : {-1, ninfer::kMaxCudaDevices}) {
        Cache<int> cache;
        t_current_device = ordinal;
        bool ran         = false;
        bool rejected    = false;
        try {
            (void)cache.get([&] {
                ran = true;
                return 0;
            });
        } catch (const std::out_of_range&) { rejected = true; }
        failures += check(rejected && !ran, "an out-of-range device ordinal was accepted");
    }
    return failures;
}

int concurrent_first_use() {
    constexpr int kDevices          = 4;
    constexpr int kThreadsPerDevice = 8;
    Cache<int> cache;
    std::atomic<int> calls[kDevices]{};
    std::atomic<int> mismatches{0};
    std::latch start(kDevices * kThreadsPerDevice);
    std::vector<std::thread> threads;
    for (int device = 0; device < kDevices; ++device) {
        for (int index = 0; index < kThreadsPerDevice; ++index) {
            threads.emplace_back([&, device] {
                t_current_device = device;
                start.arrive_and_wait();
                const int value = cache.get([&] {
                    calls[device].fetch_add(1);
                    // Holds the first caller inside the initializer so that the others are
                    // already past the unlocked check and must rely on the locked one.
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    return 1000 + device;
                });
                if (value != 1000 + device) { mismatches.fetch_add(1); }
            });
        }
    }
    for (std::thread& thread : threads) { thread.join(); }

    int failures = check(mismatches.load() == 0, "a concurrent caller saw another device's value");
    for (int device = 0; device < kDevices; ++device) {
        failures += check(calls[device].load() == 1,
                          "concurrent first use ran an initializer more than once per device");
    }
    return failures;
}

int steady_state_does_not_allocate() {
    Cache<int> cache;
    t_current_device = 2;
    (void)cache.get([] { return 5; });
    const std::size_t before = g_allocations.load();
    int sum                  = 0;
    for (int iteration = 0; iteration < 1000; ++iteration) {
        sum += cache.get([] { return 0; });
    }
    return check(sum == 5000 && g_allocations.load() == before,
                 "an initialized lookup allocated or returned a different value");
}

} // namespace

// Counting replacements for the global allocation functions, observed by
// steady_state_does_not_allocate. The sized and array forms forward to these by default.
void* operator new(std::size_t size) {
    g_allocations.fetch_add(1);
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) { return pointer; }
    throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
    int failures = 0;
    failures += once_per_device();
    failures += independent_objects();
    failures += throwing_initializer_is_retried();
    failures += out_of_range_ordinals_are_rejected();
    failures += concurrent_first_use();
    failures += steady_state_does_not_allocate();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
