#pragma once

#include <cstdint>
#include <functional>
#include <chrono>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rtp_llm {

struct CacheBlockGeneration {
    std::string tag;
    int32_t     block;
    uint64_t    generation;
    bool        host_source{false};
};

// Non-allocator ranks mirror TP0's physical block IDs. A local allocation
// refcount cannot protect these spans. Serialize their cache access with SDK
// drain and reject RPCs queued for a block allocation that has been replaced.
class WorkerCacheIOFence {
public:
    using Lease = std::unique_lock<std::recursive_timed_mutex>;
    explicit WorkerCacheIOFence(std::chrono::milliseconds wait_budget = std::chrono::seconds(60)):
        wait_budget_(wait_budget) {}

    Lease lockCompute() {
        Lease lease(mutex_, std::defer_lock);
        if (!lease.try_lock_for(wait_budget_)) {
            // Never bypass an outstanding SDK drain. RPC callers report a
            // failure; compute callers fail closed through the engine error path.
            throw std::runtime_error("worker cache I/O fence wait timed out; SDK drain still owns the span");
        }
        return lease;
    }
    void observe(const std::vector<CacheBlockGeneration>& blocks) {
        auto lease = lockCompute();
        for (const auto& block : blocks) {
            auto& current = generations_[{block.tag, block.block, block.host_source}];
            if (block.generation < current) {
                throw std::runtime_error("worker received stale cache allocation generations");
            }
            current = block.generation;
        }
    }
    Lease lockTransfer(const std::vector<CacheBlockGeneration>& blocks) {
        auto lease = lockCompute();
        // Validate the complete batch before advancing any generation.
        for (const auto& block : blocks) {
            const auto found = generations_.find({block.tag, block.block, block.host_source});
            if (block.generation == 0 || (found != generations_.end() && block.generation < found->second)) {
                throw std::runtime_error("worker cache RPC refers to a replaced allocation");
            }
        }
        for (const auto& block : blocks) {
            generations_[{block.tag, block.block, block.host_source}] = block.generation;
        }
        return lease;
    }
    // Caller holds a compute/transfer lease. The SDK waits only for the cache
    // producer event, rather than synchronizing unrelated NCCL collectives.
    void setGpuCompletion(std::function<void()> completion) {
        gpu_completion_ = std::move(completion);
    }
    void waitGpuCompletion() {
        if (gpu_completion_) {
            gpu_completion_();
            gpu_completion_ = {};
        }
    }

private:
    std::recursive_timed_mutex                                 mutex_;
    std::chrono::milliseconds                                  wait_budget_;
    std::map<std::tuple<std::string, int32_t, bool>, uint64_t> generations_;
    std::function<void()>                                      gpu_completion_;
};

}  // namespace rtp_llm
