#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"

#include <gtest/gtest.h>
#include <future>
#include <chrono>
#include <atomic>

namespace rtp_llm {

TEST(WorkerCacheIOFenceTest, DrainingTransferBlocksForwardAndReplacement) {
    WorkerCacheIOFence fence;
    auto               transfer = fence.lockTransfer({{"kv", 3, 1}});
    std::promise<void> entered;
    auto               entered_future = entered.get_future();
    auto               compute        = std::async(std::launch::async, [&] {
        entered.set_value();
        fence.observe({{"kv", 3, 2}});
        auto lease = fence.lockCompute();
        return true;
    });
    entered_future.wait();
    EXPECT_EQ(compute.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    transfer.unlock();
    EXPECT_TRUE(compute.get());
    EXPECT_THROW(fence.lockTransfer({{"kv", 3, 1}}), std::runtime_error);
    EXPECT_NO_THROW(fence.lockTransfer({{"kv", 3, 2}}));
}

TEST(WorkerCacheIOFenceTest, CompletedGpuProducerIsWaitedBeforeSdkAccess) {
    WorkerCacheIOFence fence;
    std::atomic<int>   waited{0};
    {
        auto compute = fence.lockCompute();
        fence.setGpuCompletion([&] { ++waited; });
    }
    auto transfer = fence.lockTransfer({{"kv", 1, 1}});
    fence.waitGpuCompletion();
    EXPECT_EQ(waited.load(), 1);
    fence.waitGpuCompletion();
    EXPECT_EQ(waited.load(), 1);
}

TEST(WorkerCacheIOFenceTest, ComputeTimeoutFailsClosedWithoutReleasingDrainingSpan) {
    WorkerCacheIOFence fence(std::chrono::milliseconds(20));
    auto               transfer = fence.lockTransfer({{"kv", 3, 1}});
    auto               compute  = std::async(std::launch::async, [&] {
        try {
            auto lease = fence.lockCompute();
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    });
    ASSERT_EQ(compute.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(compute.get());
    EXPECT_TRUE(transfer.owns_lock());
    transfer.unlock();
    EXPECT_NO_THROW(fence.lockCompute());
}

TEST(WorkerCacheIOFenceTest, RejectedBatchDoesNotAdvanceOtherGenerations) {
    WorkerCacheIOFence fence;
    fence.observe({{"kv", 1, 2}, {"state", 2, 1}});
    EXPECT_THROW(fence.lockTransfer({{"state", 2, 3}, {"kv", 1, 1}}), std::runtime_error);
    EXPECT_NO_THROW(fence.lockTransfer({{"state", 2, 1}}));
}

TEST(WorkerCacheIOFenceTest, HostAndDeviceAllocationsHaveIndependentGenerations) {
    WorkerCacheIOFence fence;
    fence.observe({{"kv", 1, 9}});
    {
        auto host = fence.lockTransfer({{"kv", 1, 1, true}});
        EXPECT_TRUE(host.owns_lock());
    }
    EXPECT_NO_THROW(fence.lockTransfer({{"kv", 1, 9}}));
    {
        auto host = fence.lockTransfer({{"kv", 1, 2, true}});
        EXPECT_TRUE(host.owns_lock());
    }
    EXPECT_THROW(fence.lockTransfer({{"kv", 1, 1, true}}), std::runtime_error);
    EXPECT_NO_THROW(fence.lockTransfer({{"kv", 1, 9}}));
}

}  // namespace rtp_llm
