#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <cstdint>

#include <sys/stat.h>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(AlignedHostMemoryTest, AllocatesAlignedWritablePinnedMemory) {
    constexpr size_t kUsableBytes = 8192;
    constexpr size_t kAlignment   = 4096;

    AlignedHostMemory memory(kUsableBytes, kAlignment, "test aligned host memory");

    ASSERT_NE(memory.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(memory.data()) % kAlignment, 0);
    const auto tensor = torch::from_blob(memory.data(), {kUsableBytes}, torch::TensorOptions().dtype(torch::kUInt8));
    EXPECT_TRUE(tensor.is_pinned());

    memory.data()[0]                = 0x12;
    memory.data()[kUsableBytes - 1] = 0x34;
    EXPECT_EQ(memory.data()[0], 0x12);
    EXPECT_EQ(memory.data()[kUsableBytes - 1], 0x34);
}

#if USING_CUDA && defined(__linux__)
TEST(AlignedHostMemoryTest, SharedHostMemoryUsesMemfdAndSharedMapping) {
    constexpr size_t kUsableBytes = 8192;
    constexpr size_t kAlignment   = 4096;

    AlignedHostMemory memory(kUsableBytes, kAlignment, "test shared host memory", true);

    ASSERT_NE(memory.data(), nullptr);
    ASSERT_GE(memory.fd(), 0);
    EXPECT_EQ(memory.size(), kUsableBytes);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(memory.data()) % kAlignment, 0);
    struct stat file_stat {};
    ASSERT_EQ(fstat(memory.fd(), &file_stat), 0);
    EXPECT_EQ(static_cast<size_t>(file_stat.st_size), kUsableBytes);

    memory.data()[0]                = 0x12;
    memory.data()[kUsableBytes - 1] = 0x34;
    EXPECT_EQ(memory.data()[0], 0x12);
    EXPECT_EQ(memory.data()[kUsableBytes - 1], 0x34);
}
#endif

}  // namespace
}  // namespace rtp_llm
