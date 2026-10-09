#pragma once

#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {

inline CacheConfig makeHeterogeneousRemoteCacheConfig() {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.layer_num          = 4;
    config.seq_size_per_block = 8;
    config.linear_step        = 1;
    // Interleaved global layers and two tags on layer zero exercise group-local
    // layer offsets. The INT8 pool also carries independent scale buffers.
    config.fromGroupedSpecs({makeMhaSpec("full0", 8, DataType::TYPE_FP16, 1, 2),
                             makeMhaSpec("full1", 8, DataType::TYPE_INT8, 2, 3),
                             makeLinearSpec("linear0", 8, DataType::TYPE_FP16, 1, 2)},
                            {{0, 2}, {0, 3}, {1}},
                            {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::LINEAR},
                            {"full0", "full1", "linear0"});
    return finalizeCacheConfig(std::move(config), 16);
}

}  // namespace rtp_llm::test
