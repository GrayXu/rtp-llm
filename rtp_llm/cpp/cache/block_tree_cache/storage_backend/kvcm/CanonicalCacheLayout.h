#pragma once

#include <map>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::kvcm {

std::string genCanonicalSpecName(int shard, const std::string& group_name);

// One logical KV head per shard; MLA is one replicated latent component.
// LINEAR uses one key-head group (its associated value heads, SSM and Q/K/V history).
// Wire order is layer, K, V, K-scale, V-scale, independent of local TP/CP.
class CanonicalCacheLayout {
public:
    CanonicalCacheLayout(const CacheConfig& config, const ParallelismConfig& parallelism);

    const std::map<std::string, int>& shardCounts() const {
        return shard_counts_;
    }
    size_t             shardBytes(const std::string& tag) const;
    const std::string& fingerprint() const {
        return fingerprint_;
    }
    size_t blocksPerKey() const {
        return cp_size_;
    }
    std::vector<int> ranks(const std::string& tag, int shard, size_t global_block, bool write) const;
    std::vector<BlockInfo>
    slice(int layer, const std::string& tag, int shard, const std::vector<BlockInfo>& buffers) const;

private:
    struct Layer {
        int    id;
        int    local_shards;
        size_t kv_bytes;
        size_t scale_bytes;
        size_t k_bytes;
        size_t v_bytes;
        size_t k_scale_bytes;
        size_t v_scale_bytes;
        size_t pages;
        size_t history;
        size_t value_head_ratio;
        bool   mla;
        bool   linear;
    };
    struct Group {
        int                partitions;
        size_t             bytes = 0;
        CacheGroupType     type;
        std::vector<Layer> layers;
    };
    std::map<std::string, Group> groups_;
    std::map<std::string, int>   shard_counts_;
    std::string                  fingerprint_;
    int                          workers_;
    int                          rank_;
    size_t                       cp_size_;
};

}  // namespace rtp_llm::kvcm
