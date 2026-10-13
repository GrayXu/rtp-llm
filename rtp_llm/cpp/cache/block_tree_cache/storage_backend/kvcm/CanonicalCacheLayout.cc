#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CanonicalCacheLayout.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rtp_llm::kvcm {
namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void appendSlice(std::vector<BlockInfo>& out, const BlockInfo& source, size_t offset, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    require(source.addr != nullptr && offset <= source.size_bytes && bytes <= source.size_bytes - offset,
            "canonical_v1 slice exceeds the physical component");
    auto part       = source;
    part.addr       = static_cast<char*>(source.addr) + offset;
    part.size_bytes = bytes;
    out.push_back(part);
}
}  // namespace

std::string genCanonicalSpecName(int shard, const std::string& group_name) {
    return "v1_" + group_name + "_h" + std::to_string(shard);
}

CanonicalCacheLayout::CanonicalCacheLayout(const CacheConfig& config, const ParallelismConfig& parallelism):
    workers_(static_cast<int>(parallelism.tp_size)),
    rank_(static_cast<int>(parallelism.tp_rank)),
    cp_size_(parallelism.prefill_cp_config.kv_cache_sharded && parallelism.prefill_cp_config.is_enabled() ?
                 static_cast<size_t>(parallelism.tp_size) :
                 1) {
    require(parallelism.tp_size > 0 && parallelism.tp_size <= std::numeric_limits<int>::max()
                && parallelism.tp_rank >= 0 && parallelism.tp_rank < parallelism.tp_size && parallelism.pp_size == 1,
            "canonical_v1 requires valid worker ranks and pp_size=1");
    const int attn_tp = static_cast<int>(parallelism.get_attn_tp_size());
    require(!parallelism.prefill_cp_config.kv_cache_sharded || parallelism.prefill_cp_config.is_enabled()
                || parallelism.prefill_cp_config.is_prefill_enabled(),
            "canonical_v1 sharded CP requires an enabled prefill method or a decode producer hint");
    require(attn_tp == 1 || attn_tp == workers_, "canonical_v1 does not support orthogonal TP x CP execution");
    std::ostringstream identity;
    identity << "canonical_v1;block=" << config.seq_size_per_block;
    for (const auto& group : config.groups()) {
        require(group.policy.memory_placement == CacheMemoryPlacement::DEVICE,
                "canonical_v1 requires DEVICE cache groups");
        require(cp_size_ == 1 || group.policy.group_type != CacheGroupType::SWA,
                "canonical_v1 does not support compact CP SWA rings");
        require(cp_size_ == 1
                    || (group.policy.group_type == CacheGroupType::FULL ?
                            group.policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN :
                            group.policy.cp_mapping == CpBlockMappingMode::NONE),
                "canonical_v1 requires FULL round-robin CP and replicated LINEAR states");
        Group result{};
        result.type = group.policy.group_type;
        int count   = 0;
        identity << ";tag=" << group.tag << ";policy=" << static_cast<int>(group.policy.group_type) << ','
                 << group.policy.enable_prefix_reuse << ',' << group.policy.sliding_window_size;
        for (int layer_id : config.layerIdsForGroup(group.tag)) {
            const auto& physical = config.physicalGroupForLayer(layer_id, group.tag);
            const auto& spec     = *physical.spec;
            require(spec.seq_size_per_block == config.seq_size_per_block
                        && spec.cacheKeyTokenStride() == config.seq_size_per_block,
                    "canonical_v1 requires one fixed token block per group cache key");
            const bool  mha    = spec.type == MultiHeadAttention;
            const bool  mla    = spec.type == MultiHeadLatentAttention;
            const auto* linear = dynamic_cast<const LinearKVCacheSpec*>(&spec);
            require(mha || mla || linear, "canonical_v1 has no logical layout for this opaque cache component");
            require((linear != nullptr) == (group.policy.group_type == CacheGroupType::LINEAR),
                    "canonical_v1 cache policy does not match the state component");
            int    shards     = 1;
            int    partitions = 1;
            size_t history    = 0;
            if (mha) {
                require(spec.global_kv_head_num > 0, "canonical_v1 needs the model's global KV head count");
                shards     = static_cast<int>(spec.global_kv_head_num);
                partitions = std::gcd(shards, attn_tp);
                require(spec.local_kv_head_num == static_cast<uint32_t>(shards / partitions),
                        "canonical_v1 local MHA heads disagree with the weight loader partition");
            } else if (linear) {
                require(linear->global_key_heads > 0 && linear->global_kv_head_num > 0 && linear->conv_kernel > 0,
                        "canonical_v1 needs resolved LINEAR component geometry");
                shards     = static_cast<int>(linear->global_key_heads);
                partitions = attn_tp;
                history    = linear->conv_kernel;
                require(shards % attn_tp == 0 && spec.global_kv_head_num % attn_tp == 0
                            && spec.global_kv_head_num % shards == 0
                            && spec.local_kv_head_num == spec.global_kv_head_num / attn_tp,
                        "canonical_v1 LINEAR heads disagree with the execution layout");
            }
            require(count == 0 || (count == shards && result.partitions == partitions),
                    "canonical_v1 group layers disagree on logical head partitions");
            count             = shards;
            result.partitions = partitions;
            Layer layer{layer_id,
                        shards / partitions,
                        physical.kvBlockStrideBytes(),
                        physical.kvScaleStrideBytes(),
                        spec.k_block_size_bytes(),
                        spec.v_block_size_bytes(),
                        spec.k_scale_block_size_bytes(),
                        spec.v_scale_block_size_bytes(),
                        physical.kernelBlocksPerKvBlock(),
                        history,
                        linear ? spec.global_kv_head_num / linear->global_key_heads : 1,
                        mla,
                        linear != nullptr};
            require(layer.kv_bytes == spec.block_size_bytes() && layer.scale_bytes == spec.scale_block_size_bytes(),
                    "canonical_v1 physical stride disagrees with the model spec");
            if (!mla) {
                require(layer.k_bytes + layer.v_bytes == layer.kv_bytes
                            && layer.k_scale_bytes + layer.v_scale_bytes == layer.scale_bytes,
                        "canonical_v1 requires exact K/V and scale component bounds");
                require(layer.kv_bytes % layer.local_shards == 0 && layer.scale_bytes % layer.local_shards == 0,
                        "canonical_v1 component bytes must divide into logical head groups");
            }
            const size_t bytes = (layer.kv_bytes + layer.scale_bytes) / layer.local_shards;
            require(bytes > 0 && bytes <= std::numeric_limits<size_t>::max() - result.bytes,
                    "canonical_v1 shard byte size overflow");
            result.bytes += bytes;
            result.layers.push_back(layer);
            identity << ";layer=" << layer_id << ',' << static_cast<int>(spec.type) << ','
                     << static_cast<int>(spec.memoryLayoutDType()) << ',' << shards << ',' << bytes << ','
                     << layer.k_bytes / layer.local_shards << ',' << layer.v_bytes / layer.local_shards << ','
                     << layer.k_scale_bytes / layer.local_shards << ',' << layer.v_scale_bytes / layer.local_shards;
            if (linear) {
                identity << ',' << linear->global_kv_head_num << ',' << linear->head_dim << ',' << history << ','
                         << static_cast<int>(linear->ssmStateDType()) << ','
                         << static_cast<int>(linear->convStateDType());
            }
        }
        require(count > 0, "canonical_v1 group has no layers");
        shard_counts_.emplace(group.tag, count);
        groups_.emplace(group.tag, std::move(result));
    }
    fingerprint_ = identity.str();
}

size_t CanonicalCacheLayout::shardBytes(const std::string& tag) const {
    return groups_.at(tag).bytes;
}

std::vector<int> CanonicalCacheLayout::ranks(const std::string& tag, int shard, size_t global_block, bool write) const {
    const auto& group = groups_.at(tag);
    require(shard >= 0 && shard < shard_counts_.at(tag), "canonical_v1 logical shard out of range");
    if (cp_size_ > 1 && group.type == CacheGroupType::FULL) {
        return {static_cast<int>(global_block % cp_size_)};
    }
    // Weight loader sp_head uses rank / (TP / gcd(global_heads, TP)).
    // The first replica owns writes; readers populate every local replica.
    const int        replicas = workers_ / group.partitions;
    const int        first    = shard / (shard_counts_.at(tag) / group.partitions) * replicas;
    std::vector<int> result;
    for (int rank = first; rank < first + (write ? 1 : replicas); ++rank) {
        result.push_back(rank);
    }
    return result;
}

std::vector<BlockInfo> CanonicalCacheLayout::slice(int                           layer_id,
                                                   const std::string&            tag,
                                                   int                           shard,
                                                   const std::vector<BlockInfo>& buffers) const {
    const auto& group = groups_.at(tag);
    const auto  found = std::find_if(
        group.layers.begin(), group.layers.end(), [layer_id](const Layer& layer) { return layer.id == layer_id; });
    require(found != group.layers.end(), "canonical_v1 unknown group layer");
    const auto& layer       = *found;
    const int   replicas    = workers_ / group.partitions;
    const int   first_shard = rank_ / replicas * layer.local_shards;
    require(shard >= first_shard && shard < first_shard + layer.local_shards,
            "canonical_v1 RPC requests a head absent on this worker");
    require(buffers.size() == (layer.scale_bytes ? 2 : 1) && buffers[0].size_bytes == layer.kv_bytes
                && (!layer.scale_bytes || buffers[1].size_bytes == layer.scale_bytes),
            "canonical_v1 buffer resolver returned an incompatible physical block");
    std::vector<BlockInfo> result;
    const size_t           local = static_cast<size_t>(shard - first_shard);
    if (layer.mla) {
        // The token-major latent/rope/embedded-scale record is replicated, not split as MHA heads.
        for (const auto& buffer : buffers) {
            appendSlice(result, buffer, 0, buffer.size_bytes);
        }
    } else if (layer.linear) {
        appendSlice(result, buffers[0], local * layer.k_bytes / layer.local_shards, layer.k_bytes / layer.local_shards);
        // Conv is [history, Q|K|V]. Preserve all three channel components at every history position.
        const size_t row = layer.v_bytes / layer.history;
        require(layer.v_bytes % layer.history == 0 && row % (2 + layer.value_head_ratio) == 0,
                "canonical_v1 LINEAR convolution history has invalid channel geometry");
        const size_t qk = row / (2 + layer.value_head_ratio);
        require(qk % layer.local_shards == 0, "canonical_v1 LINEAR channels do not divide into key-head groups");
        const size_t qk_head = qk / layer.local_shards;
        for (size_t history = 0; history < layer.history; ++history) {
            const size_t start = layer.k_bytes + history * row;
            appendSlice(result, buffers[0], start + local * qk_head, qk_head);
            appendSlice(result, buffers[0], start + qk + local * qk_head, qk_head);
            appendSlice(result,
                        buffers[0],
                        start + 2 * qk + local * qk_head * layer.value_head_ratio,
                        qk_head * layer.value_head_ratio);
        }
    } else {
        for (size_t component = 0; component < buffers.size(); ++component) {
            const size_t k = component == 0 ? layer.k_bytes : layer.k_scale_bytes;
            const size_t v = component == 0 ? layer.v_bytes : layer.v_scale_bytes;
            require(k % (layer.pages * layer.local_shards) == 0 && v % (layer.pages * layer.local_shards) == 0,
                    "canonical_v1 kernel pages do not divide into head slices");
            // Local full-attention views contain packed kernel pages. Serialize K pages first, then V pages.
            for (bool value : {false, true}) {
                const size_t head_bytes = (value ? v : k) / layer.pages / layer.local_shards;
                for (size_t page = 0; page < layer.pages; ++page) {
                    appendSlice(result,
                                buffers[component],
                                page * (k + v) / layer.pages + (value ? k / layer.pages : 0) + local * head_bytes,
                                head_bytes);
                }
            }
        }
    }
    return result;
}

}  // namespace rtp_llm::kvcm
