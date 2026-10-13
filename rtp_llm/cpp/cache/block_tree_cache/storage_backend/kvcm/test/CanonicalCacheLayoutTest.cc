#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CanonicalCacheLayout.h"

#include <gtest/gtest.h>
#include <cstring>
#include <map>
#include <numeric>

namespace rtp_llm::kvcm {
namespace {

ParallelismConfig topology(int workers, int rank, bool cp = false) {
    ParallelismConfig result;
    result.tp_size = workers;
    result.tp_rank = rank;
    if (cp) {
        result.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
        result.prefill_cp_config.kv_cache_sharded = true;
    }
    return result;
}

CacheConfig configFor(const ParallelismConfig& parallelism,
                      int                      heads      = 8,
                      DataType                 dtype      = TYPE_INT8,
                      int                      kernel     = 4,
                      bool                     mla        = false,
                      bool                     linear     = false,
                      DataType                 ssm_dtype  = TYPE_FP32,
                      DataType                 conv_dtype = TYPE_BF16) {
    AttentionConfigs attn{};
    attn.kv_head_num   = heads;
    attn.size_per_head = 2;
    attn.kv_lora_rank  = 256;
    attn.rope_head_dim = 64;
    LinearAttentionConfig state{};
    state.linear_num_key_heads   = 4;
    state.linear_num_value_heads = 8;
    state.linear_key_head_dim    = 2;
    state.linear_value_head_dim  = 2;
    state.linear_conv_kernel_dim = 4;
    state.ssm_state_dtype        = ssm_dtype;
    state.conv_state_dtype       = conv_dtype;
    SpecBuildContext context;
    context.attn_config             = &attn;
    context.linear_attention_config = &state;
    context.parallelism_config      = &parallelism;
    context.seq_size_per_block      = 8;
    context.kernel_tokens_per_block = kernel;
    context.dtype                   = dtype;
    KVCacheSpecDesc desc;
    desc.tag        = "kv";
    desc.dtype      = dtype;
    desc.cache_type = mla ? MultiHeadLatentAttention : MultiHeadAttention;
    std::vector<KVCacheSpecPtr>   specs{SpecBuilder::build(desc, context)};
    std::vector<std::vector<int>> layers{{0, 1}};
    std::vector<CacheGroupType>   types{CacheGroupType::FULL};
    if (linear) {
        desc.tag        = "state";
        desc.cache_type = LinearAttention;
        specs.push_back(SpecBuilder::build(desc, context));
        layers.push_back({2});
        types.push_back(CacheGroupType::LINEAR);
    }
    CacheConfig result;
    result.layer_num          = linear ? 3 : 2;
    result.dtype              = dtype;
    result.use_mla            = mla;
    result.seq_size_per_block = 8;
    result.fromGroupedSpecs(specs, layers, types);
    return result;
}

struct CpuBlock {
    std::vector<char> kv;
    std::vector<char> scale;
    explicit CpuBlock(const KVCacheSpec& spec):
        kv(spec.block_size_bytes(), 0), scale(spec.scale_block_size_bytes(), 0) {}
    std::vector<BlockInfo> buffers() {
        BlockInfo data;
        data.addr       = kv.data();
        data.size_bytes = kv.size();
        std::vector<BlockInfo> result{data};
        if (!scale.empty()) {
            data.addr       = scale.data();
            data.size_bytes = scale.size();
            result.push_back(data);
        }
        return result;
    }
};

std::vector<char> gather(const std::vector<BlockInfo>& parts) {
    std::vector<char> result;
    for (const auto& part : parts) {
        const auto* data = static_cast<const char*>(part.addr);
        result.insert(result.end(), data, data + part.size_bytes);
    }
    return result;
}
void scatter(const std::vector<char>& bytes, const std::vector<BlockInfo>& parts) {
    size_t offset = 0;
    for (const auto& part : parts) {
        ASSERT_LE(part.size_bytes, bytes.size() - offset);
        std::memcpy(part.addr, bytes.data() + offset, part.size_bytes);
        offset += part.size_bytes;
    }
    ASSERT_EQ(offset, bytes.size());
}

// Independent HND fixture: unique bytes per head, token, component and layer.
// Kernel page size is deliberately different between producer and consumer.
void fillMha(CpuBlock& block, const KVCacheSpec& spec, int global_head_begin, int layer) {
    const size_t heads       = spec.local_kv_head_num;
    const size_t tokens      = spec.seq_size_per_block;
    const size_t page_tokens = spec.kernel_seq_size_per_block;
    const size_t pages       = tokens / page_tokens;
    auto         fill        = [&](std::vector<char>& bytes, size_t head_token_bytes, int component_base) {
        for (size_t page = 0; page < pages; ++page) {
            for (size_t component = 0; component < 2; ++component) {
                for (size_t head = 0; head < heads; ++head) {
                    for (size_t token = 0; token < page_tokens; ++token) {
                        const size_t offset =
                            (((page * 2 + component) * heads + head) * page_tokens + token) * head_token_bytes;
                        for (size_t byte = 0; byte < head_token_bytes; ++byte) {
                            bytes[offset + byte] = static_cast<char>(layer * 29 + component_base + component * 61
                                                                     + (global_head_begin + head) * 7
                                                                     + (page * page_tokens + token) * 3 + byte);
                        }
                    }
                }
            }
        }
    };
    fill(block.kv, spec.k_block_size_bytes() / heads / tokens, 0);
    if (!block.scale.empty()) {
        fill(block.scale, spec.k_scale_block_size_bytes() / heads / tokens, 31);
    }
}

TEST(CanonicalCacheLayoutTest, MhaKVAndScaleRoundTripAcrossTp1Tp2Tp4IncludingGqaReplicas) {
    for (int global_heads : {1, 2, 8}) {
        for (DataType dtype : {TYPE_FP16, TYPE_INT8, TYPE_FP8_E4M3}) {
            for (int producer : {1, 2, 4}) {
                std::map<std::pair<int, int>, std::vector<char>> wire;
                for (int rank = 0; rank < producer; ++rank) {
                    const auto           p      = topology(producer, rank);
                    const auto           config = configFor(p, global_heads, dtype, 4);
                    CanonicalCacheLayout layout(config, p);
                    const auto&          spec       = *config.group("kv").spec;
                    const int            partitions = std::gcd(global_heads, producer);
                    const int            first      = rank / (producer / partitions) * spec.local_kv_head_num;
                    for (int layer = 0; layer < 2; ++layer) {
                        CpuBlock block(spec);
                        fillMha(block, spec, first, layer);
                        for (int head = 0; head < global_heads; ++head) {
                            const auto owners = layout.ranks("kv", head, 0, true);
                            if (owners.front() == rank) {
                                EXPECT_TRUE(wire.emplace(std::make_pair(layer, head),
                                                         gather(layout.slice(layer, "kv", head, block.buffers())))
                                                .second);
                            }
                        }
                    }
                }
                ASSERT_EQ(wire.size(), static_cast<size_t>(2 * global_heads));
                for (int consumer : {1, 2, 4}) {
                    for (int rank = 0; rank < consumer; ++rank) {
                        const auto           p      = topology(consumer, rank);
                        const auto           config = configFor(p, global_heads, dtype, 8);
                        CanonicalCacheLayout layout(config, p);
                        const auto&          spec = *config.group("kv").spec;
                        const int first = rank / (consumer / std::gcd(global_heads, consumer)) * spec.local_kv_head_num;
                        for (int layer = 0; layer < 2; ++layer) {
                            CpuBlock actual(spec), expected(spec);
                            fillMha(expected, spec, first, layer);
                            for (int head = first; head < first + static_cast<int>(spec.local_kv_head_num); ++head) {
                                scatter(wire.at({layer, head}), layout.slice(layer, "kv", head, actual.buffers()));
                            }
                            EXPECT_EQ(actual.kv, expected.kv);
                            EXPECT_EQ(actual.scale, expected.scale);
                        }
                    }
                }
            }
        }
    }
}

TEST(CanonicalCacheLayoutTest, FullBlocksRoundTripBetweenLegalTpAndShardedCpTopologies) {
    for (bool source_cp : {false, true}) {
        for (int source_workers : {1, 2, 4}) {
            std::map<std::pair<int, int>, std::vector<char>> wire;
            for (int global_block = 0; global_block < 9; ++global_block) {
                for (int rank = 0; rank < source_workers; ++rank) {
                    const auto           p      = topology(source_workers, rank, source_cp);
                    const auto           config = configFor(p);
                    CanonicalCacheLayout layout(config, p);
                    const auto&          spec  = *config.group("kv").spec;
                    const int            first = source_cp ? 0 : rank * spec.local_kv_head_num;
                    CpuBlock             source(spec);
                    fillMha(source, spec, first, global_block);
                    for (int head = 0; head < 8; ++head) {
                        if (layout.ranks("kv", head, global_block, true).front() == rank) {
                            ASSERT_TRUE(wire.emplace(std::make_pair(global_block, head),
                                                     gather(layout.slice(0, "kv", head, source.buffers())))
                                            .second);
                        }
                    }
                }
            }
            ASSERT_EQ(wire.size(), 9u * 8);
            for (bool target_cp : {false, true}) {
                for (int target_workers : {1, 2, 4}) {
                    for (int global_block = 0; global_block < 9; ++global_block) {
                        for (int rank = 0; rank < target_workers; ++rank) {
                            if (target_cp && global_block % target_workers != rank) {
                                continue;
                            }
                            const auto           p      = topology(target_workers, rank, target_cp);
                            const auto           config = configFor(p, 8, TYPE_INT8, 8);
                            CanonicalCacheLayout layout(config, p);
                            const auto&          spec  = *config.group("kv").spec;
                            const int            first = target_cp ? 0 : rank * spec.local_kv_head_num;
                            CpuBlock             actual(spec), expected(spec);
                            fillMha(expected, spec, first, global_block);
                            for (int head = first; head < first + static_cast<int>(spec.local_kv_head_num); ++head) {
                                scatter(wire.at({global_block, head}), layout.slice(0, "kv", head, actual.buffers()));
                            }
                            EXPECT_EQ(actual.kv, expected.kv);
                            EXPECT_EQ(actual.scale, expected.scale);
                        }
                    }
                }
            }
        }
    }
}

TEST(CanonicalCacheLayoutTest, NamespaceGeometryIsIndependentOfTpCpAndKernelPageSize) {
    const auto p1       = topology(1, 0);
    const auto base     = configFor(p1);
    const auto expected = CanonicalCacheLayout(base, p1).fingerprint();
    for (bool cp : {false, true}) {
        for (int workers : {1, 2, 4}) {
            const auto           p      = topology(workers, 0, cp);
            const auto           config = configFor(p, 8, TYPE_INT8, 8);
            CanonicalCacheLayout layout(config, p);
            EXPECT_EQ(layout.fingerprint(), expected);
            EXPECT_EQ(layout.shardCounts().at("kv"), 8);
            EXPECT_EQ(layout.shardBytes("kv"), 2u * (2 * 8 * 2 + 2 * 8 * sizeof(float)));
        }
    }
    const auto bf16 = configFor(p1, 8, TYPE_BF16);
    EXPECT_NE(CanonicalCacheLayout(bf16, p1).fingerprint(), expected);
    const auto gqa = configFor(p1, 2);
    EXPECT_NE(CanonicalCacheLayout(gqa, p1).fingerprint(), expected);
    EXPECT_EQ(genCanonicalSpecName(0, "Fkv"), "v1_Fkv_h0");
    EXPECT_NE(genCanonicalSpecName(0, "Fkv"), "tp0_Fkv");
}

TEST(CanonicalCacheLayoutTest, ShardedCpUsesGlobalPageOwnersAndReplicatedLinearEndpoints) {
    for (int workers : {2, 4}) {
        const auto           p      = topology(workers, 0, true);
        const auto           config = configFor(p, 8, TYPE_BF16, 8, false, true);
        CanonicalCacheLayout layout(config, p);
        EXPECT_EQ(layout.blocksPerKey(), static_cast<size_t>(workers));
        for (size_t block = 0; block < 9; ++block) {
            EXPECT_EQ(layout.ranks("kv", 3, block, true), std::vector<int>{static_cast<int>(block % workers)});
            EXPECT_EQ(layout.ranks("kv", 3, block, false), std::vector<int>{static_cast<int>(block % workers)});
            EXPECT_EQ(layout.ranks("state", 2, block, true), std::vector<int>{0});
            auto readers = layout.ranks("state", 2, block, false);
            EXPECT_EQ(readers.size(), static_cast<size_t>(workers));
        }
    }
}

TEST(CanonicalCacheLayoutTest, LinearSsmAndConvolutionComponentsRoundTripAcrossTpAndCp) {
    std::map<int, std::vector<char>> wire;
    const auto                       p1   = topology(1, 0);
    const auto                       base = configFor(p1, 8, TYPE_BF16, 8, false, true);
    const auto&                      spec = *base.group("state").spec;
    CpuBlock                         original(spec);
    for (size_t i = 0; i < original.kv.size(); ++i) {
        original.kv[i] = static_cast<char>(i * 13 + 7);
    }
    CanonicalCacheLayout writer(base, p1);
    for (int shard = 0; shard < 4; ++shard) {
        wire[shard] = gather(writer.slice(2, "state", shard, original.buffers()));
    }
    for (bool cp : {false, true}) {
        for (int workers : {1, 2, 4}) {
            for (int rank = 0; rank < workers; ++rank) {
                const auto           p      = topology(workers, rank, cp);
                const auto           config = configFor(p, 8, TYPE_BF16, 8, false, true);
                CanonicalCacheLayout reader(config, p);
                EXPECT_EQ(reader.fingerprint(), writer.fingerprint());
                CpuBlock  actual(*config.group("state").spec);
                const int first = cp ? 0 : rank * (4 / workers);
                const int count = cp ? 4 : 4 / workers;
                for (int shard = first; shard < first + count; ++shard) {
                    scatter(wire.at(shard), reader.slice(2, "state", shard, actual.buffers()));
                    EXPECT_EQ(gather(reader.slice(2, "state", shard, actual.buffers())), wire.at(shard));
                }
                CpuBlock expected(*config.group("state").spec);
                // Oracle from the global typed SSM[value_head,V,K] and Conv[history,Q|K|V] shapes.
                const size_t ssm_per_key     = 2 * 2 * 2 * sizeof(float);
                const size_t channel_per_key = 2 * sizeof(uint16_t);
                const size_t global_qk       = 4 * channel_per_key;
                const size_t global_row      = global_qk * 4;
                const size_t local_ssm       = count * ssm_per_key;
                std::memcpy(expected.kv.data(), original.kv.data() + first * ssm_per_key, local_ssm);
                for (size_t row = 0; row < 3; ++row) {
                    const size_t src = 4 * ssm_per_key + row * global_row;
                    const size_t dst = local_ssm + row * count * channel_per_key * 4;
                    std::memcpy(expected.kv.data() + dst,
                                original.kv.data() + src + first * channel_per_key,
                                count * channel_per_key);
                    std::memcpy(expected.kv.data() + dst + count * channel_per_key,
                                original.kv.data() + src + global_qk + first * channel_per_key,
                                count * channel_per_key);
                    std::memcpy(expected.kv.data() + dst + 2 * count * channel_per_key,
                                original.kv.data() + src + 2 * global_qk + first * 2 * channel_per_key,
                                count * 2 * channel_per_key);
                }
                EXPECT_EQ(actual.kv, expected.kv);
                // Reconstruct TP1 to verify the SSM and every Q/K/V history channel.
                CpuBlock roundtrip(spec);
                for (int shard = first; shard < first + count; ++shard) {
                    scatter(gather(reader.slice(2, "state", shard, actual.buffers())),
                            writer.slice(2, "state", shard, roundtrip.buffers()));
                }
                for (int shard = first; shard < first + count; ++shard) {
                    EXPECT_EQ(gather(writer.slice(2, "state", shard, roundtrip.buffers())), wire.at(shard));
                }
            }
        }
    }
}

TEST(CanonicalCacheLayoutTest, MlaLatentRecordIsReplicatedAndKeepsEmbeddedScales) {
    for (DataType dtype : {TYPE_BF16, TYPE_FP8_E4M3}) {
        const auto           p1   = topology(1, 0);
        const auto           base = configFor(p1, 1, dtype, 8, true);
        CanonicalCacheLayout writer(base, p1);
        CpuBlock             original(*base.group("kv").spec);
        std::iota(original.kv.begin(), original.kv.end(), char{0});
        for (bool cp : {false, true}) {
            for (int workers : {2, 4}) {
                for (int rank = 0; rank < workers; ++rank) {
                    const auto           p      = topology(workers, rank, cp);
                    const auto           config = configFor(p, 1, dtype, 8, true);
                    CanonicalCacheLayout reader(config, p);
                    EXPECT_EQ(reader.fingerprint(), writer.fingerprint());
                    CpuBlock actual(*config.group("kv").spec);
                    scatter(original.kv, reader.slice(0, "kv", 0, actual.buffers()));
                    EXPECT_EQ(actual.kv, original.kv);
                }
            }
        }
    }
}

TEST(CanonicalCacheLayoutTest, SameWidthLinearComponentDTypesUseDifferentNamespaces) {
    const auto           p         = topology(1, 0);
    const auto           base      = configFor(p, 8, TYPE_BF16, 8, false, true, TYPE_BF16, TYPE_BF16);
    const auto           ssm_fp16  = configFor(p, 8, TYPE_BF16, 8, false, true, TYPE_FP16, TYPE_BF16);
    const auto           conv_fp16 = configFor(p, 8, TYPE_BF16, 8, false, true, TYPE_BF16, TYPE_FP16);
    CanonicalCacheLayout expected(base, p), ssm_changed(ssm_fp16, p), conv_changed(conv_fp16, p);
    EXPECT_EQ(expected.shardBytes("state"), ssm_changed.shardBytes("state"));
    EXPECT_EQ(expected.shardBytes("state"), conv_changed.shardBytes("state"));
    EXPECT_NE(expected.fingerprint(), ssm_changed.fingerprint());
    EXPECT_NE(expected.fingerprint(), conv_changed.fingerprint());
}

TEST(CanonicalCacheLayoutTest, NamespaceDTypesArePrintableForBf16AndLinearState) {
    const auto p           = topology(1, 0);
    const auto config      = configFor(p, 8, TYPE_BF16, 8, false, true, TYPE_BF16, TYPE_BF16);
    const auto fingerprint = CanonicalCacheLayout(config, p).fingerprint();
    for (const unsigned char byte : fingerprint) {
        EXPECT_GE(byte, 0x20);
        EXPECT_LE(byte, 0x7e);
    }
    EXPECT_NE(fingerprint.find(";layer=0,0,14,"), std::string::npos);
}

TEST(CanonicalCacheLayoutTest, RejectsUnknownGeometryInvalidRpcHeadAndShortBuffers) {
    const auto           p      = topology(2, 0);
    const auto           config = configFor(p);
    CanonicalCacheLayout layout(config, p);
    CpuBlock             block(*config.group("kv").spec);
    EXPECT_THROW(layout.slice(0, "kv", 7, block.buffers()), std::invalid_argument);
    auto buffers = block.buffers();
    --buffers.front().size_bytes;
    EXPECT_THROW(layout.slice(0, "kv", 0, buffers), std::invalid_argument);
    auto groups                 = config.groups();
    auto invalid                = groups.front().spec->clone();
    invalid->global_kv_head_num = 0;
    groups.front().spec         = invalid;
    CacheConfig unknown         = config;
    unknown.setTopology(groups, config.topology().layers());
    EXPECT_THROW(CanonicalCacheLayout(unknown, p), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm::kvcm
