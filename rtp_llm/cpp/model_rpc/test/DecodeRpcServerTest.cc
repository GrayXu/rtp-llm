#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(std::string              request_key,
                                                    std::vector<std::string> peer_addrs,
                                                    CacheKeysType            cache_keys,
                                                    GroupBlockIds            group_block_ids,
                                                    int32_t                  prefill_cp_size,
                                                    int64_t                  reuse_block_size = 0,
                                                    std::vector<CacheStoreGeometryPB> geometry = {}) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            std::move(group_block_ids),
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size,
            /*prefill_cp_sharded=*/prefill_cp_size > 1,
            std::move(geometry)};
}

GroupBase makeRpcGroup(std::string tag) {
    auto spec = std::make_shared<MHAKVCacheSpec>(tag, 8, 8, 1);

    GroupBase group;
    group.tag       = std::move(tag);
    group.spec      = std::move(spec);
    group.policy    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.block_num = 8;
    return group;
}

// Tokens per logical (cache-key sized) block, and the CP-scaled tokens per block
// of a compacted fixed/state group under prefill CP=2 (cp.scale_seq_size).
constexpr size_t kBaseSeqSizePerBlock    = 64;
constexpr size_t kCompactSeqSizePerBlock = kBaseSeqSizePerBlock * 2;

CacheGroupPolicy makeCompactStatePolicy(uint32_t active_tail_blocks) {
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.active_tail_blocks = active_tail_blocks;
    policy.cp_slice           = CpBlockSliceMode::PAYLOAD_BYTES;
    EXPECT_EQ(policy.cp_mapping, CpBlockMappingMode::COMPACT_LAST_RANK);
    return policy;
}

using KeyOffsetPairs = std::vector<std::pair<int, int>>;

KeyOffsetPairs keyOffsetPairs(const std::vector<CacheStoreBlockPair>& plan) {
    KeyOffsetPairs pairs;
    pairs.reserve(plan.size());
    for (const auto& pair : plan) {
        pairs.emplace_back(pair.key_index, pair.offset_index);
    }
    return pairs;
}

std::shared_ptr<NormalGenerateStream> makeGenerateStream(int seq_length) {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::zeros({seq_length}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = seq_length + 16;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

// Keep the published BlockBuffers, then apply the CacheStore service's existing
// contiguous partition contract to the real decode destination addresses.
class PageHandoffCacheStore final: public NormalCacheStore {
public:
    PageHandoffCacheStore() {
        // NormalCacheStore drains its backing store during destruction.
        request_block_buffer_store_ = std::make_shared<RequestBlockBufferStore>(nullptr);
    }

    struct LoadRecord {
        std::string                     peer;
        int                             partition_count;
        int                             partition_id;
        std::unordered_set<std::string> keys;
    };

    std::string publication_peer;
    std::map<std::string, std::unordered_map<std::string, std::shared_ptr<BlockBuffer>>> published;
    std::vector<LoadRecord> loads;
    std::vector<std::shared_ptr<LoadContext>> load_contexts;
    int                    fail_load_call    = -1;
    int                    timeout_load_call = -1;
    size_t                 combined_load_calls = 0;

    void store(const std::shared_ptr<RequestBlockBuffer>& buffer, CacheStoreStoreDoneCallback callback) override {
        auto& blocks = published[publication_peer];
        for (const auto& [key, block] : buffer->getBlocks()) {
            EXPECT_FALSE(block->gpu_mem);
            EXPECT_TRUE(blocks.emplace(key, block).second) << key;
        }
        callback(true, CacheStoreErrorCode::None);
    }

    void load(const std::shared_ptr<RequestBlockBuffer>& buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&                         ip,
              uint32_t,
              uint32_t,
              uint32_t,
              int partition_count,
              int partition_id) override {
        const auto call = static_cast<int>(combined_load_calls++);
        ASSERT_LT(static_cast<size_t>(call), loads.size());
        EXPECT_EQ(buffer->getBlocksCount(), loads[call].keys.size()) << "combine_load dropped a duplicate source key";
        EXPECT_EQ(ip, loads[call].peer);
        if (call == timeout_load_call) {
            return;
        }
        if (call == fail_load_call) {
            callback(false, CacheStoreErrorCode::LoadErrorUnknown);
            return;
        }

        ASSERT_GT(partition_count, 0);
        ASSERT_GE(partition_id, 0);
        ASSERT_LT(partition_id, partition_count);
        const auto peer = published.find(ip);
        ASSERT_NE(peer, published.end());
        for (const auto& [key, destination] : buffer->getBlocks()) {
            EXPECT_TRUE(loads[call].keys.count(key)) << key;
            const auto source = peer->second.find(key);
            ASSERT_NE(source, peer->second.end()) << key;
            const auto& block = source->second;
            ASSERT_FALSE(destination->gpu_mem);
            ASSERT_EQ(block->len % partition_count, 0u) << key;
            const auto length = block->len / partition_count;
            ASSERT_EQ(destination->len, length) << key;
            const auto* begin = static_cast<const uint8_t*>(block->addr.get()) + length * partition_id;
            std::memcpy(destination->addr.get(), begin, length);
        }
        callback(true, CacheStoreErrorCode::None);
    }

    std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& buffers,
                const std::string&                                      ip,
                uint32_t                                                port,
                uint32_t                                                rdma_port,
                int64_t                                                 timeout_ms,
                LoadContext::CheckCancelFunc                            check_cancel_func,
                int                                                     partition_count,
                int                                                     partition_id) override {
        LoadRecord record{ip, partition_count, partition_id, {}};
        for (const auto& buffer : buffers) {
            for (const auto& [key, block] : buffer->getBlocks()) {
                EXPECT_TRUE(record.keys.insert(key).second) << "duplicate source key inside one LoadContext: " << key;
            }
        }
        const auto call = static_cast<int>(loads.size());
        loads.push_back(std::move(record));
        auto context = std::make_shared<LoadContext>(shared_from_this(), /*combine_load=*/true);
        load_contexts.push_back(context);
        context->load(buffers,
                       ip,
                       port,
                       rdma_port,
                       call == timeout_load_call ? 0 : timeout_ms,
                       std::move(check_cancel_func),
                       partition_count,
                       partition_id);
        return context;
    }

    std::shared_ptr<StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&, int64_t) override {
        return nullptr;
    }
    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }
    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}
    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }
    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return memory_util_;
    }
    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> memory_util_;
};

class PageHandoffEngine final: public EngineBase {
public:
    PageHandoffEngine(std::shared_ptr<KVCacheManager> cache_manager, bool mtp): EngineBase(EngineInitParams()), mtp_(mtp) {
        resource_context_.cache_manager = std::move(cache_manager);
    }
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used by PageHandoffTest");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    bool isMTPEagle() override {
        return mtp_;
    }

private:
    bool mtp_;
};

CacheConfig makePageHandoffConfig(size_t kernel_tokens,
                                   size_t global_heads,
                                   size_t local_heads,
                                   size_t head_dim,
                                   DataType dtype,
                                   int layers,
                                   int capacity) {
    auto spec = test::makeResolvedMhaSpec(dtype, local_heads, head_dim, /*physical_tokens=*/64);
    spec->kernel_seq_size_per_block = static_cast<uint32_t>(kernel_tokens);
    spec->global_kv_head_num        = static_cast<uint32_t>(global_heads);
    return test::makeSingleGroupCacheConfig(std::move(spec), CacheGroupType::FULL, layers, capacity);
}

CacheStoreGeometryPB pageHandoffGeometry(size_t kernel_tokens) {
    CacheStoreGeometryPB geometry;
    geometry.set_tag("default");
    geometry.set_physical_tokens_per_block(64);
    geometry.set_kernel_tokens_per_block(kernel_tokens);
    return geometry;
}

// Preserve TP widths, singleton head partitions, and GQA/MQA replicas.
// Page slicing depends on the local head count and replica fanout.
constexpr std::pair<int, size_t> kPageHandoffTopologies[] = {{1, 8}, {2, 8}, {4, 8}, {4, 4}, {4, 2}, {4, 1}};

// Four-byte vectors encode independent semantic coordinates. Physical offsets
// are checked by emitting/consuming nested layout order, without planner math.
uint8_t pageHandoffByte(size_t model, size_t layer, size_t block, size_t component, size_t head, size_t token, size_t byte) {
    const uint8_t coordinates[] = {static_cast<uint8_t>(token),
                                   static_cast<uint8_t>(head),
                                   static_cast<uint8_t>(block),
                                   static_cast<uint8_t>(model * 53 + layer * 17 + component * 31)};
    return coordinates[byte % 4] ^ static_cast<uint8_t>(0x80 + byte * 7);
}

class PageHandoffTest: public DeviceTestBase {
protected:
    void roundTrip(int cp,
                     int tp,
                     size_t heads,
                     size_t source_kernel,
                     size_t destination_kernel,
                     DataType dtype,
                     bool mtp = false,
                     int fail_load_call = -1,
                     int timeout_load_call = -1) {
        constexpr size_t physical_tokens = 64;
        constexpr size_t main_layers     = 2;
        const size_t block_count = static_cast<size_t>(cp * 2 + 1);  // Includes a partial CP round.
        const int    capacity    = static_cast<int>(block_count + 3);
        const bool   has_scale   = dtype == DataType::TYPE_INT8;
        auto         store       = std::make_shared<PageHandoffCacheStore>();
        CacheKeysType keys;
        std::vector<std::string> peers;
        for (size_t block = 0; block < block_count; ++block) {
            keys.push_back(static_cast<CacheKeyType>(100 + block));
        }
        auto source_config = makePageHandoffConfig(source_kernel, heads, heads, 4, dtype, main_layers, capacity);
        const auto source_mtp = makePageHandoffConfig(source_kernel, heads, heads, 8, dtype, 1, capacity);
        if (mtp) {
            source_config.mtp_sub_configs.push_back(
                source_config.mergeMTPModule(source_mtp, /*module_index=*/0, main_layers));
        }
        for (int peer = 0; peer < cp; ++peer) {
            const auto peer_ip = "prefill-" + std::to_string(peer);
            peers.push_back(peer_ip + ":1234:1235");
            store->publication_peer = peer_ip;
            std::vector<int32_t> local_ids;
            for (size_t logical = peer; logical < block_count; logical += cp) {
                local_ids.push_back(static_cast<int32_t>(local_ids.size() * 2 + 1));
            }
            // Use a non-monotonic physical table independent of logical keys.
            std::reverse(local_ids.begin(), local_ids.end());
            torch_ext::PyCacheStoreInputs inputs;
            inputs.input_lengths_host    = torch::tensor({static_cast<int32_t>(block_count * physical_tokens)}, torch::kInt32);
            inputs.prefix_lengths_host   = torch::tensor({0}, torch::kInt32);
            inputs.host_kv_cache_offset  = torch::tensor(local_ids, torch::kInt32).reshape({1, -1});
            inputs.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
            inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
            inputs.cache_keys            = torch::tensor(keys, torch::kInt64).reshape({1, -1});

            for (size_t model = 0; model < (mtp ? 2u : 1u); ++model) {
                const auto& cfg = model == 0 ? source_config : source_mtp;
                const auto& group = cfg.group("default");
                for (size_t layer = 0; layer < cfg.layer_num; ++layer) {
                    auto kv = torch::full({capacity, static_cast<int64_t>(group.kvBlockStrideBytes())},
                                            0xfd,
                                            torch::kUInt8);
                    auto scales = has_scale ?
                                      torch::full({capacity, static_cast<int64_t>(group.kvScaleStrideBytes())},
                                                    0xfd,
                                                    torch::kUInt8) :
                                      torch::Tensor();
                    for (size_t slot = 0; slot < local_ids.size(); ++slot) {
                        const size_t logical = peer + slot * cp;
                        for (size_t part = 0; part < (has_scale ? 2u : 1u); ++part) {
                            auto& tensor = part == 0 ? kv : scales;
                            auto* begin = static_cast<uint8_t*>(tensor[local_ids[slot]].data_ptr());
                            auto* out   = begin;
                            const size_t width = part == 0 ? (model == 0 ? 4 : 8) * getTypeSize(dtype) : sizeof(float);
                            for (size_t page = 0; page < physical_tokens / source_kernel; ++page) {
                                for (size_t component = 0; component < 2; ++component) {
                                    for (size_t head = 0; head < heads; ++head) {
                                        for (size_t token = 0; token < source_kernel; ++token) {
                                            for (size_t byte = 0; byte < width; ++byte) {
                                                *out++ = pageHandoffByte(model,
                                                                          layer,
                                                                          logical,
                                                                          part * 2 + component,
                                                                          head,
                                                                          page * source_kernel + token,
                                                                          byte);
                                            }
                                        }
                                    }
                                }
                            }
                            EXPECT_EQ(static_cast<size_t>(out - begin), static_cast<size_t>(tensor.size(1)));
                        }
                    }
                    torch_ext::LayerKVCache layer_cache(kv, physical_tokens, layer, "default", scales);
                    ASSERT_NO_THROW(runtimeWriteCacheStore(inputs, layer_cache, cfg, store, model == 0 ? 11 : 17, peer, cp, nullptr));
                }
            }
            const size_t components = has_scale ? 4 : 2;
            ASSERT_EQ(store->published.at(peer_ip).size(),
                      local_ids.size() * (main_layers + (mtp ? 1 : 0)) * (physical_tokens / source_kernel) * components);
        }

        // Feed the real source allocation metadata into the byte handoff.
        PrefillRpcServer prefill_server;
        prefill_server.engine_ = std::make_shared<PageHandoffEngine>(std::make_shared<KVCacheManager>(source_config), mtp);
        auto& parallelism = prefill_server.maga_init_params_.parallelism_config;
        parallelism.tp_size                           = cp;
        parallelism.prefill_cp_config.method          = CPRotateMethod::ALL_GATHER;
        parallelism.prefill_cp_config.kv_cache_sharded = true;
        parallelism.prefill_cp_config.prefill_cp_size  = 99;
        prefill_server.maga_init_params_.model_config_.attn_config.tokens_per_block = 32;
        GenerateInputPB input;
        input.set_request_id(42);
        RPCContext rpc_context{&input, nullptr};
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext prefill_context(
            &prefill_server.resource_, rpc_context, /*timeout_ms=*/0, nullptr, metrics_reporter, nullptr);
        prefill_context.prefill_worker_cache_store_addrs = peers;
        const auto allocate = prefill_server.buildAllocateRequest(prefill_context);
        prefill_context.markRpcHandlingCompleted();
        ASSERT_EQ(allocate.prefill_cp_size(), cp);
        ASSERT_TRUE(allocate.prefill_cp_sharded());
        ASSERT_EQ(allocate.cache_store_geometry_size(), 1);
        EXPECT_EQ(allocate.cache_store_geometry(0).tag(), "default");
        EXPECT_EQ(allocate.cache_store_geometry(0).physical_tokens_per_block(), physical_tokens);
        EXPECT_EQ(allocate.cache_store_geometry(0).kernel_tokens_per_block(), source_kernel);

        const size_t head_partitions = std::gcd(static_cast<size_t>(tp), heads);
        const size_t local_heads     = heads / head_partitions;
        for (int rank = 0; rank < tp; ++rank) {
            SCOPED_TRACE(rank);
            auto config = makePageHandoffConfig(destination_kernel, heads, local_heads, 4, dtype, main_layers, capacity);
            auto mtp_config = makePageHandoffConfig(destination_kernel, heads, local_heads, 8, dtype, 1, capacity);
            if (mtp) {
                config.mtp_sub_configs.push_back(config.mergeMTPModule(mtp_config, /*module_index=*/0, main_layers));
            }
            auto cache_manager = std::make_shared<KVCacheManager>(config);
            cache_manager->coordinator_manager_ = std::make_shared<CoordinatorCacheManager>(config, AllocationType::HOST);
            ASSERT_TRUE(cache_manager->coordinator_manager_->init());
            auto blocks = std::make_shared<BlockIds>();
            BlockIndicesType block_ids;
            const auto& pool = cache_manager->coordinator_manager_->groupBlockPools().front();
            for (size_t logical = 0; logical < block_count; ++logical) {
                const auto block = pool->malloc();
                ASSERT_TRUE(block.has_value());
                block_ids.push_back(*block);
            }
            std::reverse(block_ids.begin(), block_ids.end());
            blocks->assign(block_ids);
            for (size_t layer = 0; layer < main_layers + (mtp ? 1 : 0); ++layer) {
                for (const auto id : block_ids) {
                    for (const auto& part : cache_manager->convertIndexToBuffer(layer, "default", id)) {
                        ASSERT_FALSE(part.is_cuda);
                        std::memset(part.addr, 0xfd, part.size_bytes);
                    }
                }
            }

            DecodeRpcServer server;
            server.engine_ = std::make_shared<PageHandoffEngine>(cache_manager, mtp);
            server.maga_init_params_.model_id = 11;
            server.maga_init_params_.model_config_.num_layers = main_layers;
            server.maga_init_params_.model_config_.attn_config.kv_head_num = heads;
            server.maga_init_params_.parallelism_config.tp_size = tp;
            server.maga_init_params_.parallelism_config.tp_rank = rank;
            for (int worker = 0; worker < tp; ++worker) {
                server.resource_.workers.push_back("decode-" + std::to_string(worker));
            }
            auto mtp_models = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
            auto module0 = std::make_unique<EngineInitParams>();
            module0->model_id = 17;
            module0->model_config_.num_layers = 1;
            mtp_models->push_back(std::move(module0));
            ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, 1, std::move(mtp_models));
            server.propose_maga_init_params_ = &propose_params;
            DecodeRpcServer::LoadKVCacheContext context{
                allocate.request_id(),
                "request",
                {allocate.peer_addrs().begin(), allocate.peer_addrs().end()},
                keys,
                {{{"default", 0}}, {blocks}},
                /*reuse_block_size=*/0,
                /*timeout_ms=*/1000,
                /*partition_count=*/1,
                /*partition_id=*/0,
                /*server_context=*/nullptr,
                allocate.prefill_cp_size(),
                allocate.prefill_cp_sharded(),
                {allocate.cache_store_geometry().begin(), allocate.cache_store_geometry().end()}};
            const auto request = server.constructRemoteLoadRequest(context, rank, context.peer_addrs);
            EXPECT_EQ(request.prefill_cp_size(), allocate.prefill_cp_size());
            EXPECT_EQ(request.prefill_cp_sharded(), allocate.prefill_cp_sharded());
            ASSERT_EQ(request.cache_store_geometry_size(), allocate.cache_store_geometry_size());
            EXPECT_EQ(request.cache_store_geometry(0).tag(), allocate.cache_store_geometry(0).tag());
            EXPECT_EQ(request.cache_store_geometry(0).physical_tokens_per_block(),
                      allocate.cache_store_geometry(0).physical_tokens_per_block());
            EXPECT_EQ(request.cache_store_geometry(0).kernel_tokens_per_block(),
                      allocate.cache_store_geometry(0).kernel_tokens_per_block());
            ASSERT_EQ(request.peer_addrs_size(), cp);
            for (int peer = 0; peer < cp; ++peer) {
                EXPECT_EQ(request.peer_addrs(peer), peers[peer]);
            }
            context.partition_count = request.partition_count();
            context.partition_id    = request.partition_id();
            store->loads.clear();
            store->load_contexts.clear();
            store->combined_load_calls = 0;
            store->fail_load_call      = fail_load_call;
            store->timeout_load_call   = timeout_load_call;
            server.resource_.cache_store = store;
            const auto result = server.loadCache(context);
            ASSERT_EQ(store->loads.size(), store->combined_load_calls);
            ASSERT_GE(store->loads.size(), static_cast<size_t>(cp));
            if (fail_load_call >= 0 || timeout_load_call >= 0) {
                ASSERT_LT(static_cast<size_t>(std::max(fail_load_call, timeout_load_call)), store->loads.size());
                EXPECT_FALSE(result.error_info.ok());
                EXPECT_EQ(result.loaded_cache_block_count, 0u);
                if (timeout_load_call >= 0) {
                    EXPECT_EQ(store->load_contexts[timeout_load_call]->getErrorInfo().code(),
                              ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT)
                        << "the timed-out bucket must be settled even after an earlier bucket failed";
                    if (fail_load_call < 0) {
                        EXPECT_EQ(result.error_info.code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
                    }
                }
                continue;
            }
            ASSERT_TRUE(result.error_info.ok()) << result.error_info.ToString();
            EXPECT_EQ(result.loaded_cache_block_count, block_count) << "buckets must not multiply the physical block count";
            if (source_kernel != destination_kernel && (source_kernel > destination_kernel || local_heads > 1)) {
                std::map<std::pair<std::string, std::string>, size_t> requests_per_source;
                bool repeated_across_contexts = false;
                for (const auto& load : store->loads) {
                    for (const auto& key : load.keys) {
                        repeated_across_contexts |= ++requests_per_source[{load.peer, key}] > 1;
                    }
                }
                EXPECT_TRUE(repeated_across_contexts);
                EXPECT_GT(store->loads.size(), static_cast<size_t>(cp));
            }

            // Decode TP ranks with replicated MQA/GQA heads consume the same
            // semantic head range; no transfer planner offsets are reused here.
            const size_t head_begin = (rank / (static_cast<size_t>(tp) / head_partitions)) * local_heads;
            for (size_t model = 0; model < (mtp ? 2u : 1u); ++model) {
                const size_t layer_count = model == 0 ? main_layers : 1;
                const auto& cfg = model == 0 ? config : *config.mtp_sub_configs.front();
                for (size_t layer = 0; layer < layer_count; ++layer) {
                    const size_t global_layer = model == 0 ? layer : main_layers + layer;
                    for (size_t logical = 0; logical < block_count; ++logical) {
                        const auto parts = cache_manager->convertIndexToBuffer(global_layer, "default", block_ids[logical]);
                        ASSERT_EQ(parts.size(), has_scale ? 2u : 1u);
                        for (size_t part = 0; part < parts.size(); ++part) {
                            const auto* begin = static_cast<const uint8_t*>(parts[part].addr);
                            const auto* in = begin;
                            const size_t width = part == 0 ? (model == 0 ? 4 : 8) * getTypeSize(dtype) : sizeof(float);
                            const auto& group = cfg.group("default");
                            ASSERT_EQ(parts[part].size_bytes, part == 0 ? group.kvBlockStrideBytes() : group.kvScaleStrideBytes());
                            for (size_t page = 0; page < physical_tokens / destination_kernel; ++page) {
                                for (size_t component = 0; component < 2; ++component) {
                                    for (size_t head = 0; head < local_heads; ++head) {
                                        for (size_t token = 0; token < destination_kernel; ++token) {
                                            for (size_t byte = 0; byte < width; ++byte) {
                                                ASSERT_EQ(*in++, pageHandoffByte(model,
                                                                                  layer,
                                                                                  logical,
                                                                                  part * 2 + component,
                                                                                  head_begin + head,
                                                                                  page * destination_kernel + token,
                                                                                  byte))
                                                    << "model=" << model << " layer=" << layer << " block=" << logical
                                                    << " component=" << part * 2 + component << " head=" << head
                                                    << " page=" << page << " token=" << token << " byte=" << byte;
                                            }
                                        }
                                    }
                                }
                            }
                            EXPECT_EQ(static_cast<size_t>(in - begin), parts[part].size_bytes);
                        }
                    }
                }
            }
        }
    }
};

}  // namespace

TEST_F(PageHandoffTest, CpPublishedPagesLoadIntoEveryDecodeHeadPartition) {
    for (const auto dtype : {DataType::TYPE_FP16, DataType::TYPE_INT8}) {
        for (const int cp : {2, 4}) {
            for (const auto [tp, heads] : kPageHandoffTopologies) {
                SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int>(dtype)
                                                 << " CP=" << cp << " TP=" << tp << " heads=" << heads);
                roundTrip(cp, tp, heads, /*source_kernel=*/16, /*destination_kernel=*/16, dtype);
            }
        }
    }
}

TEST_F(PageHandoffTest, CpPublishedPagesLoadAcrossBothKernelTokenSpans) {
    for (const auto [source_kernel, destination_kernel] : {std::pair<size_t, size_t>{16, 64}, {64, 16}}) {
        for (const auto dtype : {DataType::TYPE_FP16, DataType::TYPE_INT8}) {
            for (const int cp : {2, 4}) {
                for (const auto [tp, heads] : kPageHandoffTopologies) {
                    SCOPED_TRACE(::testing::Message() << "source_kernel=" << source_kernel
                                                     << " destination_kernel=" << destination_kernel
                                                     << " dtype=" << static_cast<int>(dtype)
                                                     << " CP=" << cp << " TP=" << tp << " heads=" << heads);
                    roundTrip(cp, tp, heads, source_kernel, destination_kernel, dtype);
                }
            }
        }
    }
}

TEST_F(PageHandoffTest, MtpModule0LoadsItsOwnBytesWithTheSharedTagGeometry) {
    roundTrip(4, 2, 8, /*source_kernel=*/16, /*destination_kernel=*/64, DataType::TYPE_INT8, /*mtp=*/true);
    roundTrip(2, 4, 2, /*source_kernel=*/64, /*destination_kernel=*/16, DataType::TYPE_INT8, /*mtp=*/true);
}

TEST_F(PageHandoffTest, OneFailedOrTimedOutBucketRejectsTheWholeHandoff) {
    roundTrip(4, 2, 8, 16, 64, DataType::TYPE_INT8, /*mtp=*/false, /*fail_load_call=*/2);
    roundTrip(4, 2, 8, 64, 16, DataType::TYPE_INT8, /*mtp=*/false, /*fail_load_call=*/-1, /*timeout_load_call=*/2);
    roundTrip(4, 2, 8, 64, 16, DataType::TYPE_INT8, /*mtp=*/false, /*fail_load_call=*/1, /*timeout_load_call=*/2);
}

TEST_F(PageHandoffTest, InvalidProducerGeometryFailsBeforeAnyLoadSubmission) {
    auto config = makePageHandoffConfig(/*kernel_tokens=*/16, /*global_heads=*/2, /*local_heads=*/1,
                                          /*head_dim=*/4, DataType::TYPE_INT8, /*layers=*/1, /*capacity=*/8);
    DecodeRpcServer server;
    server.engine_ = std::make_shared<PageHandoffEngine>(std::make_shared<KVCacheManager>(config), /*mtp=*/false);
    server.maga_init_params_.model_config_.num_layers = 1;
    server.maga_init_params_.parallelism_config.tp_size = 2;
    server.resource_.cache_store = std::make_shared<PageHandoffCacheStore>();
    const auto store = std::static_pointer_cast<PageHandoffCacheStore>(server.resource_.cache_store);
    auto blocks = std::make_shared<BlockIds>();
    blocks->assign({1});
    const std::vector<std::string> peers{"prefill-0:1234:1235", "prefill-1:1234:1235"};
    for (const std::string invalid : {"missing", "missing_tag", "duplicate", "zero_kernel", "nondivisible",
                                      "kernel_exceeds_physical", "physical_mismatch", "source_cp_mismatch"}) {
        SCOPED_TRACE(invalid);
        auto context = makeLoadContext("request", peers, {101}, {{{"default", 0}}, {blocks}}, 2, 0,
                                         {pageHandoffGeometry(16)});
        if (invalid == "missing") {
            context.cache_store_geometry.clear();
        } else if (invalid == "missing_tag") {
            context.cache_store_geometry.front().set_tag("other");
        } else if (invalid == "duplicate") {
            context.cache_store_geometry.push_back(context.cache_store_geometry.front());
        } else if (invalid == "zero_kernel") {
            context.cache_store_geometry.front().set_kernel_tokens_per_block(0);
        } else if (invalid == "nondivisible") {
            context.cache_store_geometry.front().set_kernel_tokens_per_block(12);
        } else if (invalid == "kernel_exceeds_physical") {
            context.cache_store_geometry.front().set_kernel_tokens_per_block(128);
        } else if (invalid == "physical_mismatch") {
            context.cache_store_geometry.front().set_physical_tokens_per_block(32);
        } else {
            context.prefill_cp_size = 4;
        }
        const auto result = server.loadCache(context);
        EXPECT_EQ(result.error_info.code(), ErrorCode::LOAD_KV_CACHE_FAILED);
        EXPECT_EQ(result.loaded_cache_block_count, 0u);
        EXPECT_TRUE(store->loads.empty());
        EXPECT_EQ(store->combined_load_calls, 0u);
    }
}

TEST(DecodeRpcServerTest, MtpSharedBlockTableRejectsDifferentKernelTokenSpans) {
    auto main = makePageHandoffConfig(16, 2, 2, 4, DataType::TYPE_INT8, 2, 8);
    auto mtp = makePageHandoffConfig(64, 2, 2, 8, DataType::TYPE_INT8, 1, 8);
    EXPECT_ANY_THROW(main.mergeMTPModule(mtp, /*module_index=*/0, /*main_layer_num=*/2));
}

TEST(DecodeRpcServerTest, TimeoutLearnedAfterConstructionKeepsAbsoluteDeadline) {
    DecodeRpcContext             rpc_context{nullptr};
    kmonitor::MetricsReporterPtr metrics_reporter;
    DecodeGenerateContext        context(rpc_context, /*timeout_ms=*/0, nullptr, metrics_reporter, nullptr);

    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(1000);
    ASSERT_TRUE(context.request_deadline.has_value());
    const auto absolute_deadline = *context.request_deadline;
    EXPECT_GT(absolute_deadline, std::chrono::system_clock::now());
    EXPECT_EQ(context.request_timeout_ms, 1000);

    context.setRequestTimeoutMs(1000);
    EXPECT_EQ(context.request_deadline, absolute_deadline) << "reapplying a timeout must not extend its budget";

    context.setRequestTimeoutMs(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_TRUE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(0);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(-1);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());
}

TEST(ModelRpcProtoTest, GroupedCacheFieldsPreserveLegacyNumbers) {
    const auto* generate = GenerateRequestPB::descriptor();
    ASSERT_NE(generate, nullptr);
    EXPECT_EQ(generate->FindFieldByName("propose_hidden")->number(), 11);
    EXPECT_EQ(generate->FindFieldByName("prefill_cp_size")->number(), 12);
    EXPECT_EQ(generate->FindFieldByName("prefill_cp_sharded")->number(), 13);
    EXPECT_EQ(generate->FindFieldByName("cache_store_geometry")->number(), 14);

    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    ASSERT_NE(broadcast, nullptr);
    EXPECT_TRUE(broadcast->IsReservedNumber(5));
    EXPECT_TRUE(broadcast->IsReservedNumber(12));
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);
    EXPECT_EQ(broadcast->FindFieldByName("cache_store_geometry")->number(), 15);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_sharded")->number(), 16);

    const auto* geometry = CacheStoreGeometryPB::descriptor();
    ASSERT_NE(geometry, nullptr);
    EXPECT_EQ(geometry->FindFieldByName("tag")->number(), 1);
    EXPECT_EQ(geometry->FindFieldByName("physical_tokens_per_block")->number(), 2);
    EXPECT_EQ(geometry->FindFieldByName("kernel_tokens_per_block")->number(), 3);

    const auto* response = BroadcastLoadResponsePB::descriptor();
    ASSERT_NE(response, nullptr);
    EXPECT_EQ(response->FindFieldByName("loaded_cache_block_count")->number(), 3);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(DecodeRpcServerTest, HeterogeneousPhysicalBlockMarksExactCoveredBaseKeys) {
    auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    EXPECT_EQ(DecodeRpcServer::cacheKeysPerPhysicalBlock(8, 4), 2u);
    EXPECT_EQ(DecodeRpcServer::keyBlocksPerLogicalBlock(policy, 8, 4), 2u);

    std::vector<size_t> transferred(5, 0);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/1, /*block_offset_index=*/0, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/3, /*block_offset_index=*/1, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/4, /*block_offset_index=*/2, /*key_span=*/2);

    EXPECT_EQ(transferred, (std::vector<size_t>{1, 1, 1, 1, 1}));
}

TEST(DecodeRpcServerTest, CompactPhysicalBlockCoversEveryBaseKey) {
    auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    EXPECT_EQ(DecodeRpcServer::cacheKeysPerPhysicalBlock(kCompactSeqSizePerBlock, kBaseSeqSizePerBlock), 2u);
    EXPECT_EQ(DecodeRpcServer::keyBlocksPerLogicalBlock(policy, kCompactSeqSizePerBlock, kBaseSeqSizePerBlock), 1u);

    std::vector<size_t> transferred(11, 0);
    DecodeRpcServer::markCacheKeyRange(transferred, /*endpoint_key_index=*/9, /*block_offset_index=*/4, /*key_span=*/2);
    DecodeRpcServer::markCacheKeyRange(
        transferred, /*endpoint_key_index=*/10, /*block_offset_index=*/5, /*key_span=*/2);

    EXPECT_EQ(transferred, (std::vector<size_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1}));
}

TEST(DecodeRpcServerTest, CompletedHandoffPrefixRequiresEveryTransferObligation) {
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/2,
                  /*required=*/{0, 0, 2, 2, 2, 2},
                  /*transferred=*/{0, 0, 2, 2, 1, 2}),
              4u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/0, /*required=*/{2, 2}, /*transferred=*/{1, 2}),
              0u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/2, /*required=*/{0, 0, 0, 0}, /*transferred=*/{0, 0, 0, 0}),
              0u);
    EXPECT_EQ(DecodeRpcServer::completedHandoffPrefixBlocks(
                  /*already_reused_blocks=*/0, /*required=*/{1}, /*transferred=*/{}),
              0u);
}

TEST(DecodeRpcServerTest, MultiRankHandoffUsesMinimumPrefix) {
    EXPECT_EQ(DecodeRpcServer::minLoadedCacheBlockCount({6, 4, 5}), 4u);
    EXPECT_EQ(DecodeRpcServer::minLoadedCacheBlockCount({}), 0u);
}

TEST(DecodeRpcServerTest, EmptyRemoteLoadTopologyFailsBeforePartitionArithmetic) {
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/0, /*peer_size=*/1).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/1, /*peer_size=*/0).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/4, /*peer_size=*/2).ok());
    EXPECT_EQ(DecodeRpcServer::validateRemoteLoadTopology(/*worker_size=*/3, /*peer_size=*/2).code(),
              ErrorCode::LOAD_KV_CACHE_FAILED);
}

TEST(DecodeRpcServerTest, OddTpWorkersWaitForEveryCompletionQueueResponse) {
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(3), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(5), (std::vector<size_t>{2, 2, 1}));
    EXPECT_EQ(DecodeRpcServer::completionQueueExpectedResponseCounts(4), (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(DecodeRpcServer::completionQueueExpectedResponseCounts(0).empty());
}

TEST(DecodeRpcServerTest, CompletedHandoffPublishesOnlyReusablePromptBlocks) {
    auto stream = makeGenerateStream(/*seq_length=*/2560);

    EXPECT_EQ(DecodeRpcServer::markLoadedCacheReuse(stream,
                                                    {ErrorInfo::OkStatus(), /*loaded_cache_block_count=*/10},
                                                    /*seq_size_per_block=*/256,
                                                    /*group_num=*/2),
              2304);
    EXPECT_EQ(stream->initialReuseLength(), 2304);
    EXPECT_EQ(stream->reuseLength(), 2304);
    EXPECT_EQ(stream->localReuseLength(), 2304);
}

TEST(DecodeRpcServerTest, FailedOrSingleGroupHandoffDoesNotPublishReuse) {
    auto stream = makeGenerateStream(/*seq_length=*/513);

    EXPECT_EQ(DecodeRpcServer::markLoadedCacheReuse(
                  stream,
                  {ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "load failed"), /*loaded_cache_block_count=*/2},
                  /*seq_size_per_block=*/256,
                  /*group_num=*/2),
              0);
    EXPECT_EQ(DecodeRpcServer::markLoadedCacheReuse(stream,
                                                    {ErrorInfo::OkStatus(), /*loaded_cache_block_count=*/2},
                                                    /*seq_size_per_block=*/256,
                                                    /*group_num=*/1),
              0);
    EXPECT_EQ(stream->initialReuseLength(), 0);
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};
    server.maga_init_params_.model_config_.attn_config.kv_head_num = 2;

    const std::string                                   request_key = "request";
    const std::vector<std::string>                      peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType>                     cache_keys  = {101, 102};
    GroupBlockIds                                       group_block_ids;
    const auto                                          load_context = makeLoadContext(request_key,
                                              peer_addrs,
                                              cache_keys,
                                              std::move(group_block_ids),
                                              /*cp_size=*/2,
                                              /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_TRUE(request.prefill_cp_sharded());
    EXPECT_EQ(request.partition_count(), 2);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.cache_keys_size(), 2);
    EXPECT_EQ(request.cache_keys(0), 101);
    EXPECT_EQ(request.cache_keys(1), 102);
}

TEST(DecodeRpcServerTest, LoadRequestsForwardSourceGeometryWithoutReplacingItWithDecodeHints) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};
    server.maga_init_params_.model_config_.attn_config.kv_head_num = 2;
    const std::vector<std::string> peers{"prefill-0", "prefill-1"};
    CacheStoreGeometryPB geometry;
    geometry.set_tag("default");
    geometry.set_physical_tokens_per_block(64);
    geometry.set_kernel_tokens_per_block(16);
    auto context = makeLoadContext("request", peers, {101, 102}, {}, 2, 0, {geometry});
    geometry.set_kernel_tokens_per_block(64);

    for (const auto& request : {server.constructRemoteLoadRequest(context, 1, peers),
                                server.constructRemoteLoadRequestForMla(context, 1, peers)}) {
        EXPECT_EQ(request.prefill_cp_size(), 2);
        EXPECT_TRUE(request.prefill_cp_sharded());
        ASSERT_EQ(request.cache_store_geometry_size(), 1);
        EXPECT_EQ(request.cache_store_geometry(0).tag(), "default");
        EXPECT_EQ(request.cache_store_geometry(0).physical_tokens_per_block(), 64);
        EXPECT_EQ(request.cache_store_geometry(0).kernel_tokens_per_block(), 16);
    }
}

TEST(DecodeRpcServerTest, CPPageOwnersAndDecodeHeadPartitionsAreIndependent) {
    for (int cp : {2, 4}) {
        std::vector<std::string> peers;
        for (int rank = 0; rank < cp; ++rank) {
            peers.push_back("prefill-" + std::to_string(rank));
        }
        for (int heads : {1, 2, 8}) {
            for (int workers : {1, 2, 4}) {
                DecodeRpcServer server;
                server.maga_init_params_.model_config_.attn_config.kv_head_num = heads;
                for (int rank = 0; rank < workers; ++rank) {
                    server.resource_.workers.push_back("decode-" + std::to_string(rank));
                }
                const auto context    = makeLoadContext("request", peers, {101, 102}, {}, cp);
                const int  partitions = std::min(heads, workers);
                for (int rank = 0; rank < workers; ++rank) {
                    const auto request = server.constructRemoteLoadRequest(context, rank, peers);
                    EXPECT_EQ(request.partition_count(), partitions);
                    EXPECT_EQ(request.partition_id(), rank / (workers / partitions));
                    ASSERT_EQ(request.peer_addrs_size(), cp);
                    for (int peer = 0; peer < cp; ++peer) {
                        EXPECT_EQ(request.peer_addrs(peer), peers[peer]);
                    }
                }
            }
        }
    }
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string                                   request_key = "request";
    const std::vector<std::string>                      peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType>                     cache_keys  = {101};
    GroupBlockIds                                       group_block_ids;
    const auto                                          load_context = makeLoadContext(request_key,
                                              peer_addrs,
                                              cache_keys,
                                              std::move(group_block_ids),
                                              /*cp_size=*/2,
                                              /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
}

TEST(DecodeRpcServerTest, CPShardedOpaqueLoadRequestKeepsWholeComponents) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};
    const std::vector<std::string> peers{"prefill-0", "prefill-1"};
    const auto                     context = makeLoadContext("request", peers, {101}, {}, 2);
    const auto                     request = server.constructRemoteLoadRequest(context, 1, peers, false);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.peer_addrs_size(), 2);
}

TEST(DecodeRpcServerTest, TaggedBlockRowsPreserveReceivedOrderAcrossTopologies) {
    auto topology =
        CacheTopology::create({makeRpcGroup("linear"), makeRpcGroup("full")}, {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks.orderedTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(blocks.blocks("full"), (BlockIndicesType{10}));
    EXPECT_EQ(blocks.blocks("linear"), (BlockIndicesType{20}));

    auto reordered =
        CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_NE(topology->groupTags(), reordered->groupTags());
    const auto reordered_blocks = DecodeRpcServer::decodeGroupBlockIds(request, *reordered);
    EXPECT_EQ(reordered_blocks.orderedTags(), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(reordered_blocks.blocks("full"), blocks.blocks("full"));
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, LoadContextOwnsGroupIdentityAndRequestData) {
    auto full_blocks = std::make_shared<BlockIds>();
    full_blocks->assign({10});
    auto linear_blocks = std::make_shared<BlockIds>();
    linear_blocks->assign({20, NULL_BLOCK_IDX});
    GroupBlockIds blocks{{{"linear", 0}, {"full", 1}}, {linear_blocks, full_blocks}};
    auto          context = makeLoadContext("request", {"peer"}, {101}, blocks, /*prefill_cp_size=*/1);
    EXPECT_EQ(context.groupBlockIds().rows_[1], full_blocks);
    // The context owns its index, while the mutable BlockIds holders remain shared.
    blocks.tag_to_index_.clear();
    blocks.rows_.clear();
    full_blocks->assign({11});
    full_blocks.reset();
    linear_blocks.reset();

    EXPECT_EQ(context.request_key, "request");
    EXPECT_EQ(context.peer_addrs, (std::vector<std::string>{"peer"}));
    EXPECT_EQ(context.cache_keys, (CacheKeysType{101}));
    EXPECT_EQ(context.groupBlockIds().orderedTags(), (std::vector<std::string>{"linear", "full"}));
    EXPECT_EQ(context.blockIdsForGroup("full").blocks(), (BlockIndicesType{11}));
    EXPECT_EQ(context.blockIdsForGroup("linear").blocks(), (BlockIndicesType{20, NULL_BLOCK_IDX}));
    EXPECT_ANY_THROW(context.blockIdsForGroup("missing"));
}

TEST(DecodeRpcServerTest, ResourceRowsKeepTheirOwnOrderAndSharedLifetime) {
    auto resource = std::make_unique<KVCacheResource>();
    resource->initGroups(
        CacheTopology::create({makeRpcGroup("linear"), makeRpcGroup("full")}, {{0, {"full", "linear"}}}));
    resource->mutableBlockIds("linear").assign({20});
    resource->mutableBlockIds("full").assign({10});
    auto context = makeLoadContext("request", {"peer"}, {}, resource->groupBlockIds(), 1);
    EXPECT_NE(&context.groupBlockIds().tag_to_index_, &resource->groupBlockIds().tag_to_index_);
    EXPECT_EQ(context.groupBlockIds().rows_[1], resource->groupBlockIds().rows_[1]);
    resource->mutableBlockIds("full").assign({11, NULL_BLOCK_IDX});
    resource.reset();
    auto sender_topology =
        CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB request;
    DecodeRpcServer::appendGroupBlockIds(context, *sender_topology, request);
    EXPECT_EQ(context.groupBlockIds().orderedTags(), (std::vector<std::string>{"linear", "full"}));
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "linear");
    EXPECT_EQ(context.blockIdsForGroup("full").blocks(), (BlockIndicesType{11, NULL_BLOCK_IDX}));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full")}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatchWithoutReplacingResult) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto holder   = std::make_shared<BlockIds>();
    holder->assign({42});
    GroupBlockIds          previous{{{"unchanged", 0}}, {holder}};
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));

    BroadcastLoadRequestPB duplicate;
    for (int i = 0; i < 2; ++i) {
        auto* duplicate_row = duplicate.add_tagged_group_block_ids();
        duplicate_row->set_tag("full");
        duplicate_row->add_block_ids(i + 1);
    }
    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(duplicate, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));

    BroadcastLoadRequestPB unknown;
    auto*                  unknown_row = unknown.add_tagged_group_block_ids();
    unknown_row->set_tag("foreign");
    unknown_row->add_block_ids(1);
    EXPECT_ANY_THROW(previous = DecodeRpcServer::decodeGroupBlockIds(unknown, *topology));
    EXPECT_EQ(previous.blocks("unchanged"), (BlockIndicesType{42}));
}

TEST(DecodeRpcServerTest, DenseContextRejectsInvalidIdentityBeforeUse) {
    auto                             holder = std::make_shared<BlockIds>();
    const std::vector<GroupBlockIds> malformed{
        {{}, {holder}},                                    // Missing identity.
        {{{"", 0}}, {holder}},                             // Empty tag.
        {{{"full", 0}, {"linear", 0}}, {holder, holder}},  // Duplicate index and uncovered row.
        {{{"full", 1}}, {holder}},                         // Out-of-range index.
        {{{"full", 0}, {"linear", 2}}, {holder, holder}},  // Hole in the dense index.
        {{{"full", 0}}, {nullptr}},                        // Null holder.
        {{{"full", 0}}, {}}};                              // Index without a row.
    for (const auto& blocks : malformed) {
        EXPECT_ANY_THROW(makeLoadContext("request", {"peer"}, {}, blocks, 1));
    }
}

TEST(DecodeRpcServerTest, DenseContextSerializesOwnOrderAndPreservesEmptyRows) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto full     = std::make_shared<BlockIds>();
    full->assign({10, NULL_BLOCK_IDX});
    auto linear  = std::make_shared<BlockIds>();
    auto context = makeLoadContext("request", {"peer"}, {}, {{{"linear", 0}, {"full", 1}}, {linear, full}}, 1);
    BroadcastLoadRequestPB request;
    DecodeRpcServer::appendGroupBlockIds(context, *topology, request);
    ASSERT_EQ(request.tagged_group_block_ids_size(), 2);
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "linear");
    EXPECT_EQ(request.tagged_group_block_ids(0).block_ids_size(), 0);
    EXPECT_EQ(request.tagged_group_block_ids(1).tag(), "full");
    auto rows     = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    auto received = makeLoadContext("received", {"peer"}, {}, std::move(rows), 1);
    EXPECT_EQ(received.groupBlockIds().orderedTags(), context.groupBlockIds().orderedTags());
    EXPECT_TRUE(received.blockIdsForGroup("linear").blocks().empty());
    EXPECT_EQ(received.blockIdsForGroup("full").blocks(), (BlockIndicesType{10, NULL_BLOCK_IDX}));
    EXPECT_NE(received.groupBlockIds().rows_[1], full);
}

TEST(DecodeRpcServerTest, DenseContextTopologyFailureDoesNotPublishWireRows) {
    auto topology = CacheTopology::create({makeRpcGroup("full"), makeRpcGroup("linear")}, {{0, {"full", "linear"}}});
    auto holder   = std::make_shared<BlockIds>();
    const std::vector<GroupBlockIds> mismatched{{{{"full", 0}}, {holder}},
                                                {{{"full", 0}, {"foreign", 1}}, {holder, holder}}};
    for (const auto& blocks : mismatched) {
        auto                   context = makeLoadContext("request", {"peer"}, {}, blocks, 1);
        BroadcastLoadRequestPB request;
        EXPECT_ANY_THROW(DecodeRpcServer::appendGroupBlockIds(context, *topology, request));
        EXPECT_EQ(request.tagged_group_block_ids_size(), 0);
    }
    BroadcastLoadRequestPB request;
    request.add_tagged_group_block_ids()->set_tag("full");
    request.add_tagged_group_block_ids()->set_tag("");
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

TEST(DecodeRpcServerTest, CancelledGenerateRequestReadUsesCancelledStatus) {
    const auto status = DecodeRpcServer::generateRequestReadFailureStatus(/*cancelled=*/true);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(status.error_message(), "request is cancelled");
}

TEST(DecodeRpcServerTest, NonCancelledGenerateRequestReadPreservesFailure) {
    const auto status = DecodeRpcServer::generateRequestReadFailureStatus(/*cancelled=*/false);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "poll generate request failed");
}

class PrefillCompletionRpcTest: public DeviceTestBase {};

TEST_F(PrefillCompletionRpcTest, SettlesWithoutDecodeAndPreservesProtocolFailures) {
    class CompletionService final: public RpcService::Service {
    public:
        grpc::Status RemoteGenerate(grpc::ServerContext* server_context, ServerStream* rpc_stream) override {
            DecodeRpcContext             rpc_context{rpc_stream};
            kmonitor::MetricsReporterPtr reporter;
            auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
            DecodeGenerateContext        context(rpc_context, 0, server_context, reporter, meta);
            context.request_id = 42;
            context.allocate_request.set_client_id("prefill");
            auto cache = std::make_shared<KVCacheManager>(
                test::makeSimpleMhaCacheConfig(1, 8, 2, DataType::TYPE_FP16), false, nullptr);
            EXPECT_TRUE(cache->init());
            ResourceContext resources;
            resources.cache_manager = cache;
            resources.role_type     = RoleType::DECODE;
            auto input              = std::make_shared<GenerateInput>();
            input->request_id       = 42;
            input->begin_time_us    = currentTimeUs();
            input->generate_config  = std::make_shared<GenerateConfig>();
            input->input_ids        = torch::tensor({1, 2, 3}, torch::kInt32);
            ModelConfig config;
            config.max_seq_len = 16;
            auto stream = std::make_shared<NormalGenerateStream>(input, config, RuntimeConfig{}, resources, nullptr);
            EXPECT_TRUE(stream->initKVBlock().ok());
            EXPECT_GT(stream->stream_cache_resource_->curBlocksNum(), 0);
            if (running) {
                stream->generate_status_->status.store(StreamState::RUNNING);
            }
            context.setStream(stream);
            if (expired) {
                context.request_deadline = std::chrono::system_clock::now() - std::chrono::seconds(1);
            }
            if (stream_error) {
                stream->reportError(ErrorCode::MALLOC_FAILED, "test allocation failure");
            }
            DecodeRpcServer server;
            server.localGenerate(context);
            status                  = context.error_status;
            finished                = stream->isFinished();
            had_error               = stream->hasError();
            resource_released       = stream->stream_cache_resource_->isResourceReleased();
            blocks_after_completion = stream->stream_cache_resource_->curBlocksNum();
            const auto timing       = stream->getTimeInfo();
            executed                = timing.running_started || timing.first_token_committed || timing.generation_done;
            RpcMetricsCollector metrics;
            context.collectBasicMetrics(metrics);
            error_qps = metrics.error_qps;
            if (running) {
                // This fixture has no scheduler to drain the deliberately invalid running stream.
                stream->reportError(ErrorCode::CANCELLED, "test cleanup");
                stream->moveToNext();
            }
            context.stopStream();
            error_after_cleanup = stream->hasError();
            remaining_requests  = meta->getEngineScheduleInfo(-1).running_task_info_list.size();
            return status;
        }
        bool         expired      = false;
        bool         stream_error = false;
        bool         running      = false;
        grpc::Status status;
        bool         finished = false, had_error = false, resource_released = false;
        bool         executed = false, error_qps = false, error_after_cleanup = false;
        size_t       remaining_requests      = 1;
        size_t       blocks_after_completion = 1;
    };

    for (const std::string mode :
         {"complete", "eof", "wrong_request", "wrong_client", "wrong_stage", "expired", "stream_error", "running"}) {
        SCOPED_TRACE(mode);
        CompletionService service;
        service.expired      = mode == "expired";
        service.stream_error = mode == "stream_error";
        service.running      = mode == "running";
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        auto stub = RpcService::NewStub(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto rpc = stub->RemoteGenerate(&context);
        if (mode != "eof") {
            GenerateRequestPB request;
            request.set_stage(mode == "wrong_stage" ? RemoteStage::LOAD : RemoteStage::PREFILL_COMPLETE);
            request.set_request_id(mode == "wrong_request" ? 43 : 42);
            request.set_client_id(mode == "wrong_client" ? "other" : "prefill");
            EXPECT_TRUE(rpc->Write(request));
        }
        rpc->WritesDone();
        const auto status = rpc->Finish();
        server->Shutdown();
        server->Wait();
        EXPECT_EQ(service.remaining_requests, 0u);
        EXPECT_FALSE(service.executed);
        if (mode == "complete") {
            EXPECT_TRUE(status.ok());
            EXPECT_TRUE(service.finished);
            EXPECT_TRUE(service.resource_released);
            EXPECT_EQ(service.blocks_after_completion, 0u);
            EXPECT_FALSE(service.had_error);
            EXPECT_FALSE(service.error_qps);
            EXPECT_FALSE(service.error_after_cleanup);
        } else {
            EXPECT_FALSE(status.ok());
            EXPECT_TRUE(service.error_qps);
            if (mode == "eof" || mode == "wrong_stage") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
            } else if (mode == "wrong_request" || mode == "wrong_client") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
            } else if (mode == "expired") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
            } else if (mode == "running") {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
            }
        }
    }
}

TEST(DecodeRpcServerTest, CacheLoadClientErrorPreservesCodeWithoutTopologyDetails) {
    const auto client_error =
        DecodeRpcServer::cacheLoadClientError(/*request_id=*/12345, ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);

    EXPECT_EQ(client_error.code(), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(client_error.ToString(), "cache load failed; correlation_id=12345");
    EXPECT_EQ(client_error.ToString().find("10.0.0.8:1234"), std::string::npos);
    EXPECT_EQ(client_error.ToString().find("rdma"), std::string::npos);
}

TEST(DecodeRpcServerTest, SerializedAllocationErrorCarriesGrammarDomainCode) {
    DecodeRpcServer server;
    const auto      status = server.serializeErrorMsg(
        "request", RequestInfo{}, ErrorInfo(ErrorCode::GRAMMAR_COMPILE_OVERLOADED, "grammar queue full"));

    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::GRAMMAR_COMPILE_OVERLOADED));
    EXPECT_NE(details.error_message().find("grammar queue full"), std::string::npos);
}

TEST(DecodeRpcServerTest, CacheLoadTimeoutClassifiedAsDependencyFailure) {
    // The cache timeouts map onto DEADLINE_EXCEEDED, so a predicate keyed on the
    // gRPC status would mislabel this upstream KV-transfer failure as a local
    // deadline of the decode node.
    const ErrorInfo    error_info(ErrorCode::LOAD_CACHE_TIMEOUT, "load cache timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "DependencyFailure");
}

TEST(DecodeRpcServerTest, CacheStoreLoadBufferTimeoutClassifiedAsDependencyFailure) {
    const ErrorInfo    error_info(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, "load buffer timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "DependencyFailure");
}

TEST(DecodeRpcServerTest, CancelledDuringCacheLoadKeepsCancelledClassification) {
    // A client going away while KV cache is still arriving is not a failure of the
    // Prefill dependency, so the stage alone must not decide the classification.
    const ErrorInfo    error_info(ErrorCode::CANCELLED, "request is cancelled");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::CANCELLED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, error_status),
                 "Cancelled");
}

TEST(DecodeRpcServerTest, CancelledCacheLoadClassificationDoesNotRequireTransportStatus) {
    const ErrorInfo error_info(ErrorCode::CANCELLED, "request is cancelled");

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::loadCacheFromPrefill, error_info, grpc::Status::OK),
                 "Cancelled");
}

TEST(DecodeRpcServerTest, DeadlineOutsideCacheLoadKeepsStatusClassification) {
    const ErrorInfo    error_info(ErrorCode::GENERATE_TIMEOUT, "generate timeout");
    const grpc::Status error_status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
    ASSERT_EQ(error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);

    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::localGenerate, error_info, error_status),
                 "DeadlineExceeded");
}

TEST(DecodeRpcServerTest, ExceptionUnwindingWithoutStatusIsClassifiedAsException) {
    // PhaseSpanSynthesisScope reports unwinding while error_status is still OK:
    // nothing set a gRPC status on the way out.
    EXPECT_STREQ(DecodeRpcServer::phaseErrorType(
                     /*request_ok=*/false, DecodeStatInfo::localGenerate, ErrorInfo::OkStatus(), grpc::Status::OK),
                 "Exception");
}

TEST(DecodeRpcServerTest, SuccessfulRequestHasNoPhaseErrorType) {
    EXPECT_EQ(DecodeRpcServer::phaseErrorType(/*request_ok=*/true,
                                              DecodeStatInfo::loadCacheFromPrefill,
                                              ErrorInfo(ErrorCode::LOAD_CACHE_TIMEOUT, "ignored"),
                                              grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "ignored")),
              nullptr);
}

TEST(DecodeRpcServerTest, HeterogeneousFullGroupUsesGlobalEndpointKeys) {
    const auto policy        = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/3,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/0,
                                                   /*cp_size=*/1,
                                                   /*key_blocks_per_logical_block=*/2,
                                                   /*cache_key_count=*/6);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/3,
                                                                 /*cache_key_count=*/6,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 /*group_seq_size_per_block=*/2,
                                                                 /*base_seq_size_per_block=*/1);

    EXPECT_EQ(keyOffsetPairs(producer_plan), (KeyOffsetPairs{{1, 0}, {3, 1}, {5, 2}}));
    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
}

TEST(DecodeRpcServerTest, HeterogeneousPartialBlockUsesLastAvailableKey) {
    const auto policy        = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/3,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/0,
                                                   /*cp_size=*/1,
                                                   /*key_blocks_per_logical_block=*/2,
                                                   /*cache_key_count=*/5);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/3,
                                                                 /*cache_key_count=*/5,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 /*group_seq_size_per_block=*/2,
                                                                 /*base_seq_size_per_block=*/1);

    EXPECT_EQ(keyOffsetPairs(producer_plan), (KeyOffsetPairs{{1, 0}, {3, 1}, {4, 2}}));
    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadsGlobalTailKeysIntoCanonicalSlots) {
    // 11 logical blocks under prefill CP=2 compact into a 6-slot state table:
    // slot j covers logical blocks [2j, 2j+1], so the two active tail slots 4 and
    // 5 must be filled from the *global* cache keys 9 and 10 - not from keys 4
    // and 5, which is what indexing the compacted table with logical positions
    // (or the table length with the logical key count) would produce.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/6,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{9, 4}, {10, 5}}));
}

TEST(DecodeRpcServerTest, CompactStateGroupLoadPlanMatchesProducerStorePlan) {
    // The consumer must project exactly like the producer: same (key, offset)
    // pairs, or the decode reads a key the prefill never registered.
    const auto policy        = makeCompactStatePolicy(/*active_tail_blocks=*/2);
    const auto decode_plan   = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                                 /*local_block_num=*/6,
                                                                 /*cache_key_count=*/11,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/true,
                                                                 kCompactSeqSizePerBlock,
                                                                 kBaseSeqSizePerBlock);
    const auto producer_plan = buildCacheStorePlan(policy,
                                                   /*total_logical_blocks=*/11,
                                                   /*reuse_block_size=*/0,
                                                   /*use_hybrid=*/true,
                                                   /*cp_rank=*/1,
                                                   /*cp_size=*/2);

    EXPECT_EQ(keyOffsetPairs(decode_plan), keyOffsetPairs(producer_plan));
}

TEST(DecodeRpcServerTest, CompactStateGroupIgnoresSpeculativeReserveTailSlots) {
    // MTP reserve slots make the positional table longer than the canonical slots
    // the sequence actually backs. The destinations must stay 4 and 5; picking the
    // tail of the table would target the reserve slot 6 and duplicate key 10.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/7,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{9, 4}, {10, 5}}));
}

TEST(DecodeRpcServerTest, CompactOneTailGroupLoadsOnlyTheLastCanonicalSlot) {
    // hca_state declares active_tail_blocks=1; explicit_block_num only sizes its
    // pool and must not widen the per-request projection.
    auto policy               = makeCompactStatePolicy(/*active_tail_blocks=*/1);
    policy.explicit_block_num = 256;

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/6,
                                                          /*cache_key_count=*/11,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{10, 5}}));
}

TEST(DecodeRpcServerTest, CompactStateGroupWithSingleLogicalBlockLoadsKeyZero) {
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/1,
                                                          /*cache_key_count=*/1,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kCompactSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{0, 0}}));
}

TEST(DecodeRpcServerTest, UnscaledSwaGroupKeepsLogicalTailPositions) {
    // A COMPACT_LAST_RANK policy on a group whose block still covers one logical
    // block has a flat table: the tail slots are logical positions 3 and 4.
    const auto plan = DecodeRpcServer::buildGroupLoadPlan(makeCompactStatePolicy(/*active_tail_blocks=*/2),
                                                          /*local_block_num=*/5,
                                                          /*cache_key_count=*/5,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{3, 3}, {4, 4}}));
}

TEST(DecodeRpcServerTest, TailGroupReserveSlotsDoNotStarveTheLoad) {
    // The last sequence-backed slot is 2; a reserve slot 3 must not shift the
    // one-block tail window past the stored cache keys and drop the load.
    auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.active_tail_blocks = 1;
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::NONE);

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/4,
                                                          /*cache_key_count=*/3,
                                                          /*reuse_block_size=*/0,
                                                          /*use_hybrid=*/true,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{2, 2}}));
}

TEST(DecodeRpcServerTest, FullGroupKeepsWholeLogicalBlocksAfterReuse) {
    // Decode owns whole logical blocks of a BLOCK_ROUND_ROBIN group: the plan must
    // not shard it (the per-peer split happens later, per block), and reused
    // blocks are skipped.
    const auto policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    ASSERT_EQ(policy.cp_mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);

    const auto plan = DecodeRpcServer::buildGroupLoadPlan(policy,
                                                          /*local_block_num=*/5,
                                                          /*cache_key_count=*/4,
                                                          /*reuse_block_size=*/2,
                                                          /*use_hybrid=*/false,
                                                          kBaseSeqSizePerBlock,
                                                          kBaseSeqSizePerBlock);

    EXPECT_EQ(keyOffsetPairs(plan), (KeyOffsetPairs{{2, 2}, {3, 3}}));
}

TEST(DecodeRpcServerTest, EmptyTableOrMissingCacheKeysYieldNoLoad) {
    const auto policy = makeCompactStatePolicy(/*active_tail_blocks=*/2);

    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(policy,
                                                    /*local_block_num=*/0,
                                                    /*cache_key_count=*/11,
                                                    /*reuse_block_size=*/0,
                                                    /*use_hybrid=*/true,
                                                    kCompactSeqSizePerBlock,
                                                    kBaseSeqSizePerBlock)
                    .empty());
    EXPECT_TRUE(DecodeRpcServer::buildGroupLoadPlan(policy,
                                                    /*local_block_num=*/6,
                                                    /*cache_key_count=*/0,
                                                    /*reuse_block_size=*/0,
                                                    /*use_hybrid=*/true,
                                                    kCompactSeqSizePerBlock,
                                                    kBaseSeqSizePerBlock)
                    .empty());
}

}  // namespace rtp_llm
