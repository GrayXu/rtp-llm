#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MultiPoolTestUtils.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#include <chrono>
#include <thread>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

class KVCMMultiPoolPaceTest: public DeviceTestBase {
protected:
    void roundTrip(const CacheConfig& config, int query_type = 1, bool tail_state_only = false) {
        const auto address = autil::EnvUtil::getEnv("KVCM_P2_SERVER_ADDRESS", std::string{});
        const auto instance_group = autil::EnvUtil::getEnv("KVCM_P2_INSTANCE_GROUP", std::string{});
        ASSERT_FALSE(address.empty()) << "KVCM_P2_SERVER_ADDRESS is required";
        ASSERT_FALSE(instance_group.empty()) << "KVCM_P2_INSTANCE_GROUP is required";
        KVCacheConfig options;
        options.kvcm_server_address = address;
        options.kvcm_instance_group = instance_group;
        options.kvcm_default_query_type = query_type;
        options.kvcm_read_backend_type = query_type == 1 ? 3 : 0;
        options.kvcm_min_replica_count = 1;
        options.kvcm_model_sdk_config = R"([{"type":"pace","sdk_log_level":"WARN"}])";
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        options.kvcm_instance_id_salt = "multi_pool_p2_" + std::to_string(nonce);
        RuntimeConfig runtime;
        runtime.model_name = "multi_pool_p2_byte_roundtrip";

        auto allocator = std::make_shared<CoordinatorCacheManager>(config);
        ASSERT_TRUE(allocator->init());
        // Finish allocator initialization before writing through the SDK's streams.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "GPU cache initialization left a CUDA error";
        const auto pools = allocator->groupBlockPools();
        std::vector<std::unique_ptr<ScopedReferencedBlocks>> references;
        StorageBackend::PoolsByTag pools_by_tag;
        for (size_t group = 0; group < pools.size(); ++group) {
            references.push_back(std::make_unique<ScopedReferencedBlocks>(pools[group], group + 2));
            pools_by_tag.emplace(config.groupTags()[group], pools[group]);
        }
        BackendHandle backend(std::make_unique<KVCMStorageBackend>(config, options, runtime, singleRankConfig(),
            SpeculativeExecutionConfig{}, nullptr));
        ASSERT_TRUE(backend->init(config.topologyPtr(), std::move(pools_by_tag),
            [allocator](int layer, const std::string& tag, int block) {
                return allocator->convertIndexToBuffer(layer, tag, block);
            }));
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "KVCM transfer initialization left a CUDA error";
        StorageRequest request;
        request.keys = std::make_shared<CacheKeysType>(CacheKeysType{nonce, nonce + 1});
        request.handles.resize(2);
        for (size_t key = 0; key < 2; ++key) {
            for (size_t group = 0; group < pools.size(); ++group) {
                if (tail_state_only && key == 0
                    && config.group(config.groupTags()[group]).policy.group_type == CacheGroupType::LINEAR) {
                    continue;
                }
                // Equal numeric block IDs in different pools are independent.
                request.handles[key].push_back({config.groupTags()[group], references[group]->get().at(key)});
            }
            std::reverse(request.handles[key].begin(), request.handles[key].end());
        }
        auto payload = [](size_t group, size_t key, int layer, size_t iov, size_t size) {
            std::vector<uint8_t> bytes(size);
            for (size_t byte = 0; byte < size; ++byte) {
                bytes[byte] = static_cast<uint8_t>(group * 31 + key * 17 + layer * 3 + iov + byte * 5 + (byte >> 3));
            }
            return bytes;
        };
        const auto forEachBuffer = [&](const auto& visitor) {
            for (size_t key = 0; key < 2; ++key) {
                for (size_t group = 0; group < pools.size(); ++group) {
                    const auto& tag = config.groupTags()[group];
                    const auto block = references[group]->get().at(key);
                    for (const int layer : config.layerIdsForGroup(tag)) {
                        const auto buffers = allocator->convertIndexToBuffer(layer, tag, block);
                        for (size_t iov = 0; iov < buffers.size(); ++iov) {
                            visitor(group, key, layer, iov, buffers[iov]);
                        }
                    }
                }
            }
        };
        forEachBuffer([&](size_t group, size_t key, int layer, size_t iov, const BlockInfo& buffer) {
            const auto expected = payload(group, key, layer, iov, buffer.size_bytes);
            ASSERT_EQ(cudaMemcpy(buffer.addr, expected.data(), expected.size(), cudaMemcpyHostToDevice), cudaSuccess);
        });
        ASSERT_TRUE(backend->write(backend->prepareWrite(request)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        MatchObservation matched;
        do {
            matched = match(*backend.backend, request);
            if (matched.success && matched.matched_blocks_num == 2) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (std::chrono::steady_clock::now() < deadline);
        ASSERT_TRUE(matched.success);
        ASSERT_EQ(matched.matched_blocks_num, 2u) << "Both keys must be published through real KVCM/PACE";

        RemoteOperationRequestPB locations;
        locations.set_op(REMOTE_OPERATION_MATCH_LOCATION);
        locations.set_trace_id("multi_pool_p2_locations");
        locations.mutable_metadata()->set_query_type(query_type);
        for (const auto key : *request.keys) {
            locations.mutable_metadata()->add_block_keys(key);
        }
        RemoteOperationResponsePB located;
        ASSERT_TRUE(backend->execute(locations, located));
        ASSERT_EQ(located.locations_size(), 2);
        for (const auto& location : located.locations()) {
            ASSERT_GT(location.specs_size(), 0);
            for (const auto& spec : location.specs()) {
                ASSERT_EQ(spec.uri().rfind("pace://", 0), 0u) << "The test must use real PACE payloads";
            }
        }

        // Verify a full remote read and then a suffix read after a local hit.
        for (size_t local : {0u, 1u}) {
            forEachBuffer([](size_t, size_t, int, size_t, const BlockInfo& buffer) {
                ASSERT_EQ(cudaMemset(buffer.addr, 0, buffer.size_bytes), cudaSuccess);
            });
            request.local_matched_blocks_num = local;
            matched = match(*backend.backend, request);
            ASSERT_TRUE(matched.success);
            ASSERT_EQ(matched.matched_blocks_num, 2u);
            ASSERT_TRUE(read(*backend.backend, request, matched.match_meta));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            forEachBuffer([&](size_t group, size_t key, int layer, size_t iov, const BlockInfo& buffer) {
                const auto& declared = config.group(config.groupTags()[group]);
                const bool loaded = key >= local && 2 - key <= declared.reuseBlockCount(2);
                const auto expected = loaded ? payload(group, key, layer, iov, buffer.size_bytes) :
                                               std::vector<uint8_t>(buffer.size_bytes, 0);
                std::vector<uint8_t> actual(buffer.size_bytes);
                ASSERT_EQ(cudaMemcpy(actual.data(), buffer.addr, actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_EQ(actual, expected) << "tag=" << declared.tag << " key=" << key << " layer=" << layer;
            });
        }
        for (size_t group = 0; group < pools.size(); ++group) {
            for (const auto block : references[group]->get()) {
                EXPECT_EQ(pools[group]->refCount(block), 1u);
            }
            std::cout << "verified tag=" << config.groupTags()[group]
                      << " payload_bytes=" << config.blockSizeBytesForGroup(config.groupTags()[group]) << '\n';
        }
        RemoteOperationRequestPB cleanup;
        cleanup.set_op(REMOTE_OPERATION_REMOVE_CACHE);
        cleanup.set_trace_id("multi_pool_p2_cleanup");
        for (const auto key : *request.keys) {
            cleanup.mutable_metadata()->add_block_keys(key);
        }
        RemoteOperationResponsePB response;
        EXPECT_TRUE(backend->execute(cleanup, response));
    }
};

TEST_F(KVCMMultiPoolPaceTest, RequestFinishHostCopyPublishesRemotePayloadWithoutDeviceSources) {
    const auto address = autil::EnvUtil::getEnv("KVCM_P2_SERVER_ADDRESS", std::string{});
    const auto group = autil::EnvUtil::getEnv("KVCM_P2_INSTANCE_GROUP", std::string{});
    ASSERT_FALSE(address.empty());
    ASSERT_FALSE(group.empty());
    const auto config = test::makeSimpleMhaCacheConfig(1, 16, 8, DataType::TYPE_FP16, 1, 2);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count() / 1000;
    KVCacheConfig options;
    options.enable_device_cache = false;
    options.enable_memory_cache = true;
    options.memory_cache_size_mb = 1;
    options.enable_remote_cache = true;
    options.enable_remote_cache_write_on_finish = true;
    options.kvcm_server_address = address;
    options.kvcm_instance_group = group;
    options.kvcm_instance_id_salt = "finish_host_reuse_" + std::to_string(nonce);
    options.kvcm_model_sdk_config = R"([{"type":"pace","sdk_log_level":"WARN"}])";
    RuntimeConfig runtime;
    runtime.model_name = "finish_host_reuse";
    auto allocator = std::make_shared<CoordinatorCacheManager>(config);
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto backend = std::make_shared<KVCMStorageBackend>(config, options, runtime, singleRankConfig(),
                                                       SpeculativeExecutionConfig{}, nullptr);
    auto cache = createBlockTreeCache(config, options, allocator, singleRankConfig(), backend);
    ASSERT_NE(cache, nullptr);
    ASSERT_TRUE(backend->canInitiateHostWrite());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto pool = allocator->groupBlockPools().front();
    auto sources = std::make_unique<ScopedReferencedBlocks>(pool, 2);
    std::vector<std::vector<GroupSetResource>> resources(2, std::vector<GroupSetResource>(1));
    const auto& tag = config.groupTags().front();
    for (size_t key = 0; key < 2; ++key) {
        resources[key][0].device_blocks = {sources->get()[key]};
        for (const auto& buffer : allocator->convertIndexToBuffer(0, tag, sources->get()[key])) {
            ASSERT_EQ(cudaMemset(buffer.addr, 0x61 + key, buffer.size_bytes), cudaSuccess);
        }
    }
    const CacheKeysType keys{nonce, nonce + 1};
    cache->insert(keys, resources, Tier::HOST, false, true);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
    sources.reset();
    EXPECT_EQ(pool->usedBlocksNum(), 0u);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend));
    const auto path = cache->tree()->findNode(keys);
    ASSERT_EQ(path.size(), 2u);
    EXPECT_TRUE(path.back()->group_set_resources[0].hasTier(Tier::HOST));
    EXPECT_EQ(cache->groupSets()[0]->hostPool()->referencedBlocksNum(BlockTreeRefType::STORE), 0u);

    ScopedReferencedBlocks destinations(pool, 2);
    StorageRequest request;
    request.keys = std::make_shared<CacheKeysType>(keys);
    request.handles = {{{tag, destinations.get()[0]}}, {{tag, destinations.get()[1]}}};
    auto matched = match(*backend, request);
    ASSERT_TRUE(matched.success);
    ASSERT_EQ(matched.matched_blocks_num, 2u);
    for (const auto block : destinations.get()) {
        for (const auto& buffer : allocator->convertIndexToBuffer(0, tag, block)) {
            ASSERT_EQ(cudaMemset(buffer.addr, 0, buffer.size_bytes), cudaSuccess);
        }
    }
    ASSERT_TRUE(read(*backend, request, matched.match_meta));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (size_t key = 0; key < 2; ++key) {
        for (const auto& buffer : allocator->convertIndexToBuffer(0, tag, destinations.get()[key])) {
            std::vector<uint8_t> actual(buffer.size_bytes);
            ASSERT_EQ(cudaMemcpy(actual.data(), buffer.addr, actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(actual, std::vector<uint8_t>(actual.size(), 0x61 + key));
        }
    }
    RemoteOperationRequestPB cleanup;
    cleanup.set_op(REMOTE_OPERATION_REMOVE_CACHE);
    for (const auto key : keys) {
        cleanup.mutable_metadata()->add_block_keys(key);
    }
    RemoteOperationResponsePB response;
    EXPECT_TRUE(backend->execute(cleanup, response));
}

TEST_F(KVCMMultiPoolPaceTest, HeterogeneousPoolsRoundTripThroughRealKVCMAndPace) {
    roundTrip(test::makeHeterogeneousRemoteCacheConfig());
}

TEST_F(KVCMMultiPoolPaceTest, SingleGroupRegressionThroughRealKVCMAndPace) {
    roundTrip(test::makeSimpleMhaCacheConfig(1, 16, 8, DataType::TYPE_FP16, 1, 2));
}

TEST_F(KVCMMultiPoolPaceTest, MambaMatchesHeterogeneousFullPrefixAndOnlyTheFinalState) {
    roundTrip(test::makeHeterogeneousRemoteCacheConfig(), 4, true);
}

TEST_F(KVCMMultiPoolPaceTest, SingleGroupPrefixRegressionThroughRealKVCMAndPace) {
    roundTrip(test::makeSimpleMhaCacheConfig(1, 16, 8, DataType::TYPE_FP16, 1, 2), 2);
}

}  // namespace
}  // namespace rtp_llm
