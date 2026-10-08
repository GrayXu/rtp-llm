#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

#include <array>

namespace rtp_llm {
namespace {

class KVCMHostWriteTest: public ::testing::Test {
protected:
    void SetUp() override {
        environment_ = std::make_unique<BackendEnvironment>(makeBackendEnvironment("host_write_worker"));
        auto config                      = std::make_shared<HostBlockPoolConfig>();
        config->pool_type                = BlockPoolType::HOST;
        config->pool_name                = "host_write_worker";
        config->physical_block_count     = 3;
        config->payload_bytes            = environment_->cache_config.group("default").kvBlockStrideBytes();
        config->stride_bytes             = 4096;
        config->shared_memory_for_remote = true;
        host_pool_                       = std::make_shared<HostBlockPool>(config);
        ASSERT_TRUE(host_pool_->init());
        const auto blocks = host_pool_->malloc(2);
        ASSERT_TRUE(blocks.has_value());
        wire_block_  = blocks->at(0);
        local_block_ = blocks->at(1);
        host_pool_->incTreeRef(wire_block_, BlockTreeRefType::LOAD);

        auto group = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{environment_->device_pool}, host_pool_, nullptr);
        group->initialize(0, environment_->cache_config.topologyPtr(), {"default"});
        BlockTreeCacheConfig cache_config;
        cache_config.enable_host_cache = true;
        cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({group}, cache_config);
        ASSERT_NE(cache_, nullptr);
        GroupSetResource resource;
        resource.host_block = local_block_;
        cache_->tree()->insertNode({101}, {{resource}}, false, false);
        ASSERT_EQ(host_pool_->treeRefCount(local_block_), 1u);

        client_ = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client_, initForPools(_, kv_cache_manager::RoleType::WORKER, _, _)).WillOnce(Return(true));
        auto parallelism    = singleRankConfig();
        parallelism.tp_size = 2;
        parallelism.tp_rank = 1;
        KVCacheConfig options;
        options.kvcm_server_address = "unused-test-address";
        RuntimeConfig runtime;
        runtime.model_name = "host_write_worker";
        fence_   = std::make_shared<WorkerCacheIOFence>(std::chrono::seconds(1));
        backend_ = std::make_shared<KVCMStorageBackend>(environment_->cache_config,
                                                       options, runtime, parallelism,
                                                       SpeculativeExecutionConfig{}, nullptr, client_, fence_);
        StorageBackend::HostPoolBinding binding;
        binding.pool          = host_pool_;
        binding.member_count  = 1;
        binding.payload_bytes = host_pool_->payloadBytes();
        ASSERT_TRUE(backend_->init(
            environment_->cache_config.topologyPtr(), environment_->pools_by_tag,
            [this](int layer, const std::string& tag, int block) {
                return environmentBuffers(*environment_, layer, tag, block);
            },
            {{"default", host_pool_}},
            [group](int layer, const std::string& tag, int block) {
                return group->convertHostIndexToBuffer(layer, tag, block);
            },
            {{"default", binding}}, {},
            [this](const CacheKeysType& keys, const std::vector<std::string>& tags,
                   const std::vector<uint32_t>& coordinates, int timeout_ms) {
                return cache_->resolveHostWrite(keys, tags, coordinates, timeout_ms);
            }));
        // Bind the worker resolver without starting a metadata write during fixture setup.
        cache_->storage_backend_ = backend_;
        fence_->observe({{"default", local_block_, 100}, {"default", local_block_, 500, true}});
    }

    void TearDown() override {
        if (client_) {
            EXPECT_CALL(*client_, shutdown()).Times(quarantined_client_ ? 0 : 1);
            if (quarantined_client_) {
                ::testing::Mock::AllowLeak(client_.get());
            }
        }
        cache_.reset();
        if (backend_) {
            backend_->shutdown();
        }
        if (client_) {
            EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_.get()));
        }
        if (host_pool_ && !isNullBlockIdx(wire_block_)) {
            host_pool_->decTreeRef(wire_block_, BlockTreeRefType::LOAD);
        }
        if (host_pool_ && host_pool_->hasUncertainRemoteIo() && host_pool_->isAllocated(local_block_)) {
            // The mock SDK has returned, so its deliberately quarantined pin can be released.
            host_pool_->decTreeRef(local_block_, BlockTreeRefType::STORE);
        }
    }

    RemoteOperationRequestPB request(uint64_t generation) const {
        RemoteOperationRequestPB result;
        result.set_op(REMOTE_OPERATION_WRITE_HOST);
        result.add_group_tags("default");
        result.add_block_ids(wire_block_);
        result.add_uris("host_uri");
        result.add_coordinate_ids(0);
        result.add_cache_keys(101);
        result.add_block_generations(generation);
        return result;
    }

    void releaseCachedHostBlock() {
        std::lock_guard<std::mutex> lock(cache_->mutex_);
        const auto path = cache_->tree()->findNode({101});
        ASSERT_EQ(path.size(), 1u);
        cache_->tree()->releaseNode(path.front());
    }

    void checkIoPins(bool success) {
        quarantined_client_ = !success;
        std::promise<void> entered, release;
        auto entered_future = entered.get_future();
        auto release_future = release.get_future();
        EXPECT_CALL(*client_, saveKvCachesForTag("default", _, _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&) {
                entered.set_value();
                const bool released = release_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
                EXPECT_TRUE(released);
                return std::make_pair(released && success, kv_cache_manager::UriStrVec{});
            }));
        auto pending = std::async(std::launch::async, [&] {
            RemoteOperationResponsePB response;
            return backend_->execute(request(1), response);
        });
        const auto status = entered_future.wait_for(std::chrono::seconds(5));
        if (status == std::future_status::ready) {
            releaseCachedHostBlock();
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
            EXPECT_EQ(host_pool_->treeRefCount(local_block_), 1u);
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_FALSE(host_pool_->malloc().has_value());
            auto compute = std::unique_lock<std::recursive_timed_mutex>(fence_->mutex_, std::try_to_lock);
            EXPECT_FALSE(compute.owns_lock());
        }
        release.set_value();
        ASSERT_EQ(status, std::future_status::ready);
        EXPECT_EQ(pending.get(), success);
        EXPECT_EQ(host_pool_->hasUncertainRemoteIo(), !success);
        if (success) {
            EXPECT_FALSE(host_pool_->isAllocated(local_block_));
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        } else {
            EXPECT_TRUE(host_pool_->isAllocated(local_block_));
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
        }
        EXPECT_NO_THROW(fence_->lockCompute());
    }

    std::unique_ptr<BackendEnvironment> environment_;
    std::shared_ptr<HostBlockPool>      host_pool_;
    std::shared_ptr<MockClientWrapper>  client_;
    std::shared_ptr<WorkerCacheIOFence> fence_;
    std::shared_ptr<KVCMStorageBackend> backend_;
    std::unique_ptr<BlockTreeCache>     cache_;
    bool                              quarantined_client_{false};
    BlockIdxType                       wire_block_{NULL_BLOCK_IDX};
    BlockIdxType                       local_block_{NULL_BLOCK_IDX};
};

TEST_F(KVCMHostWriteTest, UsesLocalKeyAndLeavesBothGenerationNamespacesUnchanged) {
    ASSERT_NE(wire_block_, local_block_);
    const auto generations = fence_->generations_;
    int        producer_waits = 0;
    EXPECT_CALL(*client_, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"host_uri"}, _, _))
        .Times(2)
        .WillRepeatedly(Invoke([&](const auto&, const auto&, const kv_cache_manager::BlockBuffers& buffers, const auto&) {
            EXPECT_GT(producer_waits, 0);
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_EQ(buffers.size(), 1u);
            if (buffers.size() == 1 && buffers[0].iovs.size() == 1) {
                EXPECT_EQ(buffers[0].iovs[0].base, host_pool_->blockBuffer(local_block_).addr);
                EXPECT_EQ(buffers[0].iovs[0].type, kv_cache_manager::MemoryType::CPU);
            } else {
                ADD_FAILURE() << "unexpected HOST buffer shape";
            }
            return std::make_pair(true, kv_cache_manager::UriStrVec{});
        }));
    for (uint64_t generation : {1u, 1000u}) {
        {
            auto lease = fence_->lockCompute();
            fence_->setGpuCompletion([&] { ++producer_waits; });
        }
        RemoteOperationResponsePB response;
        EXPECT_TRUE(backend_->execute(request(generation), response));
        EXPECT_EQ(fence_->generations_, generations);
        EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    }
    EXPECT_EQ(producer_waits, 2);
}

TEST_F(KVCMHostWriteTest, HostReadsAndSnapshotWritesStillRejectStaleGenerations) {
    EXPECT_CALL(*client_, loadKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_, saveKvCachesForTag(_, _, _, _)).Times(0);
    auto read = request(99);
    read.set_op(REMOTE_OPERATION_READ_HOST);
    read.set_block_ids(0, local_block_);
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend_->execute(read, response));
    auto snapshot = request(499);
    snapshot.set_op(REMOTE_OPERATION_WRITE);
    snapshot.set_host_source(true);
    snapshot.set_block_ids(0, local_block_);
    EXPECT_FALSE(backend_->execute(snapshot, response));
}

TEST_F(KVCMHostWriteTest, MissingKeyNeverUsesTheOldNumericBlockId) {
    releaseCachedHostBlock();
    EXPECT_CALL(*client_, saveKvCachesForTag(_, _, _, _)).Times(0);
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend_->execute(request(1000), response));
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_FALSE(host_pool_->hasUncertainRemoteIo());
}

TEST_F(KVCMHostWriteTest, InFlightWriteKeepsStorePinAndIoLockUntilCompletion) {
    checkIoPins(true);
}

TEST_F(KVCMHostWriteTest, FailedSubmittedWriteKeepsQuarantinedStorePin) {
    checkIoPins(false);
}

kvcm::KVCMConfigPtr makeLocalConfig() {
    auto location_infos  = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
    auto location_groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
    auto channel         = std::make_shared<kvcm::MetaChannelConfig>(/*retry_time=*/1,
                                                             /*connection_timeout=*/1000,
                                                             /*call_timeout=*/100);
    auto sdk             = std::make_shared<kvcm::SdkWrapperConfig>();
    return std::make_shared<kvcm::KVCMConfig>(/*enable_vipserver=*/false,
                                              /*vipserver_domain=*/"",
                                              /*block_size=*/8,
                                              "instance_group",
                                              "instance_id",
                                              std::vector<std::string>{"direct"},
                                              location_infos,
                                              channel,
                                              sdk,
                                              location_groups,
                                              kvcm::ModelDeployment());
}

TEST(KVCMLocalTest, DirectClientRoutesMetadataAndPayload) {
    auto  factory           = std::make_unique<kvcm::MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<kvcm::MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  meta_client       = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* meta_ptr          = meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);
    auto* transfer_ptr      = transfer_client.get();

    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_)).Times(0);
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _)).WillOnce(Invoke([&meta_client](const auto&, const auto&) {
        return std::move(meta_client);
    }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*meta_ptr, GetStorageConfig()).WillOnce(::testing::ReturnRef(storage_config));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    kvcm::ClientWrapper          wrapper(std::move(factory));
    std::array<char, 64>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(
        wrapper.init({{"", makeLocalConfig()}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));

    const std::vector<int64_t> keys{1, 2};
    kv_cache_manager::Location location;
    location.emplace_back(kv_cache_manager::LocationSpecUnit{"tp0_Ffull", "uri"});
    kv_cache_manager::Locations expected_locations{std::move(location)};
    EXPECT_CALL(*meta_ptr, MatchLocation("match", kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, expected_locations)));
    const auto [match_ok, locations] = wrapper.match(
        "", "match", kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, kv_cache_manager::BlockMaskOffset{0}, {});
    EXPECT_TRUE(match_ok);
    EXPECT_EQ(locations.size(), expected_locations.size());

    kv_cache_manager::UriStrVec    uris{"uri"};
    kv_cache_manager::BlockBuffers buffers;
    EXPECT_CALL(*transfer_ptr, LoadKvCaches(uris, _, _)).WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_TRUE(wrapper.loadKvCachesForTag("default", uris, buffers));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "session";
    EXPECT_CALL(*meta_ptr, StartWrite("start", keys, std::vector<int64_t>{}, std::vector<std::string>{"Ffull"}, 9, 0))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, write_location)));
    const auto [start_ok, actual_write_location] =
        wrapper.getWriteLocation("", "start", keys, {}, {"Ffull"}, /*write_timeout_seconds=*/9);
    EXPECT_TRUE(start_ok);
    EXPECT_EQ(actual_write_location.write_session_id, "session");

    EXPECT_CALL(*transfer_ptr, SaveKvCaches(uris, _, _))
        .WillOnce(Return(
            std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::UriStrVec{"actual_uri"})));
    const auto [save_ok, actual_uris] = wrapper.saveKvCachesForTag("default", uris, buffers);
    EXPECT_TRUE(save_ok);
    EXPECT_EQ(actual_uris, (kv_cache_manager::UriStrVec{"actual_uri"}));

    EXPECT_CALL(*meta_ptr,
                FinishWrite("finish",
                            "session",
                            ::testing::VariantWith<kv_cache_manager::BlockMaskOffset>(::testing::Eq(0u)),
                            ::testing::IsEmpty()))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_TRUE(wrapper.finishWrite("", "finish", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(KVCMLocalTest, InitRejectsMissingTopologyAndInvalidPoolShape) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_init_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, shutdown()).Times(0);
    auto backend  = makeBackend(environment, singleRankConfig(), client_wrapper);
    auto resolver = [&](int layer_id, const std::string&, int block_id) {
        return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
    };

    EXPECT_ANY_THROW(backend->init(nullptr, {}, resolver));
    EXPECT_ANY_THROW(backend->init(environment.cache_config.topologyPtr(), {}, resolver));
    EXPECT_ANY_THROW(backend->init(environment.cache_config.topologyPtr(), {{"default", nullptr}}, resolver));
}

TEST(KVCMLocalTest, MatchAndReadUseReturnedLocation) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_read");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>{101}, _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 1u);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Invoke([](const std::string&,
                            const kv_cache_manager::UriStrVec&,
                            kv_cache_manager::BlockBuffers& buffers,
                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            EXPECT_EQ(buffers.size(), 1u);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
}

TEST(KVCMLocalTest, SuccessfulMatchPreservesLocalPrefixAndReadsOnlyRemoteSuffix) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_partial_local_prefix");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks source_blocks(environment.device_pool, 3);
    const auto&            block_ids         = source_blocks.get();
    const auto             second_block_info = environment.device_pool->convertIndexToBuffer(0, block_ids[1]);
    const auto             third_block_info  = environment.device_pool->convertIndexToBuffer(0, block_ids[2]);
    ASSERT_EQ(second_block_info.size(), 1u);
    ASSERT_EQ(third_block_info.size(), 1u);

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri_102"}},
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri_103"}},
    };
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102, 103}), _, _))
        .WillOnce(Invoke([locations](const std::string&,
                                     const std::string&,
                                     kv_cache_manager::QueryType,
                                     const std::vector<int64_t>&,
                                     const kv_cache_manager::BlockMask& block_mask,
                                     const kv_cache_manager::ForwardContext&) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 1u);
            }
            return std::make_pair(true, locations);
        }));
    auto request =
        makeStorageRequest(environment, /*keys=*/{101, 102, 103}, /*local_matched_blocks=*/1, /*block_ids=*/block_ids);
    auto observation = match(*backend.backend, request);
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 3u);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper,
                loadKvCachesForTag("default", kv_cache_manager::UriStrVec({"read_uri_102", "read_uri_103"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, third_base = third_block_info.front().addr](
                             const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             kv_cache_manager::BlockBuffers& buffers,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            if (buffers.size() != 2u || buffers[0].iovs.size() != 1u || buffers[1].iovs.size() != 1u) {
                ADD_FAILURE() << "KVCM read received an invalid block-buffer shape";
                return false;
            }
            EXPECT_EQ(buffers[0].iovs[0].base, second_base);
            EXPECT_EQ(buffers[1].iovs[0].base, third_base);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(observation.match_meta)));
}

TEST(KVCMLocalTest, MatchFailurePreservesLocalPrefix) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_fallback");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102}), _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::Locations{})));
    auto observation =
        match(*backend.backend, makeStorageRequest(environment, /*keys=*/{101, 102}, /*local_matched_blocks=*/1));
    EXPECT_TRUE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 1u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, InvalidMatchLocationReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_location");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"unknown_spec", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, TP2DuplicateRankMatchReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_duplicate_rank");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).Times(0);
    auto broadcast_manager =
        std::make_shared<BroadcastManager>(std::vector<std::string>{"unused-rank-0", "unused-rank-1"});
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, std::move(broadcast_manager));
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string&, int block_id) {
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    kv_cache_manager::Locations locations{kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "rank0_uri"},
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "duplicate_rank0_uri"},
    }};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    const auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, PayloadReadFailurePropagatesToCompletion) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(0);
    ::testing::Mock::AllowLeak(client_wrapper.get());
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Return(false));
    const auto source_refs = environment.device_pool->refCount(environment.block_id);
    EXPECT_FALSE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
    EXPECT_GT(environment.device_pool->refCount(environment.block_id), source_refs);
    backend->shutdown();
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_wrapper.get()));
}

}  // namespace
}  // namespace rtp_llm
