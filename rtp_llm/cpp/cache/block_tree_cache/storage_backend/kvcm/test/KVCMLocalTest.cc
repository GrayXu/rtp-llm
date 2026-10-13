#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

#include <array>
#include <cstdint>
#include <unistd.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace {

double snapshotGauge(kmonitor::MutableMetric* metric) {
    if (metric == nullptr) {
        ADD_FAILURE() << "metric is null";
        return -1;
    }
    kmonitor::MetricsTags tags;
    kmonitor::Metric*     gauge = metric->DeclareMetric(&tags);
    if (gauge == nullptr) {
        ADD_FAILURE() << "metric series is missing";
        return -1;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    gauge->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(gauge));
    if (record.Values().size() != 1) {
        ADD_FAILURE() << "unexpected metric value count=" << record.Values().size();
        return -1;
    }
    return std::stod(record.Values().front()->Value());
}

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
                                                       SpeculativeExecutionConfig{}, nullptr, false, nullptr,
                                                       client_, fence_);
        ASSERT_TRUE(backend_->init(
            environment_->cache_config.topologyPtr(), environment_->pools_by_tag,
            [this](int layer, const std::string& tag, int block) {
                return environmentBuffers(*environment_, layer, tag, block);
            },
            {{"default", host_pool_}},
            [group](int layer, const std::string& tag, int block) {
                return group->convertHostIndexToBuffer(layer, tag, block);
            },
            [this](const CacheKeysType& keys, const std::vector<std::string>& tags,
                   const std::vector<uint32_t>& coordinates, int timeout_ms) {
                return cache_->resolveHostWrite(keys, tags, coordinates, timeout_ms);
            }));
        // Bind the worker resolver without starting a metadata write during fixture setup.
        cache_->storage_backend_ = backend_;
        fence_->observe({{"default", local_block_, 100}, {"default", local_block_, 500, true}});
        EXPECT_CALL(*client_, shutdown()).WillOnce(Invoke([this] {
            if (host_pool_->hasUncertainRemoteIo()) {
                EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            }
        }));
    }

    void TearDown() override {
        cache_.reset();
        if (backend_) {
            backend_->shutdown();
        }
        if (host_pool_) {
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE),
                      host_pool_->hasUncertainRemoteIo() ? 1u : 0u);
        }
        if (client_) {
            EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_.get()));
        }
        if (host_pool_ && !isNullBlockIdx(wire_block_)) {
            host_pool_->decTreeRef(wire_block_, BlockTreeRefType::LOAD);
        }
    }

    RemoteOperationRequestPB request(uint64_t generation) const {
        RemoteOperationRequestPB result;
        result.set_op(REMOTE_OPERATION_WRITE);
        result.set_host_source(true);
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
        std::promise<void> entered, release;
        auto entered_future = entered.get_future();
        auto release_future = release.get_future();
        EXPECT_CALL(*client_, saveKvCachesForTag("default", _, _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&) {
                entered.set_value();
                const bool released = release_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
                EXPECT_TRUE(released);
                if (!success) {
                    client_->save_status = kv_cache_manager::ER_SDK_TIMEOUT;
                }
                return std::make_pair(released && success, kv_cache_manager::UriStrVec{});
            }));
        auto pending = std::async(std::launch::async, [&] {
            RemoteOperationResponsePB response;
            const bool result = backend_->execute(request(1), response);
            return std::make_pair(result, response.transfer_status());
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
        const auto result = pending.get();
        EXPECT_EQ(result.first, success);
        EXPECT_EQ(result.second, success ? REMOTE_TRANSFER_STATUS_SUCCESS : REMOTE_TRANSFER_STATUS_TIMEOUT);
        EXPECT_EQ(host_pool_->hasUncertainRemoteIo(), !success);
        if (success) {
            EXPECT_FALSE(host_pool_->isAllocated(local_block_));
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        } else {
            EXPECT_TRUE(host_pool_->isAllocated(local_block_));
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_FALSE(host_pool_->malloc().has_value());
        }
        EXPECT_NO_THROW(fence_->lockCompute());
        if (!success) {
            backend_->shutdown();
            EXPECT_TRUE(host_pool_->isAllocated(local_block_));
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_FALSE(host_pool_->malloc().has_value());
        }
    }

    std::unique_ptr<BackendEnvironment> environment_;
    std::shared_ptr<HostBlockPool>      host_pool_;
    std::shared_ptr<MockClientWrapper>  client_;
    std::shared_ptr<WorkerCacheIOFence> fence_;
    std::shared_ptr<KVCMStorageBackend> backend_;
    std::unique_ptr<BlockTreeCache>     cache_;
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

TEST_F(KVCMHostWriteTest, ReadsRejectHostSourcesAndStaleDeviceGenerations) {
    EXPECT_CALL(*client_, loadKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_, saveKvCachesForTag(_, _, _, _)).Times(0);
    auto read = request(99);
    read.set_op(REMOTE_OPERATION_READ);
    read.set_block_ids(0, local_block_);
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend_->execute(read, response));
    read.set_host_source(false);
    EXPECT_FALSE(backend_->execute(read, response));
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

TEST_F(KVCMHostWriteTest, TimedOutSubmittedWriteRetainsStorePinAfterShutdown) {
    checkIoPins(false);
}

TEST_F(KVCMHostWriteTest, FailedSubmittedWriteReleasesStorePin) {
    EXPECT_CALL(*client_, saveKvCachesForTag("default", _, _, _))
        .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&) {
            releaseCachedHostBlock();
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            return std::make_pair(false, kv_cache_manager::UriStrVec{});
        }));
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend_->execute(request(1), response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_FALSE(host_pool_->isAllocated(local_block_));
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_FALSE(host_pool_->hasUncertainRemoteIo());
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

    const kv_cache_manager::UriStrVec failed_uris{"failed_uri"};
    EXPECT_CALL(*transfer_ptr, SaveKvCaches(failed_uris, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT,
                                        kv_cache_manager::UriStrVec{"partial_uri"})));
    const auto [failed_save_ok, failed_actual_uris] =
        wrapper.saveKvCachesForTag("default", failed_uris, buffers);
    EXPECT_FALSE(failed_save_ok);
    EXPECT_TRUE(failed_actual_uris.empty());

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

TEST(KVCMLocalTest, GdrRegistrationUsesActualAllocationSpan) {
    auto             environment    = makeBackendEnvironment("kvcm_storage_backend_gdr_span");
    auto             client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const kvcm::ClientWrapper::ConfigMap&,
                             kv_cache_manager::RoleType,
                             const std::vector<kvcm::ClientWrapper::PoolRegistration>& registrations,
                             const std::vector<std::string>&) {
            if (registrations.size() != 1u) {
                ADD_FAILURE() << "expected one GDR pool registration";
                return false;
            }
            const auto& registration = registrations.front();
            EXPECT_EQ(registration.span.base, environment.device_pool->getBaseAddress());
            EXPECT_EQ(registration.span.size, environment.device_pool->getAllocationSizeBytes());
            const long page_size = sysconf(_SC_PAGESIZE);
            EXPECT_GT(page_size, 0);
            if (page_size <= 0) {
                return false;
            }
            EXPECT_EQ(reinterpret_cast<uintptr_t>(registration.span.base) % static_cast<size_t>(page_size), 0u);
            EXPECT_EQ(registration.span.size % static_cast<size_t>(page_size), 0u);
            EXPECT_TRUE(registration.memory_registrations.has_value());
            if (!registration.memory_registrations.has_value()) {
                return false;
            }
            EXPECT_EQ(registration.memory_registrations->gpu.size(), 1u);
            if (registration.memory_registrations->gpu.size() != 1u) {
                return false;
            }
            EXPECT_EQ(registration.memory_registrations->gpu.front().size,
                      environment.device_pool->getAllocationSizeBytes());
            return true;
        }));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment,
                               singleRankConfig(),
                               client_wrapper,
                               /*broadcast_manager=*/nullptr,
                               /*metrics_reporter=*/nullptr,
                               /*gdr_enabled=*/true);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
}

TEST(KVCMLocalTest, LegacyEmptyRemoteResponsePreservesOneofPresence) {
    FunctionResponsePB response;
    response.mutable_remote_response();
    std::string serialized;
    ASSERT_TRUE(response.SerializeToString(&serialized));

    FunctionResponsePB parsed;
    ASSERT_TRUE(parsed.ParseFromString(serialized));
    ASSERT_TRUE(parsed.has_remote_response());
    EXPECT_EQ(parsed.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_UNSPECIFIED);
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
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Return(false));
    EXPECT_FALSE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before);
}

TEST(KVCMLocalTest, PayloadReadTimeoutQuarantinesDestinationBlock) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_timeout");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    auto metrics_reporter =
        std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper, nullptr, metrics_reporter);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    auto* quarantine_metrics = metrics_reporter->getMetricsGroup<RtpLLMRemoteCacheQuarantineMetrics>();
    ASSERT_NE(quarantine_metrics, nullptr);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 0);

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
    client_wrapper->load_status = kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag(_, _, _, _)).Times(0);
    const auto error =
        readResult(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta));
    EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 1);

    EXPECT_FALSE(backend.backend->prepareWrite(makeStorageRequest(environment)));
    const auto blocked_read = readResult(*backend.backend, makeStorageRequest(environment), nullptr);
    EXPECT_EQ(blocked_read.code(), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).Times(0);
    const auto blocked_match = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_TRUE(blocked_match.success);
    EXPECT_EQ(blocked_match.matched_blocks_num, 0u);
    EXPECT_EQ(blocked_match.match_meta, nullptr);

    backend.backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 0);
}

TEST(KVCMLocalTest, PayloadWriteTimeoutAbortsSessionAndRetainsSourceAfterShutdown) {
    auto environment = makeBackendEnvironment("kvcm_storage_backend_write_timeout");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::WriteLocation location;
    location.write_session_id = "timed_out_write";
    location.block_mask = kv_cache_manager::BlockMaskOffset{0};
    location.locations = {{{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, location)));
    EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client, finishWrite(_, _, "timed_out_write", _, _))
        .WillOnce(Invoke([](const auto&, const auto&, const auto&,
                           const kv_cache_manager::BlockMask& mask, const kv_cache_manager::Locations& actual) {
            EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 0u);
            EXPECT_TRUE(actual.empty());
            return true;
        }));
    client->save_status = kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
    const auto refs = environment.device_pool->refCount(environment.block_id);
    EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs + 1);
    EXPECT_TRUE(backend->quarantineActive());
    backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs + 1);
}

}  // namespace
}  // namespace rtp_llm
