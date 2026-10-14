#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

struct WorkerEndpoints {
    std::vector<std::shared_ptr<KVCMBroadcastState>>     states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::shared_ptr<BroadcastManager>                    manager;
    explicit WorkerEndpoints(int workers) {
        std::vector<std::string> addresses;
        for (int rank = 0; rank < workers; ++rank) {
            auto state  = std::make_shared<KVCMBroadcastState>();
            auto server = std::make_unique<KVCMBroadcastRpcServer>(rank, state);
            RTP_LLM_CHECK(server->start());
            addresses.push_back(server->address());
            states.push_back(std::move(state));
            servers.push_back(std::move(server));
        }
        manager = std::make_shared<BroadcastManager>(addresses);
        RTP_LLM_CHECK(manager->init());
    }
};

ParallelismConfig cpConfig(int workers) {
    auto result                               = singleRankConfig();
    result.tp_size                            = workers;
    result.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    result.prefill_cp_config.kv_cache_sharded = true;
    return result;
}

BackendHandle canonicalBackend(const BackendEnvironment&                 environment,
                               const ParallelismConfig&                  parallelism,
                               const std::shared_ptr<MockClientWrapper>& client,
                               const std::shared_ptr<BroadcastManager>&  broadcast = nullptr,
                               const std::string&                       client_config = "") {
    KVCacheConfig config;
    config.kvcm_remote_layout  = "canonical_v1";
    config.kvcm_server_address = "unused-test-address";
    config.kvcm_client_config  = client_config;
    RuntimeConfig runtime;
    runtime.model_name = "canonical_test_model";
    return BackendHandle(std::make_unique<KVCMStorageBackend>(
        environment.cache_config, config, runtime, parallelism, SpeculativeExecutionConfig{}, broadcast, false, nullptr,
        client));
}

kv_cache_manager::Locations canonicalLocations(size_t count) {
    kv_cache_manager::Locations result;
    for (size_t i = 0; i < count; ++i) {
        result.push_back({{"v1_Fdefault_h0", "uri_" + std::to_string(i)}});
    }
    return result;
}

std::string automaticCanonicalConfig(const BackendEnvironment& environment) {
    std::string json;
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const auto& configs, auto, const auto&, const auto&) {
            json = autil::legacy::ToJsonString(configs);
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = canonicalBackend(environment, singleRankConfig(), client);
    EXPECT_TRUE(initSingleRank(*backend.backend, environment));
    return json;
}

TEST(KVCMCanonicalTest, CustomConfigAcceptsReorderedSpecs) {
    auto environment = makeMultiGroupBackendEnvironment("canonical_config_order", 2, 1);
    kvcm::ClientWrapper::ConfigMap configs;
    autil::legacy::FromJsonString(configs, automaticCanonicalConfig(environment));
    auto& groups = *configs.at("")->location_spec_groups();
    const auto original = groups;
    for (auto& [name, specs] : groups) {
        std::reverse(specs.begin(), specs.end());
    }
    ASSERT_NE(groups, original);

    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const auto& actual, auto, const auto&, const auto&) {
            EXPECT_EQ(*actual.at("")->location_spec_groups(), groups);
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = canonicalBackend(
        environment, singleRankConfig(), client, nullptr, autil::legacy::ToJsonString(configs));
    EXPECT_TRUE(initSingleRank(*backend.backend, environment));
}

TEST(KVCMCanonicalTest, RemoteDeviceOnlyInitializationDoesNotRequireHostPools) {
    auto environment = makeBackendEnvironment("canonical_device_only");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _))
        .WillOnce(Invoke([](const auto&, auto, const auto& registrations, const auto&) {
            EXPECT_EQ(registrations.size(), 1u);
            if (!registrations.empty()) {
                EXPECT_FALSE(registrations.front().memory_registrations.has_value());
            }
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = canonicalBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
}

TEST(KVCMCanonicalTest, CustomConfigRejectsMissingOrChangedGroups) {
    auto environment = makeBackendEnvironment("canonical_config_groups");
    const auto original = automaticCanonicalConfig(environment);
    for (const bool omit_groups : {false, true}) {
        SCOPED_TRACE(omit_groups);
        kvcm::ClientWrapper::ConfigMap configs;
        autil::legacy::FromJsonString(configs, original);
        auto& groups = *configs.at("")->location_spec_groups();
        if (omit_groups) {
            groups.clear();
        } else {
            groups.begin()->second.front() = "unknown_spec";
        }
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(0);
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = canonicalBackend(
            environment, singleRankConfig(), client, nullptr, autil::legacy::ToJsonString(configs));
        EXPECT_FALSE(initSingleRank(*backend.backend, environment));
    }
}

TEST(KVCMCanonicalTest, StorageInstanceAndRegistrationDoNotEncodeRuntimeTpOrCp) {
    std::string expected;
    for (bool cp : {false, true}) {
        for (int workers : {1, 2, 4}) {
            auto            environment = makeBackendEnvironment("canonical_identity");
            WorkerEndpoints endpoints(workers);
            auto            client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _))
                .WillOnce(Invoke([&](const kvcm::ClientWrapper::ConfigMap& configs,
                                     kv_cache_manager::RoleType,
                                     const std::vector<kvcm::ClientWrapper::PoolRegistration>& pools,
                                     const std::vector<std::string>&) {
                    const auto config = configs.at("");
                    EXPECT_EQ(config->instance_id().rfind("canonical_v1_", 0), 0u);
                    EXPECT_EQ(config->location_spec_infos()->size(), 1u);
                    EXPECT_EQ(config->location_spec_infos()->at("v1_Fdefault_h0"),
                              environment.cache_config.blockSizeBytesForGroup("default"));
                    EXPECT_EQ(pools.front().location_spec_name, "v1_Fdefault_h0");
                    const auto json = autil::legacy::ToJsonString(config);
                    if (expected.empty()) {
                        expected = json;
                    }
                    EXPECT_EQ(json, expected);
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto parallelism = cpConfig(workers);
            if (!cp) {
                parallelism.prefill_cp_config = PrefillCPConfig{};
            }
            auto backend = canonicalBackend(environment, parallelism, client, endpoints.manager);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        }
    }
}

TEST(KVCMCanonicalTest, CpReadsKeepGlobalKeysAndRoundDownIncompleteOrMissingShards) {
    for (int workers : {2, 4}) {
        auto            environment = makeBackendEnvironment("canonical_cp_match");
        WorkerEndpoints endpoints(workers);
        auto            client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = canonicalBackend(environment, cpConfig(workers), client, endpoints.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        CacheKeysType global{101, 102, 103, 104, 105, 106, 107, 108};
        CacheKeysType local;
        for (size_t i = workers - 1; i < global.size(); i += workers) {
            local.push_back(global[i]);
        }
        ScopedReferencedBlocks blocks(environment.device_pool, local.size());
        auto                   request = makeStorageRequest(environment, local, 0, blocks.get());
        request.remote_keys            = std::make_shared<const CacheKeysType>(global);
        for (size_t available : {size_t{8}, size_t{7}, size_t{3}, size_t{0}}) {
            EXPECT_CALL(*client, match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, global, _, _))
                .WillOnce(Return(std::make_pair(true, canonicalLocations(available))));
            const auto observed = match(*backend.backend, request);
            ASSERT_TRUE(observed.success);
            EXPECT_EQ(observed.matched_blocks_num, available / workers);
        }
        auto locations = canonicalLocations(8);
        locations[2].clear();
        EXPECT_CALL(*client, match(_, _, _, global, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        EXPECT_EQ(match(*backend.backend, request).matched_blocks_num, size_t{2} / workers);
        // Full original keys include a partial tail. The local reusable bound is four/two rounds;
        // the remote query must exclude that last key too.
        request.remote_keys =
            std::make_shared<const CacheKeysType>(CacheKeysType{101, 102, 103, 104, 105, 106, 107, 108, 999});
        EXPECT_CALL(*client, match(_, _, _, global, _, _))
            .WillOnce(Return(std::make_pair(true, canonicalLocations(8))));
        auto observed = match(*backend.backend, request);
        ASSERT_EQ(observed.matched_blocks_num, local.size());
        auto promise = std::make_shared<std::promise<bool>>();
        auto future  = promise->get_future();
        backend->read(request, observed.match_meta, [promise](ErrorInfo error) { promise->set_value(error.ok()); });
        EXPECT_TRUE(await(future));
        for (int rank = 0; rank < workers; ++rank) {
            const auto requests = snapshotRequests(endpoints.states[rank]);
            ASSERT_EQ(requests.size(), 1u);
            const auto& rpc = requests.front();
            EXPECT_EQ(rpc.storage_layout(), "canonical_v1");
            ASSERT_EQ(rpc.uris_size(), 8 / workers);
            for (int slot = 0; slot < rpc.uris_size(); ++slot) {
                EXPECT_EQ(rpc.uris(slot), "uri_" + std::to_string(slot * workers + rank));
                EXPECT_EQ(rpc.logical_shards(slot), 0);
                EXPECT_EQ(rpc.block_ids(slot), blocks.get().at(slot));
            }
        }
    }
}

TEST(KVCMCanonicalTest, CpWritesPublishEveryFullGlobalBlockIncludingAnIncompleteRound) {
    auto            environment = makeBackendEnvironment("canonical_cp_tail");
    WorkerEndpoints endpoints(4);
    auto            client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = canonicalBackend(environment, cpConfig(4), client, endpoints.manager);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    const CacheKeysType             keys{101, 102, 103, 104, 105};
    kv_cache_manager::WriteLocation location;
    location.write_session_id = "tail";
    location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    location.locations        = canonicalLocations(keys.size());
    EXPECT_CALL(*client, getWriteLocation(_, _, keys, _, _, _, _)).WillOnce(Return(std::make_pair(true, location)));
    EXPECT_CALL(*client, finishWrite(_, _, "tail", _, _))
        .WillOnce(Invoke([&](const std::string&,
                             const std::string&,
                             const std::string&,
                             const kv_cache_manager::BlockMask& mask,
                             const kv_cache_manager::Locations& actual) {
            EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), keys.size());
            EXPECT_EQ(actual.size(), keys.size());
            EXPECT_EQ(actual[4][0].uri, "actual_rank_0_1");
            return true;
        }));
    auto request            = makeStorageRequest(environment, keys);
    request.keys_are_global = true;
    backend->write(backend->prepareWrite(request));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    for (int rank = 0; rank < 4; ++rank) {
        const auto requests = snapshotRequests(endpoints.states[rank]);
        ASSERT_EQ(requests.size(), 1u);
        EXPECT_EQ(requests[0].uris_size(), rank == 0 ? 2 : 1);
        EXPECT_EQ(requests[0].uris(0), "uri_" + std::to_string(rank));
    }
}

TEST(KVCMCanonicalTest, HybridCpPublishesSparseGlobalStatesAndReadsOnlyMatchedEndpoint) {
    for (int workers : {2, 4}) {
        auto            environment = makeMultiGroupBackendEnvironment("canonical_hybrid_cp", 1, 1);
        WorkerEndpoints endpoints(workers);
        auto            client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = canonicalBackend(environment, cpConfig(workers), client, endpoints.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));

        const size_t  count = 3 * workers + 1;  // Last full token block is outside a complete CP round.
        CacheKeysType global;
        for (size_t i = 0; i < count; ++i) {
            global.push_back(100 + i);
        }
        ScopedReferencedBlocks full(environment.pools_by_tag.at("full0"), 4);
        ScopedReferencedBlocks states(environment.pools_by_tag.at("linear0"), count / 2);
        StorageRequest         write;
        write.keys            = std::make_shared<const CacheKeysType>(global);
        write.keys_are_global = true;
        write.handles.resize(count);
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "hybrid_sparse";
        allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        size_t state                = 0;
        for (size_t i = 0; i < count; ++i) {
            write.handles[i].push_back({"full0", full.get().at(i / workers)});
            kv_cache_manager::Location location{{"v1_Ffull0_h0", "full_" + std::to_string(i)}};
            if ((i + 1) % 2 == 0) {
                write.handles[i].push_back({"linear0", states.get().at(state++)});
                location.push_back({"v1_Llinear0_h0", "state_" + std::to_string(i)});
            }
            allocation.locations.push_back(std::move(location));
        }
        kv_cache_manager::Locations published;
        EXPECT_CALL(*client, getWriteLocation(_, _, global, _, _, _, _))
            .WillOnce(Invoke([&](const std::string&,
                                 const std::string&,
                                 const CacheKeysType&,
                                 const std::vector<int64_t>&,
                                 const std::vector<std::string>& groups,
                                 int64_t,
                                 int32_t) {
                EXPECT_EQ(groups.size(), count);
                if (groups.size() != count) {
                    return std::make_pair(false, allocation);
                }
                for (size_t i = 0; i < count; ++i) {
                    EXPECT_EQ(groups[i], (i + 1) % 2 == 0 ? "Ffull0Llinear0" : "Ffull0");
                }
                return std::make_pair(true, allocation);
            }));
        EXPECT_CALL(*client, finishWrite(_, _, "hybrid_sparse", _, _))
            .WillOnce(Invoke([&](const std::string&,
                                 const std::string&,
                                 const std::string&,
                                 const kv_cache_manager::BlockMask& mask,
                                 const kv_cache_manager::Locations& actual) {
                EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), count);
                published = actual;
                return true;
            }));
        backend->write(backend->prepareWrite(write));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        ASSERT_EQ(published.size(), count);

        CacheKeysType local;
        for (size_t i = workers - 1; i < count; i += workers) {
            local.push_back(global[i]);
        }
        ScopedReferencedBlocks destination_full(environment.pools_by_tag.at("full0"), 3);
        ScopedReferencedBlocks destination_state(environment.pools_by_tag.at("linear0"), 3);
        auto                   request = makeStorageRequest(environment, local, 0, destination_full.get());
        for (size_t i = 0; i < local.size(); ++i) {
            request.handles[i][0].tag = "full0";
            request.handles[i].push_back({"linear0", destination_state.get()[i]});
        }
        request.remote_keys = std::make_shared<const CacheKeysType>(global);
        const CacheKeysType queried(global.begin(), global.end() - 1);
        for (size_t available : {count - 1, count - 2}) {
            kv_cache_manager::Locations locations(published.begin(), published.begin() + available);
            EXPECT_CALL(*client, match(_, _, _, queried, _, _)).WillOnce(Return(std::make_pair(true, locations)));
            auto observed = match(*backend.backend, request);
            ASSERT_TRUE(observed.success);
            const size_t rounds = available / workers;
            ASSERT_EQ(observed.matched_blocks_num, rounds);
            auto load = request;
            load.keys = std::make_shared<const CacheKeysType>(local.begin(), local.begin() + rounds);
            load.handles.resize(rounds);
            std::vector<size_t> before;
            for (const auto& endpoint : endpoints.states) {
                before.push_back(snapshotRequests(endpoint).size());
            }
            ASSERT_TRUE(read(*backend.backend, load, observed.match_meta));
            for (int rank = 0; rank < workers; ++rank) {
                const auto requests = snapshotRequests(endpoints.states[rank]);
                ASSERT_EQ(requests.size(), before[rank] + 1);
                const auto& rpc         = requests.back();
                size_t      state_reads = 0;
                for (int i = 0; i < rpc.uris_size(); ++i) {
                    if (rpc.group_tags(i) == "linear0") {
                        ++state_reads;
                        EXPECT_EQ(rpc.block_ids(i), destination_state.get()[rounds - 1]);
                        EXPECT_EQ(rpc.uris(i), published[rounds * workers - 1][1].uri);
                    }
                }
                EXPECT_EQ(state_reads, 1u);
            }
        }
    }
}

TEST(KVCMCanonicalTest, IncompleteStartWriteAndFailedWorkerNeverCommitReadableKeys) {
    for (bool missing : {true, false}) {
        auto            environment = makeBackendEnvironment("canonical_failure");
        WorkerEndpoints endpoints(2);
        endpoints.states[1]->fail = !missing;
        auto client               = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(missing ? 1 : 0);
        auto backend = canonicalBackend(environment, cpConfig(2), client, endpoints.manager);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        kv_cache_manager::WriteLocation location;
        location.write_session_id = "abort";
        location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        location.locations        = canonicalLocations(2);
        if (missing) {
            location.locations[1].clear();
        }
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, location)));
        EXPECT_CALL(*client, finishWrite(_, _, "abort", _, _))
            .WillOnce(Invoke([](const std::string&,
                                const std::string&,
                                const std::string&,
                                const kv_cache_manager::BlockMask& mask,
                                const kv_cache_manager::Locations& locations) {
                EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 0u);
                EXPECT_TRUE(locations.empty());
                return true;
            }));
        auto request        = makeStorageRequest(environment, {102});
        request.remote_keys = std::make_shared<const CacheKeysType>(CacheKeysType{101, 102});
        backend->write(backend->prepareWrite(request));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        backend->shutdown();
        EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client.get()));
        if (!missing) {
            ::testing::Mock::AllowLeak(client.get());
        }
    }
}

TEST(KVCMCanonicalTest, RealWorkerRpcBorrowsUnallocatedSpanAndRejectsReusedAllocation) {
    auto environment = makeBackendEnvironment("canonical_real_worker");
    auto client      = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, kv_cache_manager::RoleType::WORKER, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto parallelism    = singleRankConfig();
    parallelism.tp_size = 2;
    parallelism.tp_rank = 1;
    auto          fence = std::make_shared<WorkerCacheIOFence>();
    KVCacheConfig config;
    config.kvcm_remote_layout = "canonical_v1";
    RuntimeConfig runtime;
    runtime.model_name = "worker_test";
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(
        environment.cache_config, config, runtime, parallelism, SpeculativeExecutionConfig{}, nullptr, false, nullptr,
        client, fence));
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    const int block = environment.block_id == 1 ? 2 : 1;
    ASSERT_TRUE(environment.device_pool->validBlock(block));
    ASSERT_FALSE(environment.device_pool->isAllocated(block));
    EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"worker_uri"}, _, _))
        .WillOnce(Return(true));
    auto state     = std::make_shared<KVCMBroadcastState>();
    state->execute = [&](const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        return backend->execute(request, response);
    };
    KVCMBroadcastRpcServer server(1, state);
    ASSERT_TRUE(server.start());
    BroadcastManager broadcast({server.address()});
    ASSERT_TRUE(broadcast.init());
    FunctionRequestPB request;
    auto*             remote = request.mutable_remote_request();
    remote->set_op(REMOTE_OPERATION_READ);
    remote->set_storage_layout("canonical_v1");
    remote->add_group_tags("default");
    remote->add_block_ids(block);
    remote->add_uris("worker_uri");
    remote->add_logical_shards(0);
    remote->add_block_generations(1);
    const auto call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                         const std::shared_ptr<grpc::ClientContext>& context,
                         const FunctionRequestPB&                    request,
                         grpc::CompletionQueue*                      queue) {
        return stub->AsyncExecuteFunction(context.get(), request, queue);
    };
    auto result = broadcast.broadcast<FunctionRequestPB, FunctionResponsePB>({request}, 5000, call);
    result->waitDone();
    EXPECT_TRUE(result->success());
    EXPECT_FALSE(environment.device_pool->isAllocated(block));
    fence->observe({{"default", block, 2}});
    result = broadcast.broadcast<FunctionRequestPB, FunctionResponsePB>({request}, 5000, call);
    result->waitDone();
    ASSERT_TRUE(result->success());
    const auto responses = result->responses();
    ASSERT_EQ(responses.size(), 1u);
    ASSERT_TRUE(responses.front().has_remote_response());
    EXPECT_EQ(responses.front().remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
}

TEST(KVCMCanonicalTest, RpcRejectsLegacyLayoutBeforeTransferringCanonicalBuffers) {
    auto environment = makeBackendEnvironment("canonical_rpc_guard");
    auto client      = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = canonicalBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    request.add_uris("legacy");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(request, response));
}

class KVCMCanonicalTreeTest: public DeviceTestBase {
protected:
    void writeHostTree(int workers, bool cp, bool eviction_source_paths = false, bool remote_exists = false) {
        const auto config = test::makeSimpleMhaCacheConfig(2, 8, 8, DataType::TYPE_INT8, 1, 2);
        const CacheKeysType original{191, 307, 419, 523, 631, 743, 857, 967, 1087};
        // CP input retains a partial tail outside the two ready HOST rounds.
        const CacheKeysType global(original.begin(), original.begin() + (cp ? 2 * workers + 1 : 2));
        const CPSlotMapper mapper(0, cp ? workers : 1, config.seq_size_per_block);
        const auto local_keys = mapper.canonicalCacheKeys(global);
        const CacheKeysType published_keys(global.begin(), global.begin() + local_keys.size() * mapper.cpSize());
        std::vector<CacheKeysType> remote_key_groups;
        const std::vector<CacheKeysType> no_remote_key_groups;
        if (cp) {
            remote_key_groups.reserve(local_keys.size());
            for (size_t local_index = 0; local_index < local_keys.size(); ++local_index) {
                const size_t begin = local_index * static_cast<size_t>(workers);
                remote_key_groups.emplace_back(global.begin() + begin, global.begin() + begin + workers);
            }
        }
        std::vector<std::atomic<size_t>> resolver_calls(workers);
        for (auto& count : resolver_calls) {
            count.store(0);
        }
        WorkerEndpoints endpoints(workers);
        std::vector<std::shared_ptr<MockClientWrapper>> clients;
        std::vector<std::shared_ptr<KVCMStorageBackend>> backends;
        std::vector<CoordinatorCacheManagerPtr> allocators;
        std::vector<BlockTreeCachePtr> caches;
        std::vector<std::vector<std::vector<GroupSetResource>>> resources(workers);
        std::vector<std::vector<BlockIdxType>> local_blocks(workers);
        KVCacheConfig options;
        options.kvcm_remote_layout = "canonical_v1";
        options.enable_remote_cache = true;
        options.enable_memory_cache = true;
        options.memory_cache_size_mb = 1;
        RuntimeConfig runtime;
        runtime.model_name = "canonical_host_tree";

        for (int rank = 0; rank < workers; ++rank) {
            auto parallelism = cp ? cpConfig(workers) : singleRankConfig();
            parallelism.tp_size = workers;
            parallelism.tp_rank = rank;
            auto allocator = std::make_shared<CoordinatorCacheManager>(config);
            allocator->setUseDeviceMallocBlockPool(true);
            ASSERT_TRUE(allocator->init());
            if (cp) {
                allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(rank, workers, config.seq_size_per_block));
            }
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
            if (rank != 0) {
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(0);
            }
            auto backend = std::make_shared<KVCMStorageBackend>(
                config, options, runtime, parallelism, SpeculativeExecutionConfig{}, endpoints.manager, false,
                nullptr, client);
            auto cache = createBlockTreeCache(config, options, allocator, parallelism, backend);
            ASSERT_NE(cache, nullptr);
            ASSERT_EQ(cache->groupSets().size(), 1u);
            const auto group = cache->groupSets().front();
            const auto host_pool = group->hostPool();
            ASSERT_NE(host_pool, nullptr);
            std::vector<BlockIdxType> unused_blocks;
            for (int i = 0; i < rank; ++i) {
                const auto block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
                ASSERT_FALSE(isNullBlockIdx(block));
                unused_blocks.push_back(block);
            }
            for (size_t slot = 0; slot < local_keys.size(); ++slot) {
                GroupSetResource resource;
                resource.host_block = group->allocateSingleBlock(Tier::HOST, BlockTreeRefType::LOAD);
                ASSERT_FALSE(isNullBlockIdx(resource.host_block));
                local_blocks[rank].push_back(resource.host_block);
                resources[rank].push_back({resource});
                const auto buffer = host_pool->blockBuffer(resource.host_block);
                std::memset(buffer.addr, 17 + rank * 37 + slot * 11, buffer.payload_bytes);
                if (eviction_source_paths) {
                    resource.candidate_meta.admission_seq = 1000 + slot;
                    if (rank == 0) {
                        resource.transfer_state = GroupSetTransferState::DEMOTING;
                        resource.host_eviction_lease_id = 1000 + slot;
                    }
                }
            }
            group->releaseBlocks(Tier::HOST, unused_blocks, BlockTreeRefType::LOAD);
            if (rank != 0 || eviction_source_paths) {
                // Workers already hold their local pages under the shared round endpoint keys.
                cache->tree()->insertNode(local_keys,
                                          resources[rank],
                                          true,
                                          false,
                                          eviction_source_paths ? remote_key_groups : no_remote_key_groups);
                group->releaseBlocks(Tier::HOST, local_blocks[rank], BlockTreeRefType::LOAD);
                if (rank != 0) {
                    EXPECT_NE(local_blocks[rank], local_blocks[0]);
                }
            }
            const auto factory_resolver = backend->hostWriteResolver();
            ASSERT_TRUE(static_cast<bool>(factory_resolver));
            backend->host_write_resolver_ = [factory_resolver, &resolver_calls, rank](
                const CacheKeysType& keys, const std::vector<std::string>& tags,
                const std::vector<uint32_t>& coordinates,
                int timeout_ms,
                bool keys_are_local,
                uint64_t lease_id) {
                resolver_calls[rank].fetch_add(1);
                return factory_resolver(keys, tags, coordinates, timeout_ms, keys_are_local, lease_id);
            };
            endpoints.states[rank]->execute = [backend](const auto& request, auto& response) {
                const bool success = backend->execute(request, response);
                EXPECT_TRUE(success);
                EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
                return success;
            };
            kv_cache_manager::UriStrVec expected_uris;
            if (!remote_exists && (cp || rank == 0)) {
                for (size_t slot = 0; slot < local_keys.size(); ++slot) {
                    expected_uris.push_back("uri_" + std::to_string(cp ? slot * workers + rank : slot));
                }
                EXPECT_CALL(*client, saveKvCachesForTag("default", expected_uris, _, _))
                    .WillOnce(Invoke([host_pool, blocks = local_blocks[rank], rank](
                                         const auto&, const auto& uris, const auto& buffers, const auto&) {
                        EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE),
                                  blocks.size());
                        EXPECT_EQ(buffers.size(), blocks.size());
                        if (buffers.size() != blocks.size()) {
                            return std::make_pair(false, kv_cache_manager::UriStrVec{});
                        }
                        kv_cache_manager::UriStrVec actual;
                        for (size_t slot = 0; slot < buffers.size(); ++slot) {
                            const auto host = host_pool->blockBuffer(blocks[slot]);
                            const auto begin = reinterpret_cast<uintptr_t>(host.addr);
                            const auto end = begin + host.payload_bytes;
                            size_t bytes = 0;
                            EXPECT_FALSE(buffers[slot].iovs.empty());
                            for (const auto& iov : buffers[slot].iovs) {
                                EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::CPU);
                                const auto address = reinterpret_cast<uintptr_t>(iov.base);
                                EXPECT_GE(address, begin);
                                EXPECT_LE(address + iov.size, end);
                                if (address >= begin && address <= end && iov.size <= end - address) {
                                    const auto* data = static_cast<const uint8_t*>(iov.base);
                                    EXPECT_TRUE(std::all_of(data, data + iov.size, [rank, slot](uint8_t value) {
                                        return value == static_cast<uint8_t>(17 + rank * 37 + slot * 11);
                                    }));
                                }
                                bytes += iov.size;
                            }
                            EXPECT_EQ(bytes, host.payload_bytes);
                            actual.push_back("actual_" + uris[slot]);
                        }
                        return std::make_pair(true, std::move(actual));
                    }));
            } else {
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
            }
            clients.push_back(std::move(client));
            backends.push_back(std::move(backend));
            allocators.push_back(std::move(allocator));
            caches.push_back(std::move(cache));
        }

        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "host_tree";
        allocation.block_mask = kv_cache_manager::BlockMaskOffset{remote_exists ? published_keys.size() : 0};
        if (!remote_exists) {
            allocation.locations = canonicalLocations(published_keys.size());
        }
        EXPECT_CALL(*clients[0], getWriteLocation(_, _, published_keys, _, _, _, _))
            .WillOnce(Return(std::make_pair(true, allocation)));
        EXPECT_CALL(*clients[0], finishWrite(_, _, "host_tree", _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, const auto&,
                                 const kv_cache_manager::BlockMask& mask, const kv_cache_manager::Locations& actual) {
                if (remote_exists) {
                    EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 0u);
                    EXPECT_TRUE(actual.empty());
                    return true;
                }
                EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), published_keys.size());
                EXPECT_EQ(actual.size(), published_keys.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    EXPECT_EQ(actual[i].size(), 1u);
                    if (actual[i].size() == 1) {
                        EXPECT_EQ(actual[i][0].spec_name, "v1_Fdefault_h0");
                        EXPECT_EQ(actual[i][0].uri, "actual_uri_" + std::to_string(i));
                    }
                }
                return true;
            }));
        auto remote_write = std::make_shared<StorageRequest>();
        remote_write->source_tier = Tier::HOST;
        if (eviction_source_paths) {
            remote_write->keys_are_global = true;
            remote_write->keys = std::make_shared<const CacheKeysType>(published_keys);
            remote_write->handles.resize(published_keys.size());
            remote_write->host_write_source_paths.reserve(published_keys.size());
            for (size_t global_index = 0; global_index < published_keys.size(); ++global_index) {
                const size_t local_index = global_index / static_cast<size_t>(mapper.cpSize());
                remote_write->handles[global_index].push_back({"default", local_blocks[0][local_index]});
                HostWriteSourcePath source_path;
                source_path.local_cache_keys.assign(local_keys.begin(), local_keys.begin() + local_index + 1);
                source_path.coordinate = static_cast<uint32_t>(local_index);
                source_path.host_eviction_lease_id = 1000 + local_index;
                remote_write->host_write_source_paths.push_back(std::move(source_path));
            }
        } else {
            remote_write->keys = std::make_shared<const CacheKeysType>(local_keys);
            remote_write->remote_keys = std::make_shared<const CacheKeysType>(global);
            for (const auto block : local_blocks[0]) {
                remote_write->handles.push_back({{"default", block}});
            }
        }
        if (eviction_source_paths) {
            auto write_task = backends[0]->prepareWrite(*remote_write);
            ASSERT_TRUE(static_cast<bool>(write_task));
            ASSERT_TRUE(backends[0]->write(std::move(write_task)));
        } else {
            caches[0]->insert(local_keys, resources[0], Tier::HOST, remote_write);
            caches[0]->groupSets().front()->releaseBlocks(Tier::HOST, local_blocks[0], BlockTreeRefType::LOAD);
        }
        for (int rank = 0; rank < workers; ++rank) {
            ASSERT_TRUE(waitForBackendOperationsForTest(*backends[rank]));
            const auto requests = snapshotRequests(endpoints.states[rank]);
            const bool control_rpcs = eviction_source_paths && workers > 1;
            const size_t rpc_count = workers > 1 ? (control_rpcs ? 2u : 0u) + (remote_exists ? 0u : 1u) : 0u;
            ASSERT_EQ(requests.size(), rpc_count);
            if (!remote_exists && workers > 1) {
                const auto& rpc = requests[control_rpcs ? 1 : 0];
                EXPECT_EQ(rpc.op(), REMOTE_OPERATION_WRITE);
                EXPECT_TRUE(rpc.host_source());
                EXPECT_EQ(rpc.storage_layout(), "canonical_v1");
                EXPECT_EQ(CacheKeysType(rpc.cache_keys().begin(), rpc.cache_keys().end()), published_keys);
                const bool writer = cp || rank == 0;
                const size_t count = writer ? local_keys.size() : 0;
                ASSERT_EQ(rpc.group_tags_size(), count);
                ASSERT_EQ(rpc.block_ids_size(), count);
                ASSERT_EQ(rpc.uris_size(), count);
                ASSERT_EQ(rpc.coordinate_ids_size(), count);
                ASSERT_EQ(rpc.logical_shards_size(), count);
                ASSERT_EQ(rpc.host_write_source_paths_size(), eviction_source_paths ? count : 0u);
                EXPECT_EQ(resolver_calls[rank].load(), writer ? (eviction_source_paths ? count : 1u) : 0u);
                for (size_t slot = 0; slot < count; ++slot) {
                    EXPECT_EQ(rpc.group_tags(slot), "default");
                    EXPECT_EQ(rpc.block_ids(slot), local_blocks[0][slot]);
                    EXPECT_EQ(rpc.coordinate_ids(slot), cp ? slot * workers + rank : slot);
                    if (eviction_source_paths) {
                        const auto& source_path = rpc.host_write_source_paths(slot);
                        EXPECT_EQ(source_path.local_cache_keys_size(), static_cast<int>(slot + 1));
                        EXPECT_EQ(CacheKeysType(source_path.local_cache_keys().begin(),
                                                source_path.local_cache_keys().end()),
                                  CacheKeysType(local_keys.begin(), local_keys.begin() + slot + 1));
                        EXPECT_EQ(source_path.coordinate(), slot);
                        EXPECT_EQ(source_path.host_eviction_lease_id(), 1000 + slot);
                    }
                }
            } else {
                EXPECT_EQ(resolver_calls[rank].load(), 0u);
            }
            if (control_rpcs) {
                const auto& reservation = requests.front();
                EXPECT_EQ(reservation.op(), REMOTE_OPERATION_HOST_EVICTION_SETTLEMENT);
                EXPECT_EQ(reservation.host_write_settlement(),
                          rank == 0 ? REMOTE_HOST_WRITE_SETTLEMENT_UNSPECIFIED :
                                      REMOTE_HOST_WRITE_SETTLEMENT_RESERVE);
                EXPECT_EQ(reservation.host_write_source_paths_size(), rank == 0 ? 0 : local_keys.size());
                const auto& settlement = requests.back();
                EXPECT_EQ(settlement.op(), REMOTE_OPERATION_HOST_EVICTION_SETTLEMENT);
                EXPECT_EQ(settlement.host_write_settlement(),
                          rank == 0 ? REMOTE_HOST_WRITE_SETTLEMENT_UNSPECIFIED :
                                      REMOTE_HOST_WRITE_SETTLEMENT_COMMIT);
                EXPECT_EQ(settlement.host_write_source_paths_size(), rank == 0 ? 0 : local_keys.size());
                if (rank != 0) {
                    for (int index = 0; index < settlement.host_write_source_paths_size(); ++index) {
                        const auto& source_path = settlement.host_write_source_paths(index);
                        EXPECT_EQ(source_path.host_eviction_lease_id(), 1000 + index);
                        EXPECT_EQ(source_path.group_tags_size(), 1);
                        if (source_path.group_tags_size() == 1) {
                            EXPECT_EQ(source_path.group_tags(0), "default");
                        }
                    }
                }
            }
            const auto host_pool = caches[rank]->groupSets().front()->hostPool();
            EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
            const size_t expected_cache_refs = eviction_source_paths && rank != 0 ? 0u : local_keys.size();
            EXPECT_EQ(host_pool->referencedBlocksNum(BlockTreeRefType::CACHE), expected_cache_refs);
            if (eviction_source_paths && rank != 0) {
                EXPECT_TRUE(caches[rank]->tree()->findNode(local_keys).empty());
            }
            EXPECT_FALSE(host_pool->hasUncertainRemoteIo());
        }
    }
};

TEST_F(KVCMCanonicalTreeTest, TpMqaReplicaAcceptsEmptyHostWriteWithoutResolvingOrTransferring) {
    writeHostTree(2, false);
}

TEST_F(KVCMCanonicalTreeTest, CpHostInsertPublishesCompleteRoundsFromRankLocalTreeBlocks) {
    for (int workers : {2, 4}) {
        SCOPED_TRACE(workers);
        writeHostTree(workers, true);
    }
}

TEST_F(KVCMCanonicalTreeTest, HostEvictionUsesRankLocalCanonicalSourcePathsAndLeases) {
    writeHostTree(1, false, true);
    for (int workers : {2, 4}) {
        SCOPED_TRACE(workers);
        writeHostTree(workers, false, true);
        writeHostTree(workers, true, true);
    }
}

TEST_F(KVCMCanonicalTreeTest, HostEvictionSettlesWorkerSourcesWhenRemoteKeysAlreadyExist) {
    writeHostTree(1, false, true, true);
    for (int workers : {2, 4}) {
        SCOPED_TRACE(workers);
        writeHostTree(workers, false, true, true);
        writeHostTree(workers, true, true, true);
    }
}

class CanonicalHostIOTest: public ::testing::Test {
protected:
    void SetUp() override {
        environment_.cache_config = test::makeSimpleMhaCacheConfig(2, 8, 8, DataType::TYPE_INT8, 2, 2);
        initializeEnvironmentPools(environment_, "canonical_host", 8);
        auto config = std::make_shared<HostBlockPoolConfig>();
        config->pool_type = BlockPoolType::HOST;
        config->pool_name = "canonical_host";
        config->physical_block_count = 4;
        config->payload_bytes = environment_.cache_config.blockSizeBytesForGroup("default");
        config->stride_bytes = 4096;
        host_pool_ = std::make_shared<HostBlockPool>(config);
        ASSERT_TRUE(host_pool_->init());
        const auto block = host_pool_->malloc();
        ASSERT_TRUE(block.has_value());
        host_block_ = *block;
        host_pool_->incTreeRef(host_block_, BlockTreeRefType::LOAD);
        group_ = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{environment_.device_pool}, host_pool_, nullptr);
        group_->initialize(0, environment_.cache_config.topologyPtr(), {"default"});
        client_ = std::make_shared<MockClientWrapper>();
    }

    void TearDown() override {
        if (backend_) {
            backend_->shutdown();
        }
        if (host_pool_ && !isNullBlockIdx(host_block_)) {
            host_pool_->decTreeRef(host_block_, BlockTreeRefType::LOAD);
        }
    }

    void initialize(bool gdr = false, bool retain_client = false) {
        EXPECT_CALL(*client_, initForPools(_, _, _, _))
            .WillOnce(Invoke([&](const auto&, auto, const auto& registrations, const auto& tags) {
                EXPECT_EQ(registrations.size(), 1u);
                EXPECT_EQ(tags, (std::vector<std::string>{"default"}));
                EXPECT_EQ(registrations.front().location_spec_name, "v1_Fdefault_h0");
                EXPECT_EQ(registrations.front().memory_registrations.has_value(), gdr);
                if (gdr) {
                    EXPECT_EQ(registrations.front().memory_registrations->gpu.size(), 1u);
                }
                return true;
            }));
        EXPECT_CALL(*client_, shutdown()).Times(retain_client ? 0 : 1);
        KVCacheConfig options;
        options.kvcm_remote_layout = "canonical_v1";
        RuntimeConfig runtime;
        runtime.model_name = "canonical_host";
        backend_ = std::make_shared<KVCMStorageBackend>(environment_.cache_config, options, runtime,
                                                       singleRankConfig(), SpeculativeExecutionConfig{},
                                                       nullptr, gdr, nullptr, client_);
        ASSERT_TRUE(backend_->init(
            environment_.cache_config.topologyPtr(), environment_.pools_by_tag,
            [this](int layer, const std::string& tag, int block) {
                return environmentBuffers(environment_, layer, tag, block);
            },
            {{"default", host_pool_}},
            [this](int layer, const std::string& tag, int block) {
                return group_->convertHostIndexToBuffer(layer, tag, block);
            },
            [this](const CacheKeysType& keys, const std::vector<std::string>& tags,
                   const std::vector<uint32_t>& coordinates, int, bool, uint64_t) {
                StorageRequest request;
                request.source_tier = Tier::HOST;
                request.keys = std::make_shared<CacheKeysType>(tags.size(), keys.at(0));
                request.handles.resize(tags.size());
                std::vector<BlockIdxType> blocks;
                for (size_t i = 0; i < tags.size(); ++i) {
                    EXPECT_EQ(keys.at(coordinates.at(i)), 101);
                    request.handles[i].push_back({tags[i], host_block_});
                    blocks.push_back(host_block_);
                }
                return StorageBackend::HostWriteResolution{backend_->prepareWrite(std::move(request)), std::move(blocks)};
            }));
    }

    RemoteOperationRequestPB request(bool host_write = false) const {
        RemoteOperationRequestPB result;
        result.set_op(host_write ? REMOTE_OPERATION_WRITE : REMOTE_OPERATION_READ);
        result.set_host_source(host_write);
        result.set_storage_layout("canonical_v1");
        result.add_cache_keys(101);
        for (int shard = 0; shard < 2; ++shard) {
            result.add_group_tags("default");
            result.add_block_ids(host_write ? host_block_ + 1 : environment_.block_id);
            result.add_uris("shard_" + std::to_string(shard));
            result.add_coordinate_ids(0);
            result.add_logical_shards(shard);
            result.add_block_generations(environment_.device_pool->blockAllocationGeneration(environment_.block_id));
        }
        return result;
    }

    static uint8_t headValue(size_t head) { return static_cast<uint8_t>(17 + 37 * head); }
    BackendEnvironment environment_;
    std::shared_ptr<HostBlockPool> host_pool_;
    std::shared_ptr<FullGroupSet> group_;
    std::shared_ptr<MockClientWrapper> client_;
    std::shared_ptr<KVCMStorageBackend> backend_;
    BlockIdxType host_block_{NULL_BLOCK_IDX};
};

TEST_F(CanonicalHostIOTest, ExplicitHostWritesUseSameTagCpuHeadSlicesAndLocalStorePins) {
    initialize();
    for (int layer : environment_.cache_config.layerIdsForGroup("default")) {
        for (const auto& part : group_->convertHostIndexToBuffer(layer, "default", host_block_)) {
            for (size_t component = 0; component < 2; ++component) {
                for (size_t head = 0; head < 2; ++head) {
                    std::memset(static_cast<uint8_t*>(part.addr) + component * part.size_bytes / 2
                                    + head * part.size_bytes / 4,
                                headValue(head), part.size_bytes / 4);
                }
            }
        }
    }
    EXPECT_CALL(*client_, saveKvCachesForTag("default", _, _, _))
        .WillOnce(Invoke([this](const auto&, const auto&, const auto& buffers, const auto& trace) {
            EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
            EXPECT_EQ(buffers.size(), 2u);
            EXPECT_NE(trace, nullptr);
            if (trace) {
                EXPECT_FALSE(trace->need_print);
            }
            for (size_t shard = 0; shard < buffers.size(); ++shard) {
                EXPECT_EQ(buffers[shard].iovs.size(), 8u);
                for (const auto& iov : buffers[shard].iovs) {
                    EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::CPU);
                    const auto* bytes = static_cast<const uint8_t*>(iov.base);
                    EXPECT_TRUE(std::all_of(bytes, bytes + iov.size,
                                           [&](uint8_t value) { return value == headValue(shard); }));
                }
            }
            return std::make_pair(true, kv_cache_manager::UriStrVec{});
        }));
    RemoteOperationResponsePB response;
    ASSERT_TRUE(backend_->execute(request(/*host_write=*/true), response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
}

TEST_F(CanonicalHostIOTest, AllReadsUseGpuIovsWithOrWithoutGdr) {
    for (const bool gdr : {false, true}) {
        SCOPED_TRACE(gdr);
        initialize(gdr);
        EXPECT_CALL(*client_, loadKvCachesForTag("default", _, _, _))
            .WillOnce(Invoke([](const auto&, const auto&, auto& buffers, const auto&) {
                EXPECT_EQ(buffers.size(), 2u);
                for (const auto& buffer : buffers) {
                    EXPECT_EQ(buffer.iovs.size(), 8u);
                    for (const auto& iov : buffer.iovs) {
                        EXPECT_EQ(iov.type, kv_cache_manager::MemoryType::GPU);
                    }
                }
                return true;
            }));
        RemoteOperationResponsePB response;
        ASSERT_TRUE(backend_->execute(request(), response));
        EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_SUCCESS);
        backend_->shutdown();
        backend_.reset();
        ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_.get()));
    }
}

TEST_F(CanonicalHostIOTest, ExplicitHostTimeoutRetainsStorePinsAfterShutdown) {
    initialize(/*gdr=*/false, /*retain_client=*/true);
    client_->save_status = kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend_->execute(request(/*host_write=*/true), response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_TIMEOUT);
    backend_->shutdown();
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_.get()));
    ::testing::Mock::AllowLeak(client_.get());
}

TEST_F(CanonicalHostIOTest, SingleRankHostBufferRejectionNeverPublishes) {
    initialize();
    const CacheKeysType keys{101};
    StorageRequest write;
    write.source_tier = Tier::HOST;
    write.keys = std::make_shared<const CacheKeysType>(keys);
    write.handles = {{{"default", host_block_}}};
    auto task = backend_->prepareWrite(std::move(write));
    ASSERT_TRUE(task);
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 1u);
    host_pool_->markUncertainRemoteIo();

    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "uncertain_host";
    allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations = {{{"v1_Fdefault_h0", "head_0"}, {"v1_Fdefault_h1", "head_1"}}};
    EXPECT_CALL(*client_, getWriteLocation(_, _, keys, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client_, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_, finishWrite(_, _, "uncertain_host", _, _))
        .WillOnce(Invoke([](const auto&, const auto&, const auto&,
                            const kv_cache_manager::BlockMask& mask, const kv_cache_manager::Locations& actual) {
            EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 0u);
            EXPECT_TRUE(actual.empty());
            return true;
        }));
    EXPECT_TRUE(backend_->write(std::move(task)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend_));
    EXPECT_EQ(host_pool_->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_TRUE(host_pool_->isAllocated(host_block_));
    EXPECT_TRUE(host_pool_->hasUncertainRemoteIo());
}

}  // namespace
}  // namespace rtp_llm
