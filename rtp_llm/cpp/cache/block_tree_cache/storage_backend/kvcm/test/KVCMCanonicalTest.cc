#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"

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
        environment.cache_config, config, runtime, parallelism, SpeculativeExecutionConfig{}, broadcast, client));
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
        backend->read(request, observed.match_meta, [promise](bool success) { promise->set_value(success); });
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
        EXPECT_CALL(*client, shutdown()).Times(1);
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
        environment.cache_config, config, runtime, parallelism, SpeculativeExecutionConfig{}, nullptr, client, fence));
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
    EXPECT_FALSE(result->success());
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

}  // namespace
}  // namespace rtp_llm
