#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MultiPoolTestUtils.h"

namespace rtp_llm {
namespace {

BackendEnvironment configEnvironment(const CacheConfig& config) {
    BackendEnvironment environment;
    environment.cache_config = config;
    initializeEnvironmentPools(environment, "multi_pool_config", 8);
    return environment;
}

std::string automaticConfig(const BackendEnvironment& environment) {
    std::string json;
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke([&](const auto& configs, auto, const auto&, const auto&) {
        json = autil::legacy::ToJsonString(configs);
        return true;
    }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    EXPECT_TRUE(initSingleRank(*backend.backend, environment));
    return json;
}

bool initializeConfig(const BackendEnvironment& environment, const std::string& json, bool expected) {
    KVCacheConfig options;
    options.kvcm_client_config = json;
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(expected ? 1 : 0).WillRepeatedly(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config, options, RuntimeConfig{},
        singleRankConfig(), SpeculativeExecutionConfig{}, nullptr, client));
    return initSingleRank(*backend.backend, environment);
}

TEST(KVCMMultiPoolConfigTest, RequestFinishHostReuseRequiresSingleRankCoordinator) {
    const auto config = test::makeSimpleMhaCacheConfig(1, 16, 8, DataType::TYPE_FP16, 1, 2);
    for (const int size : {1, 2}) {
        for (int rank = 0; rank < size; ++rank) {
            auto parallel = singleRankConfig();
            parallel.tp_size = size;
            parallel.tp_rank = rank;
            BackendHandle backend(std::make_unique<KVCMStorageBackend>(config, KVCacheConfig{}, RuntimeConfig{},
                parallel, SpeculativeExecutionConfig{}, nullptr));
            EXPECT_EQ(backend->canInitiateHostWrite(), size == 1 && rank == 0);
        }
    }
}

TEST(KVCMMultiPoolConfigTest, CustomConfigPreservesHeterogeneousByteSizesAndAcceptsReorderedSpecs) {
    auto environment = configEnvironment(test::makeHeterogeneousRemoteCacheConfig());
    kvcm::ClientWrapper::ConfigMap configs;
    autil::legacy::FromJsonString(configs, automaticConfig(environment));
    const auto& config = configs.at("");
    EXPECT_EQ(config->block_size(), 8);
    const auto& sizes = *config->location_spec_infos();
    EXPECT_EQ(sizes.at("tp0_Ffull0"), 128);
    EXPECT_EQ(sizes.at("tp0_Ffull1"), 448);
    EXPECT_NE(sizes.at("tp0_Ffull1"), sizes.at("tp0_Llinear0"));
    for (auto& [name, specs] : *config->location_spec_groups()) {
        std::reverse(specs.begin(), specs.end());
    }
    EXPECT_TRUE(initializeConfig(environment, autil::legacy::ToJsonString(configs), true));
}

TEST(KVCMMultiPoolConfigTest, CustomConfigRejectsTokenSpecAndGroupMismatchesBeforeRegistration) {
    auto environment = configEnvironment(test::makeHeterogeneousRemoteCacheConfig());
    const auto original = automaticConfig(environment);
    for (int mutation = 0; mutation < 6; ++mutation) {
        SCOPED_TRACE(mutation);
        autil::legacy::json::JsonMap configs;
        autil::legacy::FromJsonString(configs, original);
        auto config = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(configs.at(""));
        if (mutation == 0) {
            config["block_size"] = int32_t{16};
        } else if (mutation == 1 || mutation == 2) {
            kvcm::KVCMConfig::LocationSpecInfoMap sizes;
            autil::legacy::FromJson(sizes, config.at("location_spec_infos"));
            if (mutation == 1) {
                sizes["tp0_Ffull1"] = sizes.at("tp0_Ffull0");
            } else {
                sizes.erase("tp0_Ffull1");
            }
            config["location_spec_infos"] = autil::legacy::ToJson(sizes);
        } else {
            kvcm::KVCMConfig::LocationSpecGroups groups;
            autil::legacy::FromJson(groups, config.at("location_spec_groups"));
            if (mutation == 3) {
                groups.clear();
            } else if (mutation == 4) {
                groups["Ffull0Ffull1Llinear0"].pop_back();
            } else {
                groups["Ffull0Ffull1Llinear0"][0] = "tp1_Ffull0";
            }
            config["location_spec_groups"] = autil::legacy::ToJson(groups);
        }
        configs[""] = config;
        EXPECT_FALSE(initializeConfig(environment, autil::legacy::ToJsonString(configs), false));
    }
}

TEST(KVCMMultiPoolConfigTest, DifferentCacheKeyTokenStridesAreRejectedExplicitly) {
    auto config = test::makeHeterogeneousRemoteCacheConfig();
    auto groups = config.topology().groups();
    auto spec = groups[1].spec->clone();
    spec->cache_key_token_stride = 16;
    groups[1].spec = std::move(spec);
    config.setTopology(std::move(groups), config.topology().layers());
    auto environment = configEnvironment(config);
    EXPECT_FALSE(initializeConfig(environment, "", false));
}

TEST(KVCMMultiPoolConfigTest, SingleGroupCustomConfigCanOmitExplicitLocationGroups) {
    auto environment = makeBackendEnvironment("single_group_config");
    kvcm::ClientWrapper::ConfigMap configs;
    autil::legacy::FromJsonString(configs, automaticConfig(environment));
    configs.at("")->location_spec_groups()->clear();
    EXPECT_TRUE(initializeConfig(environment, autil::legacy::ToJsonString(configs), true));
}

std::string instanceId(const std::string& json) {
    autil::legacy::json::JsonMap configs;
    autil::legacy::FromJsonString(configs, json);
    const auto config = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(configs.at(""));
    return autil::legacy::AnyCast<std::string>(config.at("instance_id"));
}

TEST(KVCMMultiPoolConfigTest, EqualByteCountsWithDifferentLayoutsUseDifferentInstances) {
    auto environment = makeMultiGroupBackendEnvironment("layout_identity", 2, 1, 1);
    const auto first = instanceId(automaticConfig(environment));
    auto groups = environment.cache_config.topology().groups();
    const auto original_bytes = groups[0].kvBlockStrideBytes();
    groups[0].spec = test::makeMhaSpec("full0", 8, DataType::TYPE_FP16, 2, 1);
    ASSERT_EQ(groups[0].kvBlockStrideBytes(), original_bytes);
    environment.cache_config.setTopology(std::move(groups), environment.cache_config.topology().layers());
    EXPECT_NE(instanceId(automaticConfig(environment)), first);
}

TEST(KVCMMultiPoolConfigTest, LocalGroupOrderDoesNotChangeTheRemoteInstanceIdentity) {
    auto environment = configEnvironment(test::makeHeterogeneousRemoteCacheConfig());
    const auto first = instanceId(automaticConfig(environment));
    auto groups = environment.cache_config.topology().groups();
    std::reverse(groups.begin(), groups.end());
    environment.cache_config.setTopology(std::move(groups), environment.cache_config.topology().layers());
    EXPECT_EQ(instanceId(automaticConfig(environment)), first);
}

}  // namespace
}  // namespace rtp_llm
