#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <unistd.h>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CanonicalCacheLayout.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {

size_t hashString(const std::string& value) {
    return std::hash<std::string>{}(value);
}

std::string nextTraceId(const char* operation, std::atomic<uint64_t>& sequence) {
    return std::string("block_tree_") + operation + "_" + std::to_string(sequence.fetch_add(1));
}

struct KVCMMatchMeta final: StorageBackendMatchMeta {
    kv_cache_manager::Locations locations;
    bool                       filtered = false;
};

const StorageBlockHandle* findHandle(const std::vector<StorageBlockHandle>& handles, std::string_view tag) {
    const auto it = std::find_if(handles.begin(), handles.end(), [tag](const StorageBlockHandle& handle) {
        return handle.tag == tag && !isNullBlockIdx(handle.block);
    });
    return it == handles.end() ? nullptr : &*it;
}

}  // namespace

class KVCMStorageBackend::Impl {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;

    enum class TransferResult {
        SUCCESS,
        FAILED,
        TIMEOUT,
    };

    Impl(const CacheConfig&                            cache_config,
         const KVCacheConfig&                          kv_cache_config,
         const RuntimeConfig&                          runtime_config,
         const ParallelismConfig&                      parallelism_config,
         const SpeculativeExecutionConfig&             sp_config,
         std::shared_ptr<BroadcastManager>             broadcast_manager,
         bool                                          gdr_enabled,
         std::shared_ptr<kmonitor::MetricsReporter>    metrics_reporter,
         std::shared_ptr<kvcm::ClientWrapper>          client_wrapper,
         std::shared_ptr<WorkerCacheIOFence>            worker_fence):
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        gdr_enabled_(gdr_enabled),
        metrics_reporter_(std::move(metrics_reporter)),
        client_wrapper_(std::move(client_wrapper)),
        worker_fence_(parallelism_config.tp_rank == 0 ?
                          nullptr :
                          (worker_fence ? std::move(worker_fence) : std::make_shared<WorkerCacheIOFence>())),
        sdk_check_enabled_(autil::EnvUtil::getEnv("KVCM_SDK_CHECK", autil::EnvUtil::getEnv("RECO_SDK_CHECK", false))) {}

    bool init(const CacheTopology&                                                            topology,
              kvcm::GroupPolicy::SourceResolver                                               buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>&             pool_resolver,
              const std::function<const std::shared_ptr<HostBlockPool>&(const std::string&)>& host_pool_resolver) {
        RTP_LLM_LOG_INFO("start init BlockTree KVCM storage backend");
        if (topology.groups().empty() || parallelism_config_.tp_size <= 0 || parallelism_config_.tp_rank < 0
            || parallelism_config_.tp_rank >= parallelism_config_.tp_size) {
            RTP_LLM_LOG_ERROR("KVCM requires nonempty groups and a valid TP rank");
            return false;
        }
        buffer_resolver_    = std::move(buffer_resolver);
        pool_resolver_      = pool_resolver;
        host_pool_resolver_ = host_pool_resolver;
        if (kv_cache_config_.kvcm_remote_layout == "canonical_v1") {
            try {
                canonical_layout_ = std::make_unique<kvcm::CanonicalCacheLayout>(cache_config_, parallelism_config_);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("KVCM canonical layout init failed: %s", error.what());
                return false;
            }
        } else if (kv_cache_config_.kvcm_remote_layout != "legacy"
                   || (parallelism_config_.prefill_cp_config.kv_cache_sharded && parallelism_config_.tp_size > 1)) {
            RTP_LLM_LOG_ERROR("KVCM requires canonical_v1 for sharded CP, or a recognized remote layout");
            return false;
        }
        const auto key_tokens =
            canonical_layout_ ? cache_config_.seq_size_per_block : topology.groups().front().cacheKeyTokenStride();
        if (key_tokens == 0 || key_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            RTP_LLM_LOG_ERROR("KVCM cache-key token stride is out of range");
            return false;
        }
        cache_key_tokens_ = static_cast<int32_t>(key_tokens);
        if (parallelism_config_.tp_rank == 0 && parallelism_config_.tp_size > 1 && !broadcast_manager_) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM rank 0 requires a broadcast manager for tp_size=%ld",
                              parallelism_config_.tp_size);
            return false;
        }
        std::vector<std::string> full_group_tags;
        std::vector<std::string> other_group_tags;
        // CacheConfig::blockSizeBytesForGroup resolves MTP child-owned physical
        // strides; do not replace this with topology-derived geometry.
        std::unordered_map<std::string, size_t> group_block_size_bytes;
        const std::vector<GroupBase>&           groups = topology.groups();
        topology_ = &topology;
        group_block_size_bytes.reserve(groups.size());
        for (const auto& group : groups) {
            // Location specs describe bytes independently, but all components
            // of one remote key must cover the same token boundary.
            if (!canonical_layout_ && group.cacheKeyTokenStride() != key_tokens) {
                RTP_LLM_LOG_ERROR("KVCM requires one cache-key token stride: tag=%s stride=%zu expected=%zu",
                                  group.tag.c_str(), group.cacheKeyTokenStride(), key_tokens);
                return false;
            }
            has_swa_ = has_swa_ || group.policy.group_type == CacheGroupType::SWA;
            if (group.policy.group_type == CacheGroupType::FULL) {
                full_group_tags.push_back(group.tag);
            } else {
                other_group_tags.push_back(group.tag);
            }
            group_block_size_bytes.emplace(group.tag, cache_config_.blockSizeBytesForGroup(group.tag));
        }
        if (other_group_tags.empty()) {
            group_policy_ = std::make_unique<kvcm::FullLayerGroupPolicy>(
                topology, buffer_resolver_, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
        } else {
            group_policy_ = std::make_unique<kvcm::FullLinearLayerGroupPolicy>(topology,
                                                                               buffer_resolver_,
                                                                               full_group_tags,
                                                                               other_group_tags,
                                                                               std::max(1, cache_config_.linear_step),
                                                                               std::move(group_block_size_bytes));
        }
        if (!group_policy_->init()) {
            RTP_LLM_LOG_ERROR("BlockTree KVCM group policy init failed");
            return false;
        }
        kvcm::ClientWrapper::ConfigMap client_config_map;
        try {
            auto [location_infos, location_groups] = genLocationSpecs();
            if (!kv_cache_config_.kvcm_client_config.empty()) {
                autil::legacy::FromJsonString(client_config_map, kv_cache_config_.kvcm_client_config);
            } else {
                client_config_map = genClientConfig(location_infos, location_groups);
            }
            if (client_config_map.size() != 1 || client_config_map.count("") != 1 || !client_config_map.at("")) {
                RTP_LLM_LOG_ERROR("BlockTree KVCM requires one default instance config");
                return false;
            }
            const auto& config = client_config_map.at("");
            if (config->block_size() != cache_key_tokens_ || !config->location_spec_infos()
                || *config->location_spec_infos() != *location_infos) {
                RTP_LLM_LOG_ERROR("KVCM client token block size or location specs do not match the cache layout");
                return false;
            }
            auto actual_groups = config->location_spec_groups() ? *config->location_spec_groups() :
                                                                  kvcm::KVCMConfig::LocationSpecGroups{};
            auto expected_groups = *location_groups;
            for (auto& [name, specs] : actual_groups) {
                std::sort(specs.begin(), specs.end());
            }
            for (auto& [name, specs] : expected_groups) {
                std::sort(specs.begin(), specs.end());
            }
            // The legacy single-group protocol can omit explicit groups.
            const bool implicit_legacy_group = !canonical_layout_ && groups.size() == 1 && actual_groups.empty();
            if (actual_groups != expected_groups && !implicit_legacy_group) {
                RTP_LLM_LOG_ERROR("KVCM client location spec groups do not match the cache layout");
                return false;
            }
        } catch (const autil::legacy::ExceptionBase& error) {
            RTP_LLM_LOG_ERROR("parse KVCM_CLIENT_CONFIG failed: %s", error.what());
            return false;
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("initialize BlockTree KVCM client config failed: %s", error.what());
            return false;
        }
        const auto& config = client_config_map.at("");
        if (canonical_layout_ && !kv_cache_config_.kvcm_client_config.empty()) {
            const auto [expected_infos, expected_groups] = genLocationSpecs();
            const auto expected                          = genClientConfig(expected_infos, expected_groups).at("");
            if (!config->location_spec_infos() || !config->location_spec_groups()
                || *config->location_spec_infos() != *expected->location_spec_infos()
                || config->block_size() != expected->block_size()
                || config->instance_id().rfind("canonical_v1_", 0) != 0
                || autil::legacy::ToJsonString(config->model_deployment())
                       != autil::legacy::ToJsonString(expected->model_deployment())) {
                RTP_LLM_LOG_ERROR("KVCM_CLIENT_CONFIG disagrees with canonical_v1 storage topology/model identity");
                return false;
            }
        }
        // Use prefix queries for the instance; payload reuse is filtered locally.
        config->set_default_query_type(default_query_type_);
        if (kv_cache_config_.kvcm_min_replica_count < 0
            || (kv_cache_config_.kvcm_read_backend_type != 0
                && !isPayloadBackend(kv_cache_config_.kvcm_read_backend_type))) {
            RTP_LLM_LOG_ERROR("invalid KVCM replica/backend config");
            return false;
        }

        const auto registrations = makePoolRegistrations(pool_resolver);
        if (registrations.empty()) {
            return false;
        }
        const auto role =
            parallelism_config_.tp_rank == 0 ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::WORKER;
        if (!client_wrapper_) {
            client_wrapper_ = std::make_shared<kvcm::ClientWrapper>();
        }
        const bool initialized =
            client_wrapper_->initForPools(client_config_map, role, registrations, registration_tags_);
        if (!initialized) {
            client_wrapper_->shutdown();
            RTP_LLM_LOG_ERROR("create BlockTree KVCM clients failed");
            return false;
        }
        const auto tp_rank = parallelism_config_.tp_rank;
        RTP_LLM_LOG_INFO("BlockTree KVCM storage backend initialized, tp_rank=%ld tp_size=%ld policy={%s}",
                         tp_rank,
                         parallelism_config_.tp_size,
                         group_policy_->debugString().c_str());
        reportQuarantineMetrics(/*generation=*/0, /*lease_count=*/0, /*block_count=*/0);
        return true;
    }

    StorageMatchResult match(const StorageRequest& local_request) {
        const auto request = expandRequest(local_request);
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata match must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        // The allocator already caps this sequence at the final reusable full
        // block. Dropping another key here would exclude two tail blocks.
        const CacheKeysType& keys = *request.keys;
        if (request.local_matched_blocks_num >= keys.size()) {
            return {local_request.local_matched_blocks_num, nullptr};
        }
        const auto trace_id       = nextTraceId("match", match_trace_sequence_);
        auto query_type = static_cast<kv_cache_manager::QueryType>(default_query_type_);
        if (has_swa_) {
            // Fetch the FULL prefix as well as SWA locations, then apply each
            // group's model-defined reuse window locally.
            query_type = kv_cache_manager::QueryType::QT_BATCH_GET;
        }
        bool success = false;
        kv_cache_manager::Locations locations;
        bool positional = query_type == kv_cache_manager::QueryType::QT_BATCH_GET;
        if (kv_cache_config_.kvcm_read_backend_type != 0) {
            auto result = client_wrapper_->getCacheLocationsByBackend(
                "", trace_id, keys, {}, request.local_matched_blocks_num, {},
                static_cast<kv_cache_manager::StorageType>(kv_cache_config_.kvcm_read_backend_type));
            success = result.first;
            if (success) {
                locations = selectBackendLocations(result.second);
            }
            positional = true;
        } else {
            auto result = client_wrapper_->match(
                "", trace_id, query_type, keys, request.local_matched_blocks_num, {});
            success = result.first;
            locations = std::move(result.second);
        }
        if (!success) {
            return {local_request.local_matched_blocks_num, nullptr};
        }
        if (positional || canonical_layout_) {
            if (!positional) {
                RTP_LLM_CHECK_WITH_INFO(locations.size() <= keys.size() - request.local_matched_blocks_num,
                                        "KVCM prefix query shape mismatch");
                locations.insert(locations.begin(), request.local_matched_blocks_num, kv_cache_manager::Location{});
                locations.resize(keys.size());
            }
            auto meta = std::make_shared<KVCMMatchMeta>();
            meta->locations = reusableLocations(request, std::move(locations));
            meta->filtered = true;
            const size_t matched = request.local_matched_blocks_num + meta->locations.size();
            return {matched / blocksPerKey(), std::move(meta)};
        }
        kvcm::LocationsView locations_view;
        if (!group_policy_->filterNeedLoadLocations(locations, locations_view, /*block_mask=*/0)) {
            throw std::runtime_error("KVCM returned an invalid location shape");
        }
        if (locations_view.size() > keys.size() - request.local_matched_blocks_num) {
            throw std::runtime_error("KVCM prefix match exceeds the requested key range");
        }
        auto meta       = std::make_shared<KVCMMatchMeta>();
        meta->locations = std::move(locations);
        return {request.local_matched_blocks_num + locations_view.size(), std::move(meta)};
    }

    void read(const StorageRequest& local_request, const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
        const auto request = expandRequest(local_request);
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata read must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        const auto meta = std::dynamic_pointer_cast<KVCMMatchMeta>(match_meta);
        RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "KVCM read received invalid match metadata");
        kvcm::LocationsView locations_view;
        if (meta->filtered) {
            locations_view.resize(meta->locations.size());
            for (size_t i = 0; i < meta->locations.size(); ++i) {
                for (const auto& spec : meta->locations[i]) {
                    locations_view[i].emplace_back(spec);
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                group_policy_->filterNeedLoadLocations(meta->locations, locations_view, /*block_mask=*/0),
                "KVCM read location filtering failed");
        }
        const size_t remote_blocks = request.handles.size() - request.local_matched_blocks_num;
        RTP_LLM_CHECK_WITH_INFO(locations_view.size() == remote_blocks,
                                "KVCM read shape mismatch: locations=%zu remote_blocks=%zu",
                                locations_view.size(),
                                remote_blocks);

        std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
        const auto&                    spec_info = group_policy_->spec_info_map();
        const std::string              trace_id  = nextTraceId("read", read_trace_sequence_);
        initializeRequests(requests,
                           REMOTE_OPERATION_READ,
                           trace_id);
        for (size_t location_idx = 0; location_idx < locations_view.size(); ++location_idx) {
            const size_t key_idx = request.local_matched_blocks_num + location_idx;
            for (const auto& location_spec : locations_view[location_idx]) {
                const auto        info = spec_info.find(location_spec.spec_name);
                const std::string spec_name(location_spec.spec_name);
                RTP_LLM_CHECK_WITH_INFO(info != spec_info.end(), "KVCM read has unknown spec [%s]", spec_name.c_str());
                const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.tag);
                RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                        "KVCM read has no destination handle for key=%zu tag=%s",
                                        key_idx,
                                        info->second.tag.c_str());
                for (int rank : transferRanks(info->second, key_idx, false)) {
                    auto* remote = requests.at(static_cast<size_t>(rank)).mutable_remote_request();
                    appendTransfer(*remote, info->second, handle->block, std::string(location_spec.uri));
                    remote->add_coordinate_ids(static_cast<uint32_t>(key_idx));
                }
            }
        }
        (void)dispatchRequests(requests, kv_cache_config_.kvcm_get_broadcast_timeout);
    }

    void write(const StorageRequest& local_request) {
        const auto request = expandRequest(local_request);
        RTP_LLM_CHECK_WITH_INFO(parallelism_config_.tp_rank == 0,
                                "KVCM metadata write must run on tp rank 0, got %ld",
                                parallelism_config_.tp_rank);
        RTP_LLM_CHECK(request.keys != nullptr && request.keys->size() == request.handles.size());
        const size_t valid_keys_size = request.keys->size();
        if (valid_keys_size == 0) {
            return;
        }
        CacheKeysType            keys(request.keys->begin(), request.keys->begin() + valid_keys_size);
        std::vector<std::string> location_spec_group_names;
        RTP_LLM_CHECK_WITH_INFO(group_policy_->getNeedWriteGroups(request, valid_keys_size, location_spec_group_names),
                                "KVCM write group selection failed");
        const std::string trace_id     = nextTraceId("write", write_trace_sequence_);
        auto [success, write_location] = client_wrapper_->getWriteLocation(
            "", trace_id, keys, /*tokens=*/{}, location_spec_group_names, /*write_timeout_seconds=*/600,
            kv_cache_config_.kvcm_min_replica_count);
        RTP_LLM_CHECK_WITH_INFO(success, "KVCM StartWrite failed");
        static const kv_cache_manager::Locations empty_locations;
        bool                                     finish_attempted = false;
        try {
            std::vector<FunctionRequestPB> requests(static_cast<size_t>(parallelism_config_.tp_size));
            ActualUriGather                actual_uri_gather(requests.size());
            initializeRequests(requests, REMOTE_OPERATION_WRITE, trace_id);
            for (auto& remote_request : requests) {
                remote_request.mutable_remote_request()->set_host_source(request.source_tier == Tier::HOST);
            }
            const auto key_indices = unmaskedKeyIndices(write_location.block_mask, valid_keys_size);
            RTP_LLM_CHECK_WITH_INFO(key_indices.size() == write_location.locations.size(),
                                    "KVCM write mask/location mismatch: keys=%zu locations=%zu",
                                    key_indices.size(),
                                    write_location.locations.size());
            if (write_location.locations.empty()) {
                if (!write_location.write_session_id.empty()) {
                    // The server can create an empty, short-lived session when
                    // every key already has enough replicas. Close it normally.
                    finish_attempted = true;
                    try {
                        if (!client_wrapper_->finishWrite(
                            "", nextTraceId("finish_write", finish_write_trace_sequence_),
                            write_location.write_session_id, kv_cache_manager::BlockMaskOffset{0}, empty_locations)) {
                            RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                                write_location.write_session_id.c_str());
                        }
                    } catch (...) {
                        RTP_LLM_LOG_WARNING("KVCM failed to close empty write session [%s]",
                                            write_location.write_session_id.c_str());
                    }
                }
                return;
            }
            const auto& spec_info = group_policy_->spec_info_map();
            for (size_t location_idx = 0; location_idx < write_location.locations.size(); ++location_idx) {
                const size_t key_idx = key_indices[location_idx];
                if (canonical_layout_) {
                    std::vector<std::string> expected;
                    if (location_spec_group_names.empty()) {
                        for (const auto& [name, info] : spec_info) {
                            (void)info;
                            expected.push_back(name);
                        }
                    } else {
                        expected = location_spec_groups_.at(location_spec_group_names.at(key_idx));
                    }
                    std::vector<std::string> actual;
                    for (const auto& spec : write_location.locations[location_idx]) {
                        RTP_LLM_CHECK_WITH_INFO(!spec.uri.empty(), "KVCM StartWrite returned an empty shard URI");
                        actual.push_back(spec.spec_name);
                    }
                    std::sort(actual.begin(), actual.end());
                    std::sort(expected.begin(), expected.end());
                    RTP_LLM_CHECK_WITH_INFO(actual == expected, "KVCM StartWrite returned incomplete logical shards");
                } else {
                    RTP_LLM_CHECK_WITH_INFO(
                        group_policy_->validateWriteLocation(
                            write_location.locations[location_idx],
                            location_spec_group_names.empty() ? std::string{} : location_spec_group_names.at(key_idx)),
                        "KVCM StartWrite returned an incomplete or unexpected group at key=%zu",
                        key_idx);
                }
                for (auto& location_spec : write_location.locations[location_idx]) {
                    const auto info = spec_info.find(location_spec.spec_name);
                    RTP_LLM_CHECK_WITH_INFO(
                        info != spec_info.end(), "KVCM write has unknown spec [%s]", location_spec.spec_name.c_str());
                    const StorageBlockHandle* handle = findHandle(request.handles[key_idx], info->second.tag);
                    RTP_LLM_CHECK_WITH_INFO(handle != nullptr,
                                            "KVCM write has no source handle for key=%zu tag=%s",
                                            key_idx,
                                            info->second.tag.c_str());
                    const auto ranks = transferRanks(info->second, key_idx, true);
                    RTP_LLM_CHECK_WITH_INFO(ranks.size() == 1, "KVCM spec must have exactly one writer");
                    const size_t rank   = static_cast<size_t>(ranks.front());
                    auto*        remote = requests.at(rank).mutable_remote_request();
                    const bool host_source = request.source_tier == Tier::HOST;
                    RTP_LLM_CHECK_WITH_INFO(remote->block_ids_size() == 0 || remote->host_source() == host_source,
                                            "KVCM write cannot mix HOST and DEVICE blocks in one rank request");
                    remote->set_host_source(host_source);
                    appendTransfer(*remote, info->second, handle->block, location_spec.uri);
                    remote->add_coordinate_ids(static_cast<uint32_t>(key_idx));
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            for (auto& rank_request : requests) {
                auto* remote = rank_request.mutable_remote_request();
                if (remote->host_source()) {
                    for (const auto key : keys) {
                        remote->add_cache_keys(key);
                    }
                }
            }
            const auto responses = dispatchRequests(requests, kv_cache_config_.kvcm_put_broadcast_timeout);
            bool       has_actual_uri = false;
            for (size_t rank = 0; rank < responses.size(); ++rank) {
                const auto& actual_uris = responses[rank].remote_response().actual_uris();
                RTP_LLM_CHECK_WITH_INFO(actual_uris.empty()
                                           || static_cast<size_t>(actual_uris.size()) == actual_uri_gather[rank].size(),
                                        "KVCM write returned a partial actual URI vector for rank=%zu",
                                        rank);
                for (int uri_idx = 0; uri_idx < actual_uris.size(); ++uri_idx) {
                    if (!actual_uris[uri_idx].empty()) {
                        has_actual_uri                                             = true;
                        actual_uri_gather[rank][static_cast<size_t>(uri_idx)]->uri = actual_uris[uri_idx];
                    }
                }
            }
            const auto& actual_locations = has_actual_uri ? write_location.locations : empty_locations;
            finish_attempted             = true;
            RTP_LLM_CHECK_WITH_INFO(
                client_wrapper_->finishWrite("",
                                             nextTraceId("finish_write", finish_write_trace_sequence_),
                                             write_location.write_session_id,
                                             write_location.locations.size(),
                                             actual_locations),
                "KVCM FinishWrite failed");
        } catch (...) {
            if (!finish_attempted) {
                try {
                    if (!client_wrapper_->finishWrite("",
                                                      nextTraceId("abort_write", finish_write_trace_sequence_),
                                                      write_location.write_session_id,
                                                      /*block_mask=*/kv_cache_manager::BlockMaskOffset{0},
                                                      empty_locations)) {
                        RTP_LLM_LOG_WARNING("KVCM failed to abort write session [%s]",
                                            write_location.write_session_id.c_str());
                    }
                } catch (...) {
                    RTP_LLM_LOG_WARNING("KVCM abort write session threw, session=[%s]",
                                        write_location.write_session_id.c_str());
                }
            }
            throw;
        }
    }

    bool execute(const RemoteOperationRequestPB& request,
                 RemoteOperationResponsePB&      response) {
        const bool host_write = request.op() == REMOTE_OPERATION_WRITE && request.host_source();
        if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE) {
            return executeMetadata(request, response);
        }
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        if (tags.size() != blocks.size() || blocks.size() != uris.size()) {
            RTP_LLM_LOG_WARNING("KVCM transfer tag/block/URI count mismatch");
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (tags.empty()) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_SUCCESS);
            return true;
        }
        if (request.host_source() && request.op() != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("HOST buffers are only supported for remote writes");
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        if (canonical_layout_ ? (request.storage_layout() != "canonical_v1"
                                 || request.logical_shards_size() != request.group_tags_size()) :
                                ((!request.storage_layout().empty() && request.storage_layout() != "legacy")
                                 || request.logical_shards_size() != 0)) {
            RTP_LLM_LOG_WARNING("KVCM RPC storage layout disagrees with the executing worker");
            return false;
        }
        WorkerCacheIOFence::Lease worker_access;
        if (worker_fence_ && !tags.empty()) {
            if (host_write) {
                // Key-resolved HOST blocks carry local STORE refs. Retain I/O
                // serialization without recording them as TP0 mirror allocations.
                worker_access = worker_fence_->lockCompute();
            } else {
                if (request.block_generations_size() != request.block_ids_size()) {
                    RTP_LLM_LOG_WARNING("KVCM worker generation/block count mismatch: op=%d generations=%d blocks=%d",
                                        request.op(),
                                        request.block_generations_size(),
                                        request.block_ids_size());
                    response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
                    return false;
                }
                std::vector<CacheBlockGeneration> generations;
                for (size_t i = 0; i < tags.size(); ++i) {
                    generations.push_back({tags[i], blocks[i], request.block_generations(i), request.host_source()});
                }
                worker_access = worker_fence_->lockTransfer(generations);
            }
        }
        setCudaDevice();
        if (worker_access.owns_lock()) {
            worker_fence_->waitGpuCompletion();
        }
        kv_cache_manager::BlockBuffers buffers;
        if (canonical_layout_) {
            if (!genCanonicalBlockBuffers(request, tags, blocks, buffers,
                                          request.host_source() ? Tier::HOST : Tier::DEVICE)) {
                return false;
            }
        } else if (!group_policy_->genBlockBuffers(
                       tags, blocks, buffers, request.host_source() ? Tier::HOST : Tier::DEVICE)) {
            response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
            return false;
        }
        const auto result = executeTagTransfers(request.op(), tags, blocks, uris, buffers, response, request.host_source());
        response.set_transfer_status(result == TransferResult::SUCCESS ? REMOTE_TRANSFER_STATUS_SUCCESS :
                                     result == TransferResult::TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                         REMOTE_TRANSFER_STATUS_FAILED);
        return result == TransferResult::SUCCESS;
    }

    bool allocationOwner() const {
        return parallelism_config_.tp_rank == 0;
    }

    void shutdown() noexcept {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

    void reportQuarantineState(uint64_t generation, size_t task_count, size_t block_count) noexcept {
        reportQuarantineMetrics(generation, task_count, block_count);
    }

private:
    int resolveQueryType(int query_type) const {
        return query_type == 0 ? default_query_type_ : query_type;
    }

    bool genCanonicalBlockBuffers(const RemoteOperationRequestPB& request,
                                   const std::vector<std::string>& tags,
                                   const std::vector<int32_t>& blocks,
                                   kv_cache_manager::BlockBuffers& buffers,
                                   Tier source) const {
        RTP_LLM_CHECK(canonical_layout_ && tags.size() == blocks.size()
                      && request.logical_shards_size() == static_cast<int>(tags.size()));
        buffers.reserve(blocks.size());
        for (size_t index = 0; index < tags.size(); ++index) {
            if (source == Tier::HOST) {
                const auto& pool = host_pool_resolver_(tags[index]);
                if (!pool || pool->hasUncertainRemoteIo() || !pool->isAllocated(blocks[index])) {
                    return false;
                }
            }
            kv_cache_manager::BlockBuffer buffer;
            for (int layer : cache_config_.layerIdsForGroup(tags[index])) {
                for (const auto& part : canonical_layout_->slice(
                         layer, tags[index], request.logical_shards(index),
                         buffer_resolver_(layer, tags[index], blocks[index], source))) {
                    buffer.iovs.push_back({source == Tier::HOST ? kv_cache_manager::MemoryType::CPU :
                                                                 kv_cache_manager::MemoryType::GPU,
                                           part.addr, part.size_bytes, false});
                }
            }
            size_t bytes = 0;
            for (const auto& iov : buffer.iovs) {
                bytes += iov.size;
            }
            RTP_LLM_CHECK_WITH_INFO(bytes == canonical_layout_->shardBytes(tags[index]),
                                    "KVCM logical shard buffer size mismatch");
            buffers.push_back(std::move(buffer));
        }
        return true;
    }

    size_t blocksPerKey() const {
        return canonical_layout_ ? canonical_layout_->blocksPerKey() : 1;
    }

    StorageRequest expandRequest(const StorageRequest& request) const {
        const size_t scale = blocksPerKey();
        if (scale == 1 || request.keys_are_global) {
            return request;
        }
        RTP_LLM_CHECK_WITH_INFO(request.keys && request.remote_keys
                                    && request.keys->size() <= request.remote_keys->size() / scale,
                                "KVCM CP transfer requires the original fixed-block keys");
        const size_t   count = request.keys->size() * scale;
        StorageRequest expanded = request;
        expanded.keys =
            std::make_shared<const CacheKeysType>(request.remote_keys->begin(), request.remote_keys->begin() + count);
        expanded.handles.clear();
        expanded.handles.resize(count);
        expanded.local_matched_blocks_num = request.local_matched_blocks_num * scale;
        for (size_t local = 0; local < request.keys->size(); ++local) {
            RTP_LLM_CHECK_WITH_INFO((*request.keys)[local] == (*expanded.keys)[(local + 1) * scale - 1],
                                    "KVCM CP canonical key does not match its global endpoint");
            for (size_t offset = 0; offset < scale; ++offset) {
                auto& target = expanded.handles[local * scale + offset];
                for (const auto& handle : request.handles.at(local)) {
                    // Every rank stores the complete global LINEAR state at
                    // the round endpoint, while FULL pages have RR owners.
                    if (topology_->group(handle.tag).policy.group_type == CacheGroupType::FULL || offset + 1 == scale) {
                        target.push_back(handle);
                    }
                }
            }
        }
        return expanded;
    }

    std::vector<int> transferRanks(const kvcm::GroupPolicy::SpecInfo& info, size_t global_block, bool write) const {
        return canonical_layout_ ? canonical_layout_->ranks(info.tag, info.shard, global_block, write) :
                                   std::vector<int>{info.tp_rank};
    }

    void appendTransfer(RemoteOperationRequestPB&          remote,
                        const kvcm::GroupPolicy::SpecInfo& info,
                        BlockIdxType                       block,
                        const std::string&                 uri) const {
        remote.add_group_tags(info.tag);
        remote.add_block_ids(block);
        remote.add_uris(uri);
        remote.add_block_generations(remote.host_source() ?
                                         host_pool_resolver_(info.tag)->blockAllocationGeneration(block) :
                                         pool_resolver_(info.tag)->blockAllocationGeneration(block));
        if (canonical_layout_) {
            remote.add_logical_shards(info.shard);
        }
    }

    static bool isPayloadBackend(int type) {
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5;
    }

    kv_cache_manager::Locations selectBackendLocations(const kv_cache_manager::BackendLocations& result) const {
        kv_cache_manager::Locations locations;
        locations.reserve(result.size());
        for (const auto& key_locations : result) {
            RTP_LLM_CHECK_WITH_INFO(key_locations.size() <= 1, "KVCM returned multiple locations for one backend");
            locations.push_back(key_locations.empty() ? kv_cache_manager::Location{} :
                                                        key_locations.front().location_specs);
        }
        return locations;
    }

    kv_cache_manager::Locations reusableLocations(const StorageRequest& request,
                                                   kv_cache_manager::Locations locations) const {
        const size_t local = request.local_matched_blocks_num;
        const size_t key_count = request.keys->size();
        RTP_LLM_CHECK_WITH_INFO(locations.size() == key_count, "KVCM positional query shape mismatch");
        const auto& infos = group_policy_->spec_info_map();
        std::unordered_map<std::string, size_t> runs;
        size_t matched = local;
        for (size_t i = local; i < locations.size(); ++i) {
            std::unordered_map<std::string, const kv_cache_manager::LocationSpecUnit*> present;
            for (const auto& spec : locations[i]) {
                RTP_LLM_CHECK_WITH_INFO(infos.count(spec.spec_name) != 0, "KVCM returned an unknown spec");
                RTP_LLM_CHECK_WITH_INFO(present.emplace(spec.spec_name, &spec).second, "KVCM returned a duplicate spec");
            }
            bool complete = true;
            for (const auto& [id, group] : group_policy_->groups()) {
                (void)id;
                bool available = true;
                const int count     = canonical_layout_ ? canonical_layout_->shardCounts().at(group.tag) :
                                                          static_cast<int>(parallelism_config_.tp_size);
                for (int rank = 0; rank < count; ++rank) {
                    const auto name  = canonical_layout_ ? kvcm::genCanonicalSpecName(rank, group.group_name) :
                                                           kvcm::genLocationSpecName(rank, group.group_name);
                    const auto found = present.find(name);
                    available = available && found != present.end() && !found->second->uri.empty();
                }
                auto& run = runs[group.tag];
                run = available ? run + 1 : 0;
                const size_t required = std::min(i + 1 - local, topology_->group(group.tag).reuseBlockCount(i + 1));
                complete = complete && run >= required;
            }
            if (complete && (i + 1) % blocksPerKey() == 0) {
                matched = i + 1;
            }
        }
        locations.resize(matched);
        for (size_t i = local; i < matched; ++i) {
            auto& location = locations[i];
            location.erase(std::remove_if(location.begin(), location.end(), [&](const auto& spec) {
                const auto& group = topology_->group(infos.at(spec.spec_name).tag);
                return matched - i > group.reuseBlockCount(matched) || spec.uri.empty();
            }), location.end());
        }
        locations.erase(locations.begin(), locations.begin() + local);
        return locations;
    }

    bool executeMetadata(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
        if (parallelism_config_.tp_rank != 0) {
            RTP_LLM_LOG_WARNING("KVCM metadata operations require the TP rank 0 endpoint, got rank=%ld",
                                parallelism_config_.tp_rank);
            return false;
        }
        if (!request.has_metadata()) {
            return false;
        }
        const auto& query = request.metadata();
        const int type = resolveQueryType(query.query_type());
        if (type < 1 || type > 2 || query.p2p_host_count() < 0) {
            return false;
        }
        const std::vector<int64_t> keys(query.block_keys().begin(), query.block_keys().end());
        const std::vector<int64_t> tokens(query.token_ids().begin(), query.token_ids().end());
        const std::vector<std::string> names(query.location_spec_names().begin(), query.location_spec_names().end());
        kv_cache_manager::BlockMask mask = kv_cache_manager::BlockMaskOffset{0};
        if (query.block_mask().has_bool_masks()) {
            mask = kv_cache_manager::BlockMaskVector(query.block_mask().bool_masks().values().begin(),
                                                      query.block_mask().bool_masks().values().end());
        } else if (query.block_mask().info_case() == RemoteBlockMaskPB::kOffset) {
            if (query.block_mask().offset() < 0) {
                return false;
            }
            mask = static_cast<size_t>(query.block_mask().offset());
        }
        const auto query_type = static_cast<kv_cache_manager::QueryType>(type);
        auto appendLocation = [](const kv_cache_manager::Location& location, RemoteCacheLocationPB* output) {
            for (const auto& spec : location) {
                auto* item = output->add_specs();
                item->set_name(spec.spec_name);
                item->set_uri(spec.uri);
            }
        };
        switch (request.op()) {
            case REMOTE_OPERATION_REMOVE_CACHE:
                return client_wrapper_->removeCache("", request.trace_id(), keys, tokens, mask);
            case REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND: {
                if (query.query_type() != 0 && query.query_type() != 1) {
                    return false;
                }
                const int backend = query.backend_type() == 0 ? kv_cache_config_.kvcm_read_backend_type :
                                                                query.backend_type();
                if (!isPayloadBackend(backend)) {
                    return false;
                }
                auto [success, locations] = client_wrapper_->getCacheLocationsByBackend(
                    "", request.trace_id(), keys, tokens, mask, names, static_cast<kv_cache_manager::StorageType>(backend));
                if (success) {
                    for (const auto& key_locations : locations) {
                        auto* output = response.add_backend_locations();
                        for (const auto& location : key_locations) {
                            auto* item = output->add_locations();
                            item->set_backend_type(static_cast<int32_t>(location.type));
                            item->set_spec_size(location.spec_size);
                            appendLocation(location.location_specs, item);
                        }
                    }
                }
                return success;
            }
            case REMOTE_OPERATION_GET_HOST_CACHE_STATE: {
                if (type != 2) {
                    RTP_LLM_LOG_WARNING("KVCM host-state queries require prefix mode, got %d", type);
                    return false;
                }
                const std::vector<std::string> medium(query.medium().begin(), query.medium().end());
                auto [success, hosts] = client_wrapper_->getHostCacheState(
                    "", request.trace_id(), query_type, keys, medium, query.p2p_host_count());
                if (success) {
                    for (const auto& host : hosts) {
                        auto* output = response.add_hosts();
                        output->set_host_ip_port(host.host_ip_port);
                        output->set_local(host.local);
                        output->set_p2p_1_fetch(host.p2p_1_fetch);
                        output->set_p2p_1_total_match(host.p2p_1_total_match);
                    }
                }
                return success;
            }
            default:
                return false;
        }
    }

    std::pair<std::shared_ptr<kvcm::KVCMConfig::LocationSpecInfoMap>,
              std::shared_ptr<kvcm::KVCMConfig::LocationSpecGroups>>
    genLocationSpecs() {
        auto infos  = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
        auto groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
        RTP_LLM_CHECK_WITH_INFO(
            group_policy_->buildLocationSpecGroups(static_cast<int>(parallelism_config_.tp_size),
                                                   *groups,
                                                   canonical_layout_ ? &canonical_layout_->shardCounts() : nullptr),
            "failed to build KVCM location spec groups");
        for (const auto& [group_id, group] : group_policy_->groups()) {
            const int count = canonical_layout_ ? canonical_layout_->shardCounts().at(group.tag) :
                                                  static_cast<int>(parallelism_config_.tp_size);
            for (int rank = 0; rank < count; ++rank) {
                const std::string spec_name = canonical_layout_ ? kvcm::genCanonicalSpecName(rank, group.group_name) :
                                                                  kvcm::genLocationSpecName(rank, group.group_name);
                infos->emplace(spec_name,
                               canonical_layout_ ? canonical_layout_->shardBytes(group.tag) :
                                                   cache_config_.blockSizeBytesForGroup(group.tag));
            }
        }
        location_spec_groups_ = *groups;
        return {std::move(infos), std::move(groups)};
    }

    kvcm::ClientWrapper::ConfigMap
    genClientConfig(const std::shared_ptr<kvcm::KVCMConfig::LocationSpecInfoMap>& location_infos,
                    const std::shared_ptr<kvcm::KVCMConfig::LocationSpecGroups>&  location_groups) {
        std::vector<std::string> addresses;
        if (!kv_cache_config_.kvcm_server_address.empty()) {
            addresses.push_back(kv_cache_config_.kvcm_server_address);
        }
        auto channel = std::make_shared<kvcm::MetaChannelConfig>(kv_cache_config_.kvcm_meta_channel_retry_time,
                                                                 kv_cache_config_.kvcm_meta_channel_connection_timeout,
                                                                 kv_cache_config_.kvcm_meta_channel_call_timeout);
        auto sdk     = std::make_shared<kvcm::SdkWrapperConfig>(kv_cache_config_.kvcm_storage_thread_num,
                                                            kv_cache_config_.kvcm_storage_queue_size,
                                                            kv_cache_config_.kvcm_put_timeout_ms,
                                                            kv_cache_config_.kvcm_get_timeout_ms);
        sdk->parseBackendConfigs(kv_cache_config_.kvcm_model_sdk_config);

        const std::string model_name = runtime_config_.model_name;
        const std::string dtype      = getDataTypeStr(cache_config_.dtype);
        std::string       extra      = kv_cache_config_.kvcm_model_extra_info;
        extra += '/' + autil::EnvUtil::getEnv("BIZ_NAME", std::string("")) + '/'
                 + std::to_string(hashString(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""))));
        if (canonical_layout_) {
            extra += '/' + canonical_layout_->fingerprint();
        }
        std::string draft_info;
        if (!cache_config_.mtp_sub_configs.empty()) {
            draft_info = '{' + sp_config_.to_string() + '}';
        }
        std::stringstream identity;
        identity << "instance_group: " << kv_cache_config_.kvcm_instance_group << ";block_size:" << cache_key_tokens_
                 << ";model_name:" << model_name << ";dtype_str:" << dtype << ";use_mla:" << cache_config_.use_mla
                 << ";fp8_kv_cache:" << kv_cache_config_.fp8_kv_cache
                 << ";tp_size:" << (canonical_layout_ ? 1 : parallelism_config_.tp_size)
                 << ";dp_size:" << (canonical_layout_ ? 1 : parallelism_config_.dp_size) << ";extra_info:" << extra
                 << ";location_spec_info:" << autil::legacy::ToJsonString(location_infos, true)
                 << ";location_spec_groups:" << autil::legacy::ToJsonString(location_groups, true)
                 << ";default_query_type:" << default_query_type_
                 << ";draft_model_info:" << draft_info;
        if (!canonical_layout_ && topology_->groups().size() > 1) {
            auto layout_tags = topology_->groupTags();
            std::sort(layout_tags.begin(), layout_tags.end());
            for (const auto& tag : layout_tags) {
                identity << ";layout:" << tag << '{';
                for (const int layer : cache_config_.layerIdsForGroup(tag)) {
                    identity << layer << ':' << cache_config_.physicalGroupForLayer(layer, tag).spec->fingerprint() << ';';
                }
                identity << '}';
            }
        }
        std::string instance_id = kv_cache_config_.kvcm_instance_id_salt;
        if (!instance_id.empty()) {
            instance_id += '_';
        }
        instance_id += std::to_string(hashString(identity.str()));
        if (canonical_layout_) {
            instance_id = "canonical_v1_" + instance_id;
        }

        auto config =
            std::make_shared<kvcm::KVCMConfig>(kv_cache_config_.kvcm_enable_vipserver,
                                               kv_cache_config_.kvcm_vipserver_domain,
                                               cache_key_tokens_,
                                               kv_cache_config_.kvcm_instance_group,
                                               instance_id,
                                               addresses,
                                               location_infos,
                                               channel,
                                               sdk,
                                               location_groups,
                                               kvcm::ModelDeployment(model_name,
                                                                     dtype,
                                                                     cache_config_.use_mla,
                                                                     canonical_layout_ ? 1 : static_cast<int32_t>(parallelism_config_.tp_size),
                                                                     canonical_layout_ ? 1 : static_cast<int32_t>(parallelism_config_.dp_size),
                                                                     1,
                                                                     extra,
                                                                     kv_cache_config_.kvcm_model_user_data));
        return {{"", std::move(config)}};
    }

    std::vector<kvcm::ClientWrapper::PoolRegistration>
    makePoolRegistrations(const std::function<const DeviceBlockPoolPtr&(const std::string&)>& pool_resolver) {
        const auto&          groups = group_policy_->groups();
        std::vector<int32_t> ordered_groups;
        for (const auto& [id, group] : groups) {
            ordered_groups.push_back(id);
        }
        // Preserve the existing primary registration identity independently of map order.
        std::sort(ordered_groups.begin(), ordered_groups.end(), [&](int32_t left, int32_t right) {
            const auto& lhs = groups.at(left);
            const auto& rhs = groups.at(right);
            return std::make_pair(!lhs.is_full, lhs.group_name) < std::make_pair(!rhs.is_full, rhs.group_name);
        });
        registration_tags_.clear();
        std::vector<kvcm::ClientWrapper::PoolRegistration> registrations;
        std::vector<std::string>                             registration_tags;
        registrations.reserve(ordered_groups.size());
        registration_tags.reserve(ordered_groups.size());
        for (int32_t group_id : ordered_groups) {
            const auto& tag  = groups.at(group_id).tag;
            const auto& pool = pool_resolver(tag);
            if (!pool->getBaseAddress() || pool->getTotalSizeBytes() == 0) {
                RTP_LLM_LOG_ERROR("KVCM group %d has no valid registration span", group_id);
                return {};
            }
            std::string registration_spec;
            if (canonical_layout_) {
                for (int shard = 0; shard < canonical_layout_->shardCounts().at(tag); ++shard) {
                    const auto ranks = canonical_layout_->ranks(tag, shard, parallelism_config_.tp_rank, false);
                    if (std::find(ranks.begin(), ranks.end(), parallelism_config_.tp_rank) != ranks.end()) {
                        registration_spec = kvcm::genCanonicalSpecName(shard, groups.at(group_id).group_name);
                        break;
                    }
                }
            } else {
                registration_spec = kvcm::genLocationSpecName(static_cast<int>(parallelism_config_.tp_rank),
                                                              groups.at(group_id).group_name);
            }
            const size_t registration_size =
                gdr_enabled_ ? pool->getAllocationSizeBytes() : pool->getTotalSizeBytes();
            kvcm::ClientWrapper::PoolRegistration registration{
                {pool->getBaseAddress(), registration_size}, registration_spec};
            if (gdr_enabled_) {
#if USING_CUDA
                if (pool->where() != MemoryType::MEMORY_GPU || !pool->usesDedicatedDeviceAllocation()
                    || pool->getAllocationSizeBytes() == 0 || pool->deviceIndex() < 0) {
                    RTP_LLM_LOG_ERROR(
                        "RemoteCache GDR requires a dedicated cudaMalloc backing for group %d", group_id);
                    return {};
                }
                const long page_size = sysconf(_SC_PAGESIZE);
                if (page_size <= 0
                    || reinterpret_cast<uintptr_t>(pool->getBaseAddress()) % static_cast<size_t>(page_size) != 0
                    || pool->getAllocationSizeBytes() % static_cast<size_t>(page_size) != 0) {
                    RTP_LLM_LOG_ERROR("RemoteCache GDR registration span must be page aligned for group %d: "
                                      "base=%p size=%zu page_size=%ld",
                                      group_id,
                                      pool->getBaseAddress(),
                                      pool->getAllocationSizeBytes(),
                                      page_size);
                    return {};
                }
                kv_cache_manager::ClientMemoryRegistrations memory_registrations;
                memory_registrations.gpu.push_back(
                    {pool->getBaseAddress(), pool->getAllocationSizeBytes(), pool->deviceIndex()});
                registration.memory_registrations = std::move(memory_registrations);
#else
                RTP_LLM_LOG_ERROR("RemoteCache GDR requires a CUDA build");
                return {};
#endif
            }
            registration_tags.push_back(tag);
            registrations.push_back(std::move(registration));
        }
        registration_tags_ = std::move(registration_tags);
        return registrations;
    }

    TransferResult executeTagTransfers(RemoteOpType                       operation,
                             const std::vector<std::string>&    tags,
                             const std::vector<int32_t>&        blocks,
                             const kv_cache_manager::UriStrVec& uris,
                             kv_cache_manager::BlockBuffers&    buffers,
                             RemoteOperationResponsePB&         response,
                             bool                               cpu_source = false) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return TransferResult::FAILED;
        }
        std::unordered_map<std::string, std::vector<size_t>> indices_by_tag;
        for (size_t index = 0; index < tags.size(); ++index) {
            indices_by_tag[tags[index]].push_back(index);
        }
        // Reject an invalid route before issuing any SDK operation. Otherwise
        // the registered subsets could complete before a later unknown tag
        // turns the whole request into a failure.
        for (const auto& [tag, indices] : indices_by_tag) {
            (void)indices;
            if (std::find(registration_tags_.begin(), registration_tags_.end(), tag) == registration_tags_.end()) {
                RTP_LLM_LOG_ERROR("KVCM transfer contains unregistered tag [%s]", tag.c_str());
                return TransferResult::FAILED;
            }
        }
        // Keep replacements local until every tag completes. A timeout or
        // failure is an all-or-nothing response and leaves response.actual_uris empty.
        auto actual_uris = uris;
        for (const auto& tag : registration_tags_) {
            const auto found = indices_by_tag.find(tag);
            if (found == indices_by_tag.end()) {
                continue;
            }
            const auto&                    indices = found->second;
            kv_cache_manager::UriStrVec    batch_uris;
            kv_cache_manager::BlockBuffers batch_buffers;
            std::vector<int32_t>           batch_blocks;
            for (size_t index : indices) {
                batch_uris.push_back(uris[index]);
                batch_buffers.push_back(std::move(buffers[index]));
                batch_blocks.push_back(blocks[index]);
            }
            auto trace_info = makeTransferTraceInfo(batch_blocks);
            if (cpu_source) {
                if (!trace_info) {
                    trace_info = std::make_shared<kv_cache_manager::TransferTraceInfo>();
                }
                trace_info->need_print = false;
            }
            if (operation == REMOTE_OPERATION_READ) {
                const auto ec =
                    client_wrapper_->loadKvCachesForTagWithStatus(tag, batch_uris, batch_buffers, trace_info);
                if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                    return TransferResult::TIMEOUT;
                }
                if (ec != kv_cache_manager::ClientErrorCode::ER_OK) {
                    return TransferResult::FAILED;
                }
            } else {
                auto [ec, result] =
                    client_wrapper_->saveKvCachesForTagWithStatus(tag, batch_uris, batch_buffers, trace_info);
                if (ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT) {
                    return TransferResult::TIMEOUT;
                }
                if (ec != kv_cache_manager::ClientErrorCode::ER_OK
                    || (!result.empty() && result.size() != batch_uris.size())) {
                    return TransferResult::FAILED;
                }
                for (size_t index = 0; index < result.size(); ++index) {
                    actual_uris[indices[index]] = std::move(result[index]);
                }
            }
            indices_by_tag.erase(found);
        }
        RTP_LLM_CHECK(indices_by_tag.empty());
        if (operation == REMOTE_OPERATION_WRITE && actual_uris != uris) {
            for (auto& uri : actual_uris) {
                *response.add_actual_uris() = std::move(uri);
            }
        }
        return TransferResult::SUCCESS;
    }

    void reportQuarantineMetrics(uint64_t generation, size_t lease_count, size_t block_count) const noexcept {
        if (!metrics_reporter_) {
            return;
        }
        // Reporting may call into external metric backends. Serialize it
        // independently from quarantine state updates, and discard snapshots
        // superseded by a later state transition.
        std::lock_guard<std::mutex> lock(quarantine_metrics_mutex_);
        if (quarantine_metrics_reported_ && generation <= last_quarantine_metrics_generation_) {
            return;
        }
        try {
            RtpLLMRemoteCacheQuarantineMetricsCollector collector;
            collector.quarantined_lease_count = static_cast<int64_t>(lease_count);
            collector.quarantined_block_count = static_cast<int64_t>(block_count);
            metrics_reporter_->report<RtpLLMRemoteCacheQuarantineMetrics,
                                      RtpLLMRemoteCacheQuarantineMetricsCollector>(nullptr, &collector);
            quarantine_metrics_reported_        = true;
            last_quarantine_metrics_generation_ = generation;
        } catch (...) {
            // Keep the same generation eligible for a later retry.
            quarantine_metrics_reported_ = false;
            RTP_LLM_LOG_ERROR("failed to report KVCM quarantine metrics");
        }
    }

    void initializeRequests(std::vector<FunctionRequestPB>& requests,
                            RemoteOpType                    operation,
                            const std::string&              trace_id) const {
        for (auto& request : requests) {
            request.mutable_remote_request()->set_op(operation);
            request.mutable_remote_request()->set_trace_id(trace_id);
            request.mutable_remote_request()->set_storage_layout(kv_cache_config_.kvcm_remote_layout);
        }
    }

    std::vector<size_t> unmaskedKeyIndices(const kv_cache_manager::BlockMask& mask, size_t key_count) const {
        std::vector<size_t> result;
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, kv_cache_manager::BlockMaskOffset>) {
                    RTP_LLM_CHECK_WITH_INFO(value <= key_count, "KVCM write offset exceeds key count");
                    for (size_t key = static_cast<size_t>(value); key < key_count; ++key) {
                        result.push_back(key);
                    }
                } else {
                    RTP_LLM_CHECK_WITH_INFO(value.size() == key_count, "KVCM write mask size mismatch");
                    for (size_t key = 0; key < value.size(); ++key) {
                        if (!value[key]) {
                            result.push_back(key);
                        }
                    }
                }
            },
            mask);
        return result;
    }

    std::vector<FunctionResponsePB> dispatchRequests(const std::vector<FunctionRequestPB>& requests,
                                                      int timeout_ms) {
        std::vector<FunctionResponsePB> responses;
        if (!broadcast_manager_
            || (requests.size() == 1 && requests.front().remote_request().host_source())) {
            RTP_LLM_CHECK_WITH_INFO(
                requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
            FunctionResponsePB response;
            execute(requests.front().remote_request(), *response.mutable_remote_response());
            responses.push_back(std::move(response));
        } else {
            auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                               const std::shared_ptr<grpc::ClientContext>& context,
                               const FunctionRequestPB&                    request,
                               grpc::CompletionQueue*                      queue) {
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            };
            auto result =
                broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, timeout_ms, rpc_call);
            if (result == nullptr) {
                // No RPC has been submitted when broadcast returns nullptr.
                throw std::runtime_error("KVCM broadcast dispatch failed");
            }
            try {
                result->waitDone();
            } catch (const std::exception& exception) {
                if (result->deadlineExceeded()) {
                    throw StorageOperationTimeout(std::string("KVCM broadcast timed out: ") + exception.what(), false);
                }
                throw StorageOperationCompletionUnknown(
                    std::string("KVCM broadcast completion is unknown: ") + exception.what(), false);
            } catch (...) {
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown", false);
            }
            if (!result->success()) {
                throw StorageOperationCompletionUnknown("KVCM broadcast completion is unknown", false);
            }
            responses = result->responses();
        }

        if (responses.size() != requests.size()) {
            throw StorageOperationCompletionUnknown("KVCM worker response count mismatch", false);
        }
        bool failed = false;
        for (const auto& response : responses) {
            if (!response.has_remote_response()) {
                throw StorageOperationCompletionUnknown("KVCM worker returned no transfer response", false);
            }
            switch (response.remote_response().transfer_status()) {
                case REMOTE_TRANSFER_STATUS_UNSPECIFIED:  // successful legacy worker
                case REMOTE_TRANSFER_STATUS_SUCCESS:
                    break;
                case REMOTE_TRANSFER_STATUS_FAILED:
                    failed = true;
                    break;
                case REMOTE_TRANSFER_STATUS_TIMEOUT:
                    throw StorageOperationTimeout("KVCM transfer timed out", false);
                case REMOTE_TRANSFER_STATUS_COMPLETION_UNKNOWN:
                    throw StorageOperationCompletionUnknown("KVCM transfer completion is unknown", false);
                default:
                    throw StorageOperationCompletionUnknown("KVCM transfer completion is unknown", false);
            }
        }
        if (failed) {
            throw std::runtime_error("KVCM transfer failed");
        }
        return responses;
    }

    void setCudaDevice() const {
        const int expected = static_cast<int>(parallelism_config_.local_rank);
        int       current  = -1;
        check_cuda_value(cudaGetDevice(&current));
        if (current != expected) {
            check_cuda_value(cudaSetDevice(expected));
        }
    }

    std::shared_ptr<kv_cache_manager::TransferTraceInfo>
    makeTransferTraceInfo(const std::vector<int32_t>& block_ids) const {
        if (!sdk_check_enabled_) {
            return nullptr;
        }
        auto trace_info        = std::make_shared<kv_cache_manager::TransferTraceInfo>();
        trace_info->need_print = true;
        trace_info->block_ids.reserve(block_ids.size());
        for (const auto block_id : block_ids) {
            trace_info->block_ids.push_back(std::to_string(block_id));
        }
        return trace_info;
    }

private:
    CacheConfig                          cache_config_;
    KVCacheConfig                        kv_cache_config_;
    RuntimeConfig                        runtime_config_;
    ParallelismConfig                    parallelism_config_;
    SpeculativeExecutionConfig           sp_config_;
    std::shared_ptr<BroadcastManager>                    broadcast_manager_;
    const bool                                           gdr_enabled_;
    std::unique_ptr<kvcm::GroupPolicy>                   group_policy_;
    std::unique_ptr<kvcm::CanonicalCacheLayout>          canonical_layout_;
    std::shared_ptr<kmonitor::MetricsReporter>           metrics_reporter_;
    std::shared_ptr<kvcm::ClientWrapper>                 client_wrapper_;
    std::vector<std::string>                             registration_tags_;
    mutable std::mutex                                   quarantine_metrics_mutex_;
    mutable bool                                         quarantine_metrics_reported_{false};
    mutable uint64_t                                     last_quarantine_metrics_generation_{0};
    kvcm::GroupPolicy::SourceResolver buffer_resolver_;
    kvcm::GroupPolicy::LocationSpecGroups location_spec_groups_;
    std::shared_ptr<WorkerCacheIOFence> worker_fence_;
    std::function<const DeviceBlockPoolPtr&(const std::string&)> pool_resolver_;
    std::function<const std::shared_ptr<HostBlockPool>&(const std::string&)> host_pool_resolver_;
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    const bool            sdk_check_enabled_;
    const CacheTopology*  topology_ = nullptr;
    bool                  has_swa_ = false;
    const int32_t         default_query_type_ = static_cast<int32_t>(kv_cache_manager::QueryType::QT_PREFIX_MATCH);
    int32_t cache_key_tokens_ = 0;
};

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                            cache_config,
                                       const KVCacheConfig&                          kv_cache_config,
                                       const RuntimeConfig&                          runtime_config,
                                       const ParallelismConfig&                      parallelism_config,
                                       const SpeculativeExecutionConfig&             sp_config,
                                       std::shared_ptr<BroadcastManager>             broadcast_manager,
                                       bool                                          gdr_enabled,
                                       std::shared_ptr<kmonitor::MetricsReporter>    metrics_reporter,
                                       std::shared_ptr<kvcm::ClientWrapper>          client_wrapper,
                                       std::shared_ptr<WorkerCacheIOFence>            worker_fence):
    StorageBackend(makeStorageBackendExecutor(kv_cache_config.kvcm_asyncwrapper_thread_num,
                                              kv_cache_config.kvcm_asyncwrapper_queue_size)),
    impl_(std::make_unique<Impl>(cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 gdr_enabled,
                                 std::move(metrics_reporter),
                                 std::move(client_wrapper),
                                 std::move(worker_fence))) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

bool KVCMStorageBackend::initImpl() {
    return impl_->init(
        topology(),
        [this](int layer_id, const std::string& tag, int block_id, Tier source) {
            RTP_LLM_CHECK(source == Tier::DEVICE || source == Tier::HOST);
            const auto resolver = bufferResolver(source);
            RTP_LLM_CHECK(resolver != nullptr);
            return resolver(layer_id, tag, block_id);
        },
        [this](const std::string& tag) -> const DeviceBlockPoolPtr& { return devicePool(tag); },
        [this](const std::string& tag) -> const std::shared_ptr<HostBlockPool>& { return hostPool(tag); });
}

StorageMatchResult KVCMStorageBackend::matchImpl(const StorageRequest& request) {
    return impl_->match(request);
}

void KVCMStorageBackend::readImpl(const StorageRequest&                           request,
                                  const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
    impl_->read(request, match_meta);
}

void KVCMStorageBackend::writeImpl(const StorageRequest& request) {
    impl_->write(request);
}

void KVCMStorageBackend::shutdownImpl() noexcept {
    impl_->shutdown();
}

void KVCMStorageBackend::onQuarantineChanged(uint64_t generation, size_t task_count, size_t block_count) noexcept {
    impl_->reportQuarantineState(generation, task_count, block_count);
}

bool KVCMStorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    response.Clear();
    const bool payload = request.op() == REMOTE_OPERATION_READ || request.op() == REMOTE_OPERATION_WRITE;
    if (payload) {
        response.set_transfer_status(REMOTE_TRANSFER_STATUS_FAILED);
    }
    try {
        RemoteOperationRequestPB local_request = request;
        StorageWriteTask host_pins;
        StorageRequest transfer;
        const bool host_write = request.op() == REMOTE_OPERATION_WRITE && request.host_source();
        if (host_write) {
            if (request.group_tags_size() != request.block_ids_size()
                || request.group_tags_size() != request.coordinate_ids_size()
                || request.block_ids_size() != request.uris_size()) {
                return false;
            }
            if (request.group_tags_size() != 0) {
                if (!hostWriteResolver() || request.cache_keys_size() == 0) {
                    return false;
                }
                const CacheKeysType keys(request.cache_keys().begin(), request.cache_keys().end());
                const std::vector<std::string> tags(request.group_tags().begin(), request.group_tags().end());
                const std::vector<uint32_t> coordinates(request.coordinate_ids().begin(), request.coordinate_ids().end());
                auto resolved = hostWriteResolver()(keys, tags, coordinates, /*timeout_ms=*/1000);
                if (!resolved.pins || resolved.local_blocks.size() != tags.size()) {
                    return false;
                }
                host_pins = std::move(resolved.pins);
                local_request.clear_block_ids();
                for (const auto block : resolved.local_blocks) {
                    local_request.add_block_ids(block);
                }
            }
            transfer.keys = std::make_shared<CacheKeysType>();
        } else if (payload) {
            if (request.group_tags_size() != request.block_ids_size() || request.block_ids_size() != request.uris_size()) {
                return false;
            }
            transfer.source_tier = request.host_source() ? Tier::HOST : Tier::DEVICE;
            transfer.keys = std::make_shared<CacheKeysType>(request.block_ids_size(), 0);
            transfer.handles.resize(request.block_ids_size());
            for (int i = 0; i < request.block_ids_size(); ++i) {
                transfer.handles[i].push_back({request.group_tags(i), request.block_ids(i)});
            }
        }
        // Keep allocator references alive until the synchronous SDK call returns.
        const auto invoke = [&](StorageWriteTask) {
            const bool success = impl_->execute(local_request, response);
            if (response.transfer_status() == REMOTE_TRANSFER_STATUS_TIMEOUT) {
                host_pins.quarantine(/*sdk_can_drain=*/false);
                throw StorageOperationTimeout("KVCM worker transfer timed out", false);
            }
            if (response.transfer_status() == REMOTE_TRANSFER_STATUS_COMPLETION_UNKNOWN) {
                host_pins.quarantine(/*sdk_can_drain=*/false);
                throw StorageOperationCompletionUnknown("KVCM worker transfer completion is unknown", false);
            }
            return success;
        };
        return payload ? runTransfer(std::move(transfer), impl_->allocationOwner(), invoke) : invoke({});
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("KVCM remote operation failed: %s", error.what());
        return false;
    }
}

}  // namespace rtp_llm
