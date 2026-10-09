#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <future>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <variant>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/CanonicalCacheLayout.h"
#include "rtp_llm/cpp/cache/WorkerCacheIOFence.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
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

// An RPC failure does not prove that a peer has stopped accessing its buffers.
class UncertainTransfer final: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

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

std::optional<size_t> paceUriParameter(std::string_view uri, std::string_view name) {
    const auto query = uri.find('?');
    if (query == std::string_view::npos) {
        return std::nullopt;
    }
    std::optional<size_t> result;
    for (size_t start = query + 1; start < uri.size();) {
        const auto end    = uri.find('&', start);
        const auto part   = uri.substr(start, end == std::string_view::npos ? end : end - start);
        const auto equals = part.find('=');
        if (part.substr(0, equals) == name) {
            result.reset();
            if (equals != std::string_view::npos) {
                const auto value = part.substr(equals + 1);
                size_t parsed = 0;
                const auto [last, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (error == std::errc{} && last == value.data() + value.size()) {
                    result = parsed;
                }
            }
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return result;
}

// Explicit SSD objects require whole-object I/O. Transparent tiering keeps a
// DRAM URI, so its multi-IOV path is not changed here.
class PaceSsdBuffer {
public:
    static bool needed(const std::string& uri, const kv_cache_manager::BlockBuffer& buffer) {
        return buffer.iovs.size() > 1 && uri.rfind("pace://", 0) == 0 && paceUriParameter(uri, "media_type") == 5;
    }

    PaceSsdBuffer(const std::string& uri, const kv_cache_manager::BlockBuffer& source, bool snapshot = false):
        original_(source) {
        const auto type       = original_.iovs.front().type;
        size_t     bytes      = 0;
        auto       next       = reinterpret_cast<uintptr_t>(original_.iovs.front().base);
        bool       contiguous = !snapshot;
        for (const auto& iov : original_.iovs) {
            const auto address = reinterpret_cast<uintptr_t>(iov.base);
            RTP_LLM_CHECK_WITH_INFO(iov.base && iov.size > 0 && !iov.ignore && iov.type == type
                                        && (type == kv_cache_manager::MemoryType::CPU
                                            || type == kv_cache_manager::MemoryType::GPU)
                                        && iov.size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) - bytes
                                        && iov.size <= std::numeric_limits<uintptr_t>::max() - address,
                                    "invalid KVCM SSD object buffer");
            contiguous = contiguous && address == next;
            next = address + iov.size;
            bytes += iov.size;
        }
        RTP_LLM_CHECK_WITH_INFO(paceUriParameter(uri, "size") == bytes, "KVCM SSD object size disagrees with URI");
        if (contiguous) {
            buffer_.iovs.push_back({type, original_.iovs.front().base, bytes, false});
        } else {
            storage_ = torch::empty(
                {static_cast<int64_t>(bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
                    .pinned_memory(type == kv_cache_manager::MemoryType::GPU));
            buffer_.iovs.push_back({kv_cache_manager::MemoryType::CPU, storage_.data_ptr(), bytes, false});
        }
    }

    const kv_cache_manager::BlockBuffer& buffer() const {
        return buffer_;
    }

    void gather() {
        copy(false);
    }

    void scatter() {
        copy(true);
    }

private:
    void copy(bool read) {
        if (!storage_.defined()) {
            return;
        }
        const bool gpu    = original_.iovs.front().type == kv_cache_manager::MemoryType::GPU;
        auto*      packed = storage_.data_ptr<uint8_t>();
        size_t     offset = 0;
#if USING_CUDA
        const auto stream = gpu ? at::cuda::getStreamFromPool().stream() : nullptr;
#else
        RTP_LLM_CHECK_WITH_INFO(!gpu, "KVCM SSD GPU packing requires CUDA");
#endif
        try {
            for (const auto& iov : original_.iovs) {
                auto* source = read ? packed + offset : static_cast<uint8_t*>(iov.base);
                auto* target = read ? static_cast<uint8_t*>(iov.base) : packed + offset;
                if (gpu) {
#if USING_CUDA
                    check_cuda_value(cudaMemcpyAsync(target,
                                                     source,
                                                     iov.size,
                                                     read ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                                     stream));
#endif
                } else {
                    std::memcpy(target, source, iov.size);
                }
                offset += iov.size;
            }
#if USING_CUDA
            if (gpu) {
                check_cuda_value(cudaStreamSynchronize(stream));
            }
#endif
        } catch (...) {
#if USING_CUDA
            // Earlier copies may have been submitted before a later copy failed.
            if (gpu) {
                cudaStreamSynchronize(stream);
            }
#endif
            throw;
        }
    }

    kv_cache_manager::BlockBuffer original_;
    kv_cache_manager::BlockBuffer buffer_;
    torch::Tensor                storage_;
};

}  // namespace

class KVCMStorageBackend::Impl: public std::enable_shared_from_this<KVCMStorageBackend::Impl> {
public:
    using ActualUriGather = std::vector<std::vector<kv_cache_manager::LocationSpecUnit*>>;

    Impl(const CacheConfig&                   cache_config,
         const KVCacheConfig&                 kv_cache_config,
         const RuntimeConfig&                 runtime_config,
         const ParallelismConfig&             parallelism_config,
         const SpeculativeExecutionConfig&    sp_config,
         std::shared_ptr<BroadcastManager>    broadcast_manager,
         std::shared_ptr<kvcm::ClientWrapper> client_wrapper,
         std::shared_ptr<WorkerCacheIOFence>  worker_fence):
        cache_config_(cache_config),
        kv_cache_config_(kv_cache_config),
        runtime_config_(runtime_config),
        parallelism_config_(parallelism_config),
        sp_config_(sp_config),
        broadcast_manager_(std::move(broadcast_manager)),
        client_wrapper_(std::move(client_wrapper)),
        worker_fence_(parallelism_config.tp_rank == 0 ?
                          nullptr :
                          (worker_fence ? std::move(worker_fence) : std::make_shared<WorkerCacheIOFence>())),
        sdk_check_enabled_(autil::EnvUtil::getEnv("KVCM_SDK_CHECK", autil::EnvUtil::getEnv("RECO_SDK_CHECK", false))) {}

    // The caller has a finite wait even when a synchronous SDK call never returns.
    // A quarantined call intentionally owns itself for the process lifetime: late
    // RPC/SDK completion is not sufficient evidence that every peer stopped I/O.
    template<typename Function>
    auto run(int timeout_ms, Function function) -> std::invoke_result_t<Function&> {
        using Result = std::invoke_result_t<Function&>;
        struct Call {
            explicit Call(Function fn): function(std::move(fn)) {}
            Function function;
            std::promise<Result> promise;
            std::shared_ptr<Call> quarantine;
        };
        if (failed_.load()) {
            throw std::runtime_error("KVCM backend disabled after uncertain I/O completion");
        }
        const size_t limit = std::max<size_t>(2, 2 * kv_cache_config_.kvcm_asyncwrapper_thread_num);
        const auto active = active_operations_.fetch_add(1);
        if (active >= limit) {
            active_operations_.fetch_sub(1);
            throw std::runtime_error("KVCM I/O concurrency limit reached");
        }
        auto self = shared_from_this();
        std::shared_ptr<Call> call;
        std::future<Result> result;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        try {
            call = std::make_shared<Call>(std::move(function));
            result = call->promise.get_future();
            std::thread([self, call] {
                try {
                    if constexpr (std::is_void_v<Result>) {
                        call->function();
                        call->promise.set_value();
                    } else {
                        call->promise.set_value(call->function());
                    }
                } catch (...) {
                    call->promise.set_exception(std::current_exception());
                }
                self->active_operations_.fetch_sub(1);
            }).detach();
        } catch (...) {
            active_operations_.fetch_sub(1);
            throw;
        }
        const auto quarantine = [&] {
            failed_.store(true);
            call->quarantine = call;
            RTP_LLM_LOG_ERROR("KVCM I/O completion unknown; retaining buffers and disabling remote operations");
        };
        if (result.wait_until(deadline) != std::future_status::ready) {
            quarantine();
            throw UncertainTransfer("KVCM operation timed out; buffers quarantined");
        }
        try {
            return result.get();
        } catch (const UncertainTransfer&) {
            quarantine();
            throw;
        }
    }

    int timeoutMs(bool write) const {
        return write ? kv_cache_config_.kvcm_put_broadcast_timeout : kv_cache_config_.kvcm_get_broadcast_timeout;
    }

    bool failed() const { return failed_.load(); }

    bool init(const CacheTopology&                                                            topology,
              kvcm::GroupPolicy::SourceResolver                                               buffer_resolver,
              const std::function<const DeviceBlockPoolPtr&(const std::string&)>&             pool_resolver,
              StorageBackend::HostBindingsByTag                                               host_bindings,
              StorageBackend::HostToDevice                                                    host_to_device,
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
        topology_owner_ = std::make_shared<const CacheTopology>(topology);
        topology_ = topology_owner_.get();
        host_bindings_  = requiresSharedHostMemory() ? std::move(host_bindings) : StorageBackend::HostBindingsByTag{};
        host_to_device_ = requiresSharedHostMemory() ? std::move(host_to_device) : StorageBackend::HostToDevice{};
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
                *topology_, buffer_resolver_, full_group_tags, other_group_tags, std::move(group_block_size_bytes));
        } else {
            group_policy_ = std::make_unique<kvcm::FullLinearLayerGroupPolicy>(*topology_,
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
        default_query_type_ = config->default_query_type();
        const int query_type = resolveQueryType(kv_cache_config_.kvcm_query_type);
        if (default_query_type_ < 1 || default_query_type_ > 4 || query_type < 1 || query_type > 4
            || kv_cache_config_.kvcm_min_replica_count < 0
            || kv_cache_config_.kvcm_get_broadcast_timeout <= 0 || kv_cache_config_.kvcm_put_broadcast_timeout <= 0
            || (query_type == 3 && kv_cache_config_.kvcm_sw_size <= 0)
            || (query_type == 4 && (other_group_tags.empty() || has_swa_))
            || !config->sdk_wrapper_config() || !config->sdk_wrapper_config()->drain_on_timeout()
            || (kv_cache_config_.kvcm_read_backend_type != 0
                && (!isPayloadBackend(kv_cache_config_.kvcm_read_backend_type)
                    || kv_cache_config_.kvcm_query_type > 1))) {
            RTP_LLM_LOG_ERROR("invalid KVCM query/replica/backend config or drain_on_timeout is disabled");
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
        has_read_pool_ = std::any_of(host_read_route_by_tag_.begin(),
                                     host_read_route_by_tag_.end(),
                                     [this](const auto& item) {
                                         return client_wrapper_->hasTransferClientForTag(item.second);
                                     });
        const auto tp_rank = parallelism_config_.tp_rank;
        RTP_LLM_LOG_INFO("BlockTree KVCM storage backend initialized, tp_rank=%ld tp_size=%ld policy={%s}",
                         tp_rank,
                         parallelism_config_.tp_size,
                         group_policy_->debugString().c_str());
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
        auto query_type = static_cast<kv_cache_manager::QueryType>(
            resolveQueryType(kv_cache_config_.kvcm_query_type));
        if (has_swa_ && query_type == kv_cache_manager::QueryType::QT_REVERSE_ROLL_SW_MATCH) {
            // A SWA query omits locations outside its window, including FULL
            // blocks needed for reuse. Fetch all keys and apply each group's
            // reuse policy locally; explicit metadata queries keep their mode.
            query_type = kv_cache_manager::QueryType::QT_BATCH_GET;
        }
        bool success = false;
        kv_cache_manager::Locations locations;
        bool positional = query_type == kv_cache_manager::QueryType::QT_BATCH_GET
                          || query_type == kv_cache_manager::QueryType::QT_REVERSE_ROLL_SW_MATCH;
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
            kv_cache_manager::ForwardContext context;
            context.sw_size = kv_cache_config_.kvcm_sw_size;
            auto result = client_wrapper_->match(
                "", trace_id, query_type, keys, request.local_matched_blocks_num, context);
            success = result.first;
            locations = std::move(result.second);
        }
        if (!success) {
            return {local_request.local_matched_blocks_num, nullptr};
        }
        if (positional || has_swa_ || canonical_layout_) {
            auto meta = std::make_shared<KVCMMatchMeta>();
            meta->locations = reusableLocations(request, std::move(locations), positional);
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
                           has_read_pool_ && host_to_device_ ? REMOTE_OPERATION_READ_HOST : REMOTE_OPERATION_READ,
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
        (void)dispatchRequests(requests, kv_cache_config_.kvcm_get_broadcast_timeout, nullptr);
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
                    RTP_LLM_CHECK_WITH_INFO(remote->block_ids_size() == 0
                                                || (remote->op() == REMOTE_OPERATION_WRITE_HOST) ==
                                                       (handle->tier == Tier::HOST),
                                            "KVCM write cannot mix HOST and DEVICE blocks in one rank request");
                    if (handle->tier == Tier::HOST) {
                        remote->set_op(REMOTE_OPERATION_WRITE_HOST);
                    }
                    appendTransfer(*remote, info->second, handle->block, location_spec.uri);
                    remote->add_coordinate_ids(static_cast<uint32_t>(key_idx));
                    actual_uri_gather[rank].push_back(&location_spec);
                }
            }

            for (auto& rank_request : requests) {
                auto* remote = rank_request.mutable_remote_request();
                if (remote->op() == REMOTE_OPERATION_WRITE_HOST) {
                    for (const auto key : keys) {
                        remote->add_cache_keys(key);
                    }
                }
            }
            if (request.host_payload_dispatched) {
                request.host_payload_dispatched->store(true, std::memory_order_release);
            }
            const auto responses = dispatchRequests(requests,
                                                    kv_cache_config_.kvcm_put_broadcast_timeout,
                                                    request.host_payload_dispatched.get());
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
            if (failed()) {
                throw UncertainTransfer("KVCM caller already timed out; do not publish late write results");
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
        } catch (const UncertainTransfer&) {
            // Do not abort/recycle remote destinations while a peer may still write.
            throw;
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

    bool ownsAllocator() const {
        return parallelism_config_.tp_rank == 0;
    }

    bool execute(const RemoteOperationRequestPB& request,
                 RemoteOperationResponsePB&      response,
                 bool*                           host_write_started = nullptr) {
        const bool host_read = request.op() == REMOTE_OPERATION_READ_HOST;
        const bool host_write = request.op() == REMOTE_OPERATION_WRITE_HOST;
        if (request.op() != REMOTE_OPERATION_READ && request.op() != REMOTE_OPERATION_WRITE
            && !host_read && !host_write) {
            return executeMetadata(request, response);
        }
        const std::vector<std::string>    tags(request.group_tags().begin(), request.group_tags().end());
        const std::vector<int32_t>        blocks(request.block_ids().begin(), request.block_ids().end());
        const kv_cache_manager::UriStrVec uris(request.uris().begin(), request.uris().end());
        if (tags.size() != blocks.size() || blocks.size() != uris.size()) {
            RTP_LLM_LOG_WARNING("KVCM transfer tag/block/URI count mismatch");
            return false;
        }
        if (request.host_source() && request.op() != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("HOST buffers are only supported for remote writes");
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
        if (host_read) {
            return executeHostRead(request, tags, blocks, uris, response);
        }
        if (host_write) {
            if (!genHostBlockBuffers(tags, blocks, buffers, false)) {
                return false;
            }
        } else if (canonical_layout_) {
            for (size_t index = 0; index < tags.size(); ++index) {
                kv_cache_manager::BlockBuffer buffer;
                for (int layer : cache_config_.layerIdsForGroup(tags[index])) {
                    const auto parts = canonical_layout_->slice(
                        layer,
                        tags[index],
                        request.logical_shards(index),
                        buffer_resolver_(
                            layer, tags[index], blocks[index], request.host_source() ? Tier::HOST : Tier::DEVICE));
                    for (const auto& part : parts) {
                        buffer.iovs.push_back({request.host_source() ? kv_cache_manager::MemoryType::CPU :
                                                                       kv_cache_manager::MemoryType::GPU,
                                               part.addr,
                                               part.size_bytes,
                                               false});
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
        } else if (!group_policy_->genBlockBuffers(
                       tags, blocks, buffers, request.host_source() ? Tier::HOST : Tier::DEVICE)) {
            return false;
        }
        return executeTagTransfers(host_write ? REMOTE_OPERATION_WRITE : request.op(),
                                   tags, blocks, uris, buffers, host_write, response,
                                   host_write ? host_write_started : nullptr, nullptr, request.host_source());
    }

    bool allocationOwner() const {
        return parallelism_config_.tp_rank == 0;
    }

    void shutdown() noexcept {
        if (client_wrapper_) {
            client_wrapper_->shutdown();
        }
    }

    bool requiresSharedHostMemory() const {
        return kv_cache_config_.kvcm_remote_layout == "legacy";
    }

    bool canInitiateHostWrite() const {
        // Normal request HOST publication is owned by TP0. Follower key mappings
        // must be synchronized before request-finish writes can reuse them.
        return !canonical_layout_ && parallelism_config_.tp_rank == 0 && parallelism_config_.tp_size == 1;
    }

private:
    int resolveQueryType(int query_type) const {
        return query_type == 0 ? default_query_type_ : query_type;
    }

    bool genHostBlockBuffers(const std::vector<std::string>& tags,
                             const std::vector<int32_t>&     blocks,
                             kv_cache_manager::BlockBuffers& buffers,
                             bool                            read_pool) const {
        RTP_LLM_CHECK(tags.size() == blocks.size());
        buffers.reserve(blocks.size());
        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto binding = host_bindings_.find(tags[i]);
            if (binding == host_bindings_.end()) {
                return false;
            }
            const auto& pool = read_pool ? binding->second.read_pool : binding->second.pool;
            if (!pool || pool->sharedMemoryFd() < 0 || pool->hasUncertainRemoteIo()
                || !pool->isAllocated(blocks[i])) {
                return false;
            }
            const auto block = pool->blockBuffer(blocks[i]);
            RTP_LLM_CHECK_WITH_INFO(binding->second.offset_bytes <= block.payload_bytes
                                        && binding->second.payload_bytes <= block.payload_bytes - binding->second.offset_bytes,
                                    "KVCM HOST buffer exceeds its pool block");
            auto* address = static_cast<uint8_t*>(block.addr) + binding->second.offset_bytes;
            kv_cache_manager::BlockBuffer buffer;
            buffer.iovs.push_back({kv_cache_manager::MemoryType::CPU,
                                   address,
                                   binding->second.payload_bytes,
                                   false});
            buffers.push_back(std::move(buffer));
        }
        return true;
    }

    bool executeHostRead(const RemoteOperationRequestPB&    request,
                         const std::vector<std::string>&    tags,
                         const std::vector<int32_t>&        device_blocks,
                         const kv_cache_manager::UriStrVec& uris,
                         RemoteOperationResponsePB&         response) {
        using TargetKey = std::pair<uint32_t, size_t>;
        struct Target {
            std::shared_ptr<HostBlockPool> pool;
            size_t                         group_set_id;
            std::vector<BlockIdxType>      device_blocks;
            BlockIdxType                   host_block{NULL_BLOCK_IDX};
        };
        auto device_read = [&] {
            kv_cache_manager::BlockBuffers buffers;
            return group_policy_->genBlockBuffers(tags, device_blocks, buffers)
                   && executeTagTransfers(REMOTE_OPERATION_READ, tags, device_blocks, uris, buffers, false, response,
                                          nullptr);
        };
        if (!host_to_device_ || request.coordinate_ids_size() != static_cast<int>(tags.size())) {
            return device_read();
        }
        std::map<TargetKey, Target> targets;
        std::vector<TargetKey>      item_keys;
        item_keys.reserve(tags.size());
        for (size_t i = 0; i < tags.size(); ++i) {
            const auto binding = host_bindings_.find(tags[i]);
            const auto route = host_read_route_by_tag_.find(tags[i]);
            if (binding == host_bindings_.end() || !binding->second.read_pool
                || binding->second.read_pool->sharedMemoryFd() < 0
                || binding->second.read_pool->hasUncertainRemoteIo()
                || route == host_read_route_by_tag_.end()
                || !client_wrapper_->hasTransferClientForTag(route->second)) {
                return device_read();
            }
            const auto& info = binding->second;
            const TargetKey key{request.coordinate_ids(static_cast<int>(i)), info.group_set_id};
            auto [it, inserted] = targets.try_emplace(key,
                                                      Target{info.read_pool,
                                                             info.group_set_id,
                                                             std::vector<BlockIdxType>(info.member_count, NULL_BLOCK_IDX)});
            (void)inserted;
            if (it->second.pool != info.read_pool || info.member_index >= it->second.device_blocks.size()
                || !isNullBlockIdx(it->second.device_blocks[info.member_index])) {
                return device_read();
            }
            it->second.device_blocks[info.member_index] = device_blocks[i];
            item_keys.push_back(key);
        }
        for (const auto& item : targets) {
            const auto& target = item.second;
            if (std::any_of(target.device_blocks.begin(), target.device_blocks.end(), isNullBlockIdx)) {
                return device_read();
            }
        }
        bool host_read_in_flight = false;
        block_tree_cache_detail::ScopeRollback release_targets([&]() noexcept {
            for (const auto& item : targets) {
                const auto& target = item.second;
                if (isNullBlockIdx(target.host_block)) {
                    continue;
                }
                if (host_read_in_flight) {
                    target.pool->markUncertainRemoteIo();
                    continue;
                }
                try {
                    // incTreeRef can fail after malloc but before acquiring the LOAD reference.
                    if (target.pool->treeRefCount(target.host_block) == 0) {
                        target.pool->incTreeRef(target.host_block, BlockTreeRefType::LOAD);
                    }
                    target.pool->decTreeRef(target.host_block, BlockTreeRefType::LOAD);
                } catch (...) {
                    target.pool->markUncertainRemoteIo();
                    RTP_LLM_LOG_WARNING("KVCM HOST read cleanup failed; isolating the read pool");
                }
            }
        });
        for (auto& item : targets) {
            auto& target = item.second;
            const auto block = target.pool->malloc();
            if (!block) {
                release_targets.run();
                return device_read();
            }
            target.host_block = *block;
            target.pool->incTreeRef(*block, BlockTreeRefType::LOAD);
        }
        std::vector<int32_t> host_blocks;
        host_blocks.reserve(item_keys.size());
        for (const auto& key : item_keys) {
            host_blocks.push_back(targets.at(key).host_block);
        }
        kv_cache_manager::BlockBuffers buffers;
        bool success = genHostBlockBuffers(tags, host_blocks, buffers, true)
                       && executeTagTransfers(REMOTE_OPERATION_READ, tags, host_blocks, uris, buffers, true, response,
                                              nullptr, &host_read_in_flight);
        if (success) {
            std::vector<TransferDescriptor> copies;
            std::vector<HostBufferView>     views;
            copies.reserve(targets.size());
            views.reserve(targets.size());
            for (const auto& item : targets) {
                const auto& target = item.second;
                copies.push_back(TransferDescriptor::hostToDevice(
                    target.group_set_id, target.host_block, target.device_blocks));
                const auto host = target.pool->blockBuffer(target.host_block);
                views.push_back({host.addr, host.payload_bytes, host.stride_bytes});
            }
            host_read_in_flight = true;
            success = host_to_device_(std::move(copies), std::move(views),
                                      std::max(1, kv_cache_config_.kvcm_get_broadcast_timeout));
            host_read_in_flight = !success;
        }
        if (host_read_in_flight) {
            RTP_LLM_LOG_WARNING("KVCM HOST read failed; retaining %zu HOST blocks until pool destruction",
                                targets.size());
        }
        return success;
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
        StorageRequest expanded;
        expanded.source_tier = request.source_tier;
        expanded.keys =
            std::make_shared<const CacheKeysType>(request.remote_keys->begin(), request.remote_keys->begin() + count);
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
        remote.add_block_generations((remote.host_source() || remote.op() == REMOTE_OPERATION_WRITE_HOST) ?
                                         host_pool_resolver_(info.tag)->blockAllocationGeneration(block) :
                                         pool_resolver_(info.tag)->blockAllocationGeneration(block));
        if (canonical_layout_) {
            remote.add_logical_shards(info.shard);
        }
    }

    static bool isPayloadBackend(int type) {
        return type == 1 || type == 2 || type == 3 || type == 4 || type == 5 || type == 9;
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
                                                   kv_cache_manager::Locations locations,
                                                   bool positional) const {
        const size_t local = request.local_matched_blocks_num;
        const size_t key_count = request.keys->size();
        if (positional) {
            RTP_LLM_CHECK_WITH_INFO(locations.size() == key_count, "KVCM positional query shape mismatch");
        } else {
            RTP_LLM_CHECK_WITH_INFO(locations.size() <= key_count - local, "KVCM prefix query shape mismatch");
            locations.insert(locations.begin(), local, kv_cache_manager::Location{});
        }
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
        if (type < 1 || type > 4 || query.detail_level() < 0 || query.p2p_host_count() < 0) {
            return false;
        }
        if (type == 4 && has_swa_
            && (request.op() == REMOTE_OPERATION_MATCH_LOCATION
                || request.op() == REMOTE_OPERATION_MATCH_LOCATION_LEN
                || request.op() == REMOTE_OPERATION_GET_HOST_CACHE_STATE)) {
            RTP_LLM_LOG_WARNING("KVCM Mamba queries require a FULL+LINEAR layout without SWA groups");
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
            case REMOTE_OPERATION_MATCH_LOCATION_LEN: {
                auto [success, length] = client_wrapper_->matchLocationLen(
                    "", request.trace_id(), query_type, keys, tokens, query.sw_size());
                if (success) {
                    response.set_matched_blocks(length);
                }
                return success;
            }
            case REMOTE_OPERATION_MATCH_META: {
                auto [success, metas] = client_wrapper_->matchMeta(
                    "", request.trace_id(), keys, tokens, mask, query.detail_level());
                if (success) {
                    for (const auto& location : metas.locations) {
                        appendLocation(location, response.add_locations());
                    }
                    for (const auto& meta : metas.metas) {
                        response.add_metas(meta);
                    }
                }
                return success;
            }
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
                if (type != 2 && type != 4) {
                    RTP_LLM_LOG_WARNING("KVCM host-state queries require prefix or Mamba mode, got %d", type);
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
            case REMOTE_OPERATION_MATCH_LOCATION: {
                auto [success, locations] = client_wrapper_->queryLocations(
                    "", request.trace_id(), query_type, keys, tokens, mask, query.sw_size(), names);
                if (success) {
                    for (const auto& location : locations) {
                        appendLocation(location, response.add_locations());
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
                 << ";default_query_type:" << kv_cache_config_.kvcm_default_query_type
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

        auto config = std::make_shared<kvcm::KVCMConfig>(
            kv_cache_config_.kvcm_enable_vipserver,
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
        config->set_default_query_type(kv_cache_config_.kvcm_default_query_type);
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
        host_route_by_tag_.clear();
        host_read_route_by_tag_.clear();
        has_read_pool_ = false;
        std::vector<kvcm::ClientWrapper::PoolRegistration> registrations;
        for (int32_t group_id : ordered_groups) {
            const auto& tag  = groups.at(group_id).tag;
            const auto& pool = pool_resolver(tag);
            if (!pool->getBaseAddress() || pool->getTotalSizeBytes() == 0) {
                RTP_LLM_LOG_ERROR("KVCM group %d has no valid registration span", group_id);
                return {};
            }
            registration_tags_.push_back(tag);
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
            registrations.push_back({{pool->getBaseAddress(), pool->getTotalSizeBytes()}, registration_spec});
        }
        std::unordered_map<size_t, std::string> route_by_group_set;
        for (size_t index = 0, device_count = registration_tags_.size(); index < device_count; ++index) {
            const auto& tag = registration_tags_[index];
            const auto binding = host_bindings_.find(tag);
            if (binding == host_bindings_.end()) {
                continue;
            }
            const auto& host = binding->second;
            if (host.pool->sharedMemoryFd() < 0) {
                continue;
            }
            const auto [route_it, inserted] = route_by_group_set.emplace(
                host.group_set_id, std::string("\x1fhost:") + std::to_string(host.group_set_id));
            host_route_by_tag_.emplace(tag, route_it->second);
            const std::string read_route = std::string("\x1fread:") + std::to_string(host.group_set_id);
            if (host.read_pool) {
                host_read_route_by_tag_.emplace(tag, read_route);
            }
            if (inserted) {
                const auto base = host.pool->sharedMemoryBase();
                const auto size = host.pool->sharedMemorySize();
                registration_tags_.push_back(route_it->second);
                registrations.push_back({{base, size},
                                         registrations[index].location_spec_name,
                                         kv_cache_manager::SharedMemoryRegistration{base, size,
                                                                                     host.pool->sharedMemoryFd()}});
                if (host.read_pool) {
                    const auto read_base = host.read_pool->sharedMemoryBase();
                    const auto read_size = host.read_pool->sharedMemorySize();
                    registration_tags_.push_back(read_route);
                    registrations.push_back({{read_base, read_size},
                                             registrations[index].location_spec_name,
                                             kv_cache_manager::SharedMemoryRegistration{
                                                 read_base, read_size, host.read_pool->sharedMemoryFd()}});
                    has_read_pool_ = true;
                }
            }
        }
        return registrations;
    }

    bool executeTagTransfers(RemoteOpType                       operation,
                             const std::vector<std::string>&    tags,
                             const std::vector<int32_t>&        blocks,
                             const kv_cache_manager::UriStrVec& uris,
                             kv_cache_manager::BlockBuffers&    buffers,
                             bool                               host_buffers,
                             RemoteOperationResponsePB&         response,
                             bool*                              host_write_started,
                             bool*                              host_read_in_flight = nullptr,
                             bool                               snapshot_host = false) {
        if (operation != REMOTE_OPERATION_READ && operation != REMOTE_OPERATION_WRITE) {
            RTP_LLM_LOG_WARNING("KVCM transfer has invalid operation [%d]", operation);
            return false;
        }
        std::unordered_map<std::string, std::vector<size_t>> indices_by_tag;
        const auto& host_routes = operation == REMOTE_OPERATION_READ ? host_read_route_by_tag_ : host_route_by_tag_;
        for (size_t index = 0; index < tags.size(); ++index) {
            const auto host_route = host_routes.find(tags[index]);
            const std::string& route = host_buffers && host_route != host_routes.end()
                                           && client_wrapper_->hasTransferClientForTag(host_route->second) ?
                                           host_route->second : tags[index];
            indices_by_tag[route].push_back(index);
        }
        auto actual_uris = uris;
        for (const auto& tag : registration_tags_) {
            const auto found = indices_by_tag.find(tag);
            if (found == indices_by_tag.end()) {
                continue;
            }
            struct TransferBuffers {
                std::vector<std::unique_ptr<PaceSsdBuffer>> buffers;
                std::vector<std::vector<uint8_t>> host_payloads;
                std::shared_ptr<TransferBuffers> quarantine;
            };
            auto transfer_buffers = std::make_shared<TransferBuffers>();
            try {
                // Separate adapted CPU/GPU objects from the unchanged multi-IOV path.
                std::vector<std::vector<size_t>> batches(3);
                for (size_t index : found->second) {
                    auto& buffer = buffers[index];
                    size_t batch = 0;
                    if (PaceSsdBuffer::needed(uris[index], buffer)) {
                        // HOST snapshots use the same copy that packs the SSD object.
                        transfer_buffers->buffers.push_back(
                            std::make_unique<PaceSsdBuffer>(uris[index], buffer, snapshot_host));
                        auto& ssd = transfer_buffers->buffers.back();
                        if (operation == REMOTE_OPERATION_WRITE) {
                            ssd->gather();
                        }
                        buffer = ssd->buffer();
                        batch = buffer.iovs.front().type == kv_cache_manager::MemoryType::CPU ? 1 : 2;
                    } else if (snapshot_host) {
                        for (auto& iov : buffer.iovs) {
                            transfer_buffers->host_payloads.emplace_back(iov.size);
                            auto& payload = transfer_buffers->host_payloads.back();
                            std::memcpy(payload.data(), iov.base, iov.size);
                            iov.base = payload.data();
                        }
                    }
                    batches[batch].push_back(index);
                }
                for (const auto& indices : batches) {
                    if (indices.empty()) {
                        continue;
                    }
                    kv_cache_manager::UriStrVec    batch_uris;
                    kv_cache_manager::BlockBuffers batch_buffers;
                    std::vector<int32_t>           batch_blocks;
                    for (size_t index : indices) {
                        batch_uris.push_back(uris[index]);
                        batch_buffers.push_back(std::move(buffers[index]));
                        batch_blocks.push_back(blocks[index]);
                    }
                    auto trace_info = makeTransferTraceInfo(batch_blocks);
                    if (host_buffers || snapshot_host || (!transfer_buffers->buffers.empty()
                        && batch_buffers.front().iovs.front().type == kv_cache_manager::MemoryType::CPU)) {
                        if (!trace_info) {
                            trace_info = std::make_shared<kv_cache_manager::TransferTraceInfo>();
                        }
                        trace_info->need_print = false;
                    }
                    if (operation == REMOTE_OPERATION_READ) {
                        if (host_read_in_flight) {
                            *host_read_in_flight = true;
                        }
                        if (!client_wrapper_->loadKvCachesForTag(tag, batch_uris, batch_buffers, trace_info)) {
                            throw UncertainTransfer("KVCM SDK read failed without confirmed I/O completion");
                        }
                        if (host_read_in_flight) {
                            *host_read_in_flight = false;
                        }
                    } else {
                        if (host_write_started) {
                            *host_write_started = true;
                        }
                        auto [success, result] =
                            client_wrapper_->saveKvCachesForTag(tag, batch_uris, batch_buffers, trace_info);
                        if (!success) {
                            throw UncertainTransfer("KVCM SDK write failed without confirmed I/O completion");
                        }
                        if (!result.empty() && result.size() != batch_uris.size()) {
                            return false;
                        }
                        for (size_t index = 0; index < result.size(); ++index) {
                            actual_uris[indices[index]] = std::move(result[index]);
                        }
                    }
                }
                if (operation == REMOTE_OPERATION_READ) {
                    for (const auto& ssd : transfer_buffers->buffers) {
                        ssd->scatter();
                    }
                }
            } catch (...) {
                // A stalled call keeps its stack alive. If SDK/CUDA instead reports
                // an error, retain SSD staging and HOST snapshots before unwinding.
                if (!transfer_buffers->buffers.empty() || !transfer_buffers->host_payloads.empty()) {
                    transfer_buffers->quarantine = transfer_buffers;
                }
                throw UncertainTransfer("KVCM SDK transfer or SSD copy failed without confirmed I/O completion");
            }
        }
        if (operation == REMOTE_OPERATION_WRITE && actual_uris != uris) {
            for (auto& uri : actual_uris) {
                *response.add_actual_uris() = std::move(uri);
            }
        }
        return true;
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

    std::vector<FunctionResponsePB>
    dispatchRequests(const std::vector<FunctionRequestPB>& requests,
                     int                                   timeout_ms,
                     std::atomic<bool>*                    local_host_payload_dispatched) {
        if (!broadcast_manager_
            || (requests.size() == 1 && requests.front().remote_request().op() == REMOTE_OPERATION_WRITE_HOST)) {
            RTP_LLM_CHECK_WITH_INFO(
                requests.size() == 1, "KVCM local transfer requires exactly one request, got %zu", requests.size());
            FunctionResponsePB response;
            bool host_write_started = false;
            bool success = false;
            try {
                success = execute(requests.front().remote_request(),
                                  *response.mutable_remote_response(),
                                  &host_write_started);
            } catch (...) {
                if (!host_write_started && local_host_payload_dispatched) {
                    local_host_payload_dispatched->store(false, std::memory_order_release);
                }
                throw;
            }
            if ((success || !host_write_started) && local_host_payload_dispatched) {
                local_host_payload_dispatched->store(false, std::memory_order_release);
            }
            RTP_LLM_CHECK_WITH_INFO(success, "KVCM local transfer failed");
            return {std::move(response)};
        }
        auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                           const std::shared_ptr<grpc::ClientContext>& context,
                           const FunctionRequestPB&                    request,
                           grpc::CompletionQueue*                      queue) {
            return stub->AsyncExecuteFunction(context.get(), request, queue);
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        // RPC completion alone cannot release the controller's physical blocks.
        // Unknown peer completion quarantines the owning operation's pins.
        try {
            auto result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
                requests, timeout_ms, rpc_call, /*enforce_rpc_deadline=*/false);
            if (!result && local_host_payload_dispatched) {
                local_host_payload_dispatched->store(false, std::memory_order_release);
            }
            RTP_LLM_CHECK_WITH_INFO(result != nullptr, "KVCM broadcast dispatch failed");
            const auto mark_self_safe = [&] {
                if (local_host_payload_dispatched
                    && result->rankCompletedSuccessfully(static_cast<size_t>(parallelism_config_.tp_rank))) {
                    local_host_payload_dispatched->store(false, std::memory_order_release);
                }
            };
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
            bool in_budget = false;
            try {
                in_budget = remaining > 0 && result->waitDone(static_cast<int>(remaining));
            } catch (...) {
                mark_self_safe();
                throw;
            }
            mark_self_safe();
            if (!in_budget || !result->success()) {
                throw UncertainTransfer("KVCM peer completion unknown; buffers must remain quarantined");
            }
            return result->responses();
        } catch (const UncertainTransfer&) {
            throw;
        } catch (...) {
            // Dispatch may throw after earlier ranks have already started I/O.
            throw UncertainTransfer("KVCM broadcast failed without confirmed peer completion");
        }
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
    std::shared_ptr<BroadcastManager>    broadcast_manager_;
    std::unique_ptr<kvcm::GroupPolicy>   group_policy_;
    std::unique_ptr<kvcm::CanonicalCacheLayout>                              canonical_layout_;
    kvcm::GroupPolicy::SourceResolver                                        buffer_resolver_;
    kvcm::GroupPolicy::LocationSpecGroups                                    location_spec_groups_;
    std::shared_ptr<kvcm::ClientWrapper> client_wrapper_;
    std::shared_ptr<WorkerCacheIOFence>                                      worker_fence_;
    std::function<const DeviceBlockPoolPtr&(const std::string&)>             pool_resolver_;
    std::function<const std::shared_ptr<HostBlockPool>&(const std::string&)> host_pool_resolver_;
    std::vector<std::string>             registration_tags_;
    StorageBackend::HostBindingsByTag       host_bindings_;
    StorageBackend::HostToDevice         host_to_device_;
    std::unordered_map<std::string, std::string> host_route_by_tag_;
    std::unordered_map<std::string, std::string> host_read_route_by_tag_;
    bool has_read_pool_{false};
    // Preserve KVCM's operation-local, one-based request
    // order. Abort and finish share a sequence because both call FinishWrite.
    std::atomic<uint64_t> match_trace_sequence_{1};
    std::atomic<uint64_t> read_trace_sequence_{1};
    std::atomic<uint64_t> write_trace_sequence_{1};
    std::atomic<uint64_t> finish_write_trace_sequence_{1};
    const bool            sdk_check_enabled_;
    std::atomic<bool> failed_{false};
    std::atomic<size_t> active_operations_{0};
    std::shared_ptr<const CacheTopology> topology_owner_;
    const CacheTopology*  topology_ = nullptr;
    bool                  has_swa_ = false;
    int32_t               default_query_type_ = 2;
    int32_t               cache_key_tokens_ = 0;
};

KVCMStorageBackend::KVCMStorageBackend(const CacheConfig&                   cache_config,
                                       const KVCacheConfig&                 kv_cache_config,
                                       const RuntimeConfig&                 runtime_config,
                                       const ParallelismConfig&             parallelism_config,
                                       const SpeculativeExecutionConfig&    sp_config,
                                       std::shared_ptr<BroadcastManager>    broadcast_manager,
                                       std::shared_ptr<kvcm::ClientWrapper> client_wrapper,
                                       std::shared_ptr<WorkerCacheIOFence>  worker_fence):
    StorageBackend(makeStorageBackendExecutor(kv_cache_config.kvcm_asyncwrapper_thread_num,
                                              kv_cache_config.kvcm_asyncwrapper_queue_size)),
    impl_(std::make_shared<Impl>(cache_config,
                                 kv_cache_config,
                                 runtime_config,
                                 parallelism_config,
                                 sp_config,
                                 std::move(broadcast_manager),
                                 std::move(client_wrapper),
                                 std::move(worker_fence))) {}

KVCMStorageBackend::~KVCMStorageBackend() = default;

bool KVCMStorageBackend::initImpl() {
    return impl_->init(
        topology(),
        [device = bufferResolver(), host = bufferResolver(Tier::HOST)](
            int layer_id, const std::string& tag, int block_id, Tier source) {
            RTP_LLM_CHECK(source == Tier::DEVICE || source == Tier::HOST);
            const auto& resolver = source == Tier::HOST ? host : device;
            RTP_LLM_CHECK(resolver);
            return resolver(layer_id, tag, block_id);
        },
        [pools = devicePools()](const std::string& tag) -> const DeviceBlockPoolPtr& { return pools.at(tag); },
        hostPoolsByTag(),
        hostToDevice(),
        [pools = hostPools()](const std::string& tag) -> const std::shared_ptr<HostBlockPool>& { return pools.at(tag); });
}

bool KVCMStorageBackend::requiresSharedHostMemory() const {
    return impl_->requiresSharedHostMemory();
}

bool KVCMStorageBackend::canInitiateHostWrite() const {
    return impl_->canInitiateHostWrite();
}

StorageMatchResult KVCMStorageBackend::matchImpl(const StorageRequest& request) {
    return impl_->run(impl_->timeoutMs(false), [impl = impl_, request] { return impl->match(request); });
}

void KVCMStorageBackend::readImpl(const StorageRequest& request,
                                  const std::shared_ptr<StorageBackendMatchMeta>& match_meta) {
    auto pins = prepareWrite(request);
    impl_->run(impl_->timeoutMs(false),
               [impl = impl_, request, match_meta, pins = std::move(pins)] { impl->read(request, match_meta); });
}

void KVCMStorageBackend::writeImpl(const StorageRequest& request) {
    auto pins = prepareWrite(request);
    impl_->run(impl_->timeoutMs(true),
               [impl = impl_, request, pins = std::move(pins)] { impl->write(request); });
}

void KVCMStorageBackend::shutdownImpl() noexcept {
    // Timed-out calls retain their own SDK, pools and pins. Never join them here.
    if (impl_->failed()) {
        return;
    }
    try {
        // Invalid configuration can reach init cleanup before any I/O was admitted.
        const int budget = impl_->timeoutMs(true) > 0 ? impl_->timeoutMs(true)
                                                      : KVCacheConfig{}.kvcm_put_broadcast_timeout;
        impl_->run(budget, [impl = impl_] { impl->shutdown(); });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("KVCM shutdown did not finish: %s", error.what());
    }
}

bool KVCMStorageBackend::execute(const RemoteOperationRequestPB& request, RemoteOperationResponsePB& response) {
    try {
        RemoteOperationRequestPB local_request = request;
        StorageWriteTask host_pins;
        StorageRequest transfer;
        const bool host_write = request.op() == REMOTE_OPERATION_WRITE_HOST;
        const bool payload = host_write || request.op() == REMOTE_OPERATION_READ
                             || request.op() == REMOTE_OPERATION_WRITE || request.op() == REMOTE_OPERATION_READ_HOST;
        if (host_write) {
            if (!hostWriteResolver() || request.group_tags_size() != request.block_ids_size()
                || request.group_tags_size() != request.coordinate_ids_size()
                || request.block_ids_size() != request.uris_size() || request.cache_keys_size() == 0) {
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
        const auto invoke = [&](StorageWriteTask transfer_pins) {
            // The retained call owns RPC data and both resolved HOST and transfer refs.
            const bool write = host_write || request.op() == REMOTE_OPERATION_WRITE;
            auto result = impl_->run(impl_->timeoutMs(write),
                                     [impl = impl_, request = std::move(local_request),
                                      pins = std::move(host_pins), transfer_pins = std::move(transfer_pins)]() mutable {
                RemoteOperationResponsePB result;
                bool host_write_started = false;
                try {
                    const bool success = impl->execute(request, result, &host_write_started);
                    if (!success && host_write_started) {
                        pins.quarantine();
                    }
                    return std::make_pair(success, std::move(result));
                } catch (...) {
                    if (host_write_started) {
                        pins.quarantine();
                    }
                    throw;
                }
            });
            response = std::move(result.second);
            return result.first;
        };
        return payload ? runTransfer(std::move(transfer), impl_->allocationOwner(), invoke) : invoke({});
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("KVCM remote operation failed: %s", error.what());
        return false;
    }
}

}  // namespace rtp_llm
