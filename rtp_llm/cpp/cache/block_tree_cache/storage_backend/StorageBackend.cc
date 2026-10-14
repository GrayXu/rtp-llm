#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <algorithm>
#include <mutex>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::storage_backend_detail {

thread_local const StorageBackend* completing_backend = nullptr;

template<typename Callback>
void invokeCallback(const StorageBackend* backend, Callback&& callback) noexcept {
    struct CompletionScope {
        const StorageBackend* previous = completing_backend;
        ~CompletionScope() {
            completing_backend = previous;
        }
    } scope;
    completing_backend = backend;
    try {
        callback();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("StorageBackend completion failed: %s", error.what());
    } catch (...) { RTP_LLM_LOG_ERROR("StorageBackend completion failed with an unknown exception"); }
}

struct StorageTaskState {
    struct Pin {
        DeviceBlockPoolPtr pool;
        std::shared_ptr<HostBlockPool> host_pool;
        BlockIdxType       block;
    };

    StorageBackend* owner{nullptr};
    bool quarantined{false};
    bool sdk_can_drain{true};
    StorageRequest   request;
    std::vector<Pin> pins;
    std::vector<std::shared_ptr<IBlockPool>> borrowed_pools;
    std::once_flag   finish_once;
    void             finish() {
        std::call_once(finish_once, [this] {
            for (const Pin& pin : pins) {
                if (pin.host_pool) {
                    pin.host_pool->decTreeRef(pin.block, BlockTreeRefType::STORE);
                } else {
                    pin.pool->decRef(pin.block);
                }
            }
            pins.clear();
        });
    }

    ~StorageTaskState() {
        finish();
    }
};

}  // namespace rtp_llm::storage_backend_detail

namespace rtp_llm {
namespace {

struct BlockKey {
    IBlockPool*      pool;
    BlockIdxType     block;
    bool             operator==(const BlockKey& other) const {
        return pool == other.pool && block == other.block;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
        return std::hash<IBlockPool*>{}(key.pool) ^ (std::hash<BlockIdxType>{}(key.block) << 1U);
    }
};

}  // namespace

StorageWriteTask::StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state):
    state_(std::move(state)) {}

void StorageWriteTask::quarantine(bool sdk_can_drain) {
    if (state_) {
        state_->owner->quarantineTask(state_, sdk_can_drain);
    }
}

StorageBackend::StorageBackend(std::shared_ptr<StorageBackendExecutor> executor): executor_(std::move(executor)) {}

StorageBackend::~StorageBackend() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK_WITH_INFO(lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED,
                            "StorageBackend must be shutdown before derived destruction");
}

bool StorageBackend::init(std::shared_ptr<const CacheTopology> topology,
                          PoolsByTag                           pools_by_tag,
                          BufferResolver                       buffer_resolver,
                          HostPoolsByTag                       host_pools_by_tag,
                          BufferResolver                       host_buffer_resolver,
                          HostWriteResolver                    host_write_resolver,
                          HostWriteSettlementResolver           host_write_settlement_resolver) {
    if (init_attempted_) {
        RTP_LLM_LOG_ERROR("StorageBackend initialization has already been attempted");
        return false;
    }
    RTP_LLM_CHECK(topology && buffer_resolver);
    RTP_LLM_CHECK(pools_by_tag.size() == topology->groups().size());
    std::unordered_set<const DeviceBlockPool*> seen_pools;
    for (const auto& [tag, pool] : pools_by_tag) {
        (void)topology->group(tag);
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "null storage pool for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            seen_pools.emplace(pool.get()).second, "storage tags must own distinct pools: tag=%s", tag.c_str());
    }
    RTP_LLM_CHECK_WITH_INFO(host_pools_by_tag.empty() || host_buffer_resolver,
                            "host storage pools require a buffer resolver");
    for (const auto& [tag, pool] : host_pools_by_tag) {
        (void)topology->group(tag);
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "null host storage pool for tag=%s", tag.c_str());
    }
    init_attempted_ = true;
    if (executor_ == nullptr) {
        executor_ = makeDefaultStorageBackendExecutor();
    }
    if (executor_->bound_to_backend_.exchange(true)) {
        RTP_LLM_LOG_ERROR("StorageBackend executor cannot be shared between backends");
        return false;
    }
    topology_            = std::move(topology);
    pools_by_tag_        = std::move(pools_by_tag);
    buffer_resolver_     = std::move(buffer_resolver);
    host_pools_by_tag_ = std::move(host_pools_by_tag);
    host_buffer_resolver_ = std::move(host_buffer_resolver);
    host_write_resolver_ = std::move(host_write_resolver);
    host_write_settlement_resolver_ = std::move(host_write_settlement_resolver);
    const auto fail_init = [this] {
        shutdownImpl();
        buffer_resolver_ = {};
        pools_by_tag_.clear();
        host_buffer_resolver_ = {};
        host_pools_by_tag_.clear();
        host_write_resolver_ = {};
        host_write_settlement_resolver_ = {};
        topology_.reset();
        return false;
    };
    bool impl_initialized = false;
    try {
        impl_initialized = initImpl();
    } catch (...) {}
    if (!impl_initialized) {
        // The executor has never started and may still be used by another backend.
        executor_->bound_to_backend_.store(false);
        return fail_init();
    }
    bool started = false;
    try {
        started = executor_->start();
    } catch (...) {}
    if (!started) {
        executor_->shutdown();
        // Keep the binding: shutdown executors are terminal and must not be reused.
        return fail_init();
    }
    initialized_ = true;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::ACCEPTING;
    }
    return true;
}

bool StorageBackend::dispatch(Operation operation) {
    const auto once = std::make_shared<std::once_flag>();
    Lifecycle  outcome;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        outcome = lifecycle_;
        if (outcome == Lifecycle::FINALIZING) {
            outcome = Lifecycle::STOPPED;
        } else if (outcome != Lifecycle::STOPPED) {
            RTP_LLM_CHECK(outcome != Lifecycle::CREATED);
            ++in_flight_;
        }
    }
    if (outcome == Lifecycle::STOPPED) {
        try {
            operation(outcome);
        } catch (...) {}
        return false;
    }
    auto complete = [this, once, operation = std::move(operation)](Lifecycle result) mutable {
        std::call_once(*once, [&] {
            storage_backend_detail::invokeCallback(this, [&] { operation(result); });
            // invokeCallback isolates all exceptions, so every admitted task
            // reaches this exactly-once accounting boundary.
            taskFinished();
        });
    };
    if (outcome != Lifecycle::ACCEPTING) {
        complete(Lifecycle::STOPPING);
        return false;
    }
    try {
        if (executor_->submit([complete]() mutable { complete(Lifecycle::ACCEPTING); })) {
            return true;
        }
    } catch (...) {}
    complete(Lifecycle::STOPPING);
    return false;
}

void StorageBackend::taskFinished() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK(in_flight_ > 0);
    if (--in_flight_ == 0) {
        lifecycle_cv_.notify_all();
    }
}

void StorageBackend::shutdown() {
    RTP_LLM_CHECK_WITH_INFO(storage_backend_detail::completing_backend != this,
                            "StorageBackend shutdown cannot run from its callback");
    std::shared_ptr<StorageBackendExecutor> executor;
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        if (lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED) {
            return;
        }
        if (lifecycle_ != Lifecycle::ACCEPTING) {
            lifecycle_cv_.wait(lock, [this] { return lifecycle_ == Lifecycle::STOPPED; });
            return;
        }
        lifecycle_ = Lifecycle::STOPPING;
        executor   = executor_;
    }
    executor->shutdown();
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lifecycle_ = Lifecycle::FINALIZING;
    }
    shutdownImpl();
    std::vector<std::shared_ptr<storage_backend_detail::StorageTaskState>> released_tasks;
    uint64_t                                                               quarantine_generation;
    {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        RTP_LLM_CHECK(active_transfer_count_ == 0);
        released_tasks.swap(quarantined_tasks_);
        quarantined_block_count_ = 0;
        quarantine_active_.store(false, std::memory_order_release);
        quarantine_generation    = ++quarantine_generation_;
    }
    onQuarantineChanged(quarantine_generation, /*task_count=*/0, /*block_count=*/0);
    // Shutdown may return before unconfirmed DMA has stopped. Keep those
    // HOST and DEVICE allocations alive through process termination.
    static auto* retained_tasks = new std::vector<std::shared_ptr<storage_backend_detail::StorageTaskState>>;
    static auto* retained_mutex = new std::mutex;
    {
        std::lock_guard<std::mutex> lock(*retained_mutex);
        for (auto& task : released_tasks) {
            if (!task->sdk_can_drain) {
                retained_tasks->push_back(std::move(task));
            }
        }
    }
    // shutdownImpl() drains the transport before these pins become reusable.
    released_tasks.clear();
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::STOPPED;
    }
    lifecycle_cv_.notify_all();
}

void StorageBackend::quarantineTask(const std::shared_ptr<storage_backend_detail::StorageTaskState>& state,
                                    bool sdk_can_drain) {
    if (!state) {
        return;
    }
    size_t task_count;
    size_t block_count;
    uint64_t generation;
    {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        state->sdk_can_drain = state->sdk_can_drain && sdk_can_drain;
        if (state->quarantined) {
            return;
        }
        state->quarantined = true;
        quarantine_active_.store(true, std::memory_order_release);
        for (const auto& pin : state->pins) {
            if (pin.host_pool) {
                pin.host_pool->markUncertainRemoteIo();
            }
        }
        quarantined_block_count_ += state->pins.size();
        quarantined_tasks_.push_back(state);
        task_count  = quarantined_tasks_.size();
        block_count = quarantined_block_count_;
        generation  = ++quarantine_generation_;
    }
    RTP_LLM_LOG_WARNING("quarantine storage request after transfer completion became unknown: tasks=%zu blocks=%zu",
                        task_count,
                        block_count);
    onQuarantineChanged(generation, task_count, block_count);
}

bool StorageBackend::quarantineActive() const {
    return quarantine_active_.load(std::memory_order_acquire);
}

bool StorageBackend::tryBeginTransfer() {
    std::lock_guard<std::mutex> lock(quarantine_mutex_);
    if (quarantine_active_.load(std::memory_order_relaxed)) {
        return false;
    }
    ++active_transfer_count_;
    return true;
}

void StorageBackend::endTransfer() {
    std::lock_guard<std::mutex> lock(quarantine_mutex_);
    RTP_LLM_CHECK(active_transfer_count_ > 0);
    --active_transfer_count_;
}

std::shared_ptr<storage_backend_detail::StorageTaskState>
StorageBackend::prepare(StorageRequest request, bool allow_host, bool allocation_owner) {
    validateRequest(request, /*allow_null_blocks=*/false, allow_host);
    auto state     = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->owner   = this;
    state->request = std::move(request);
    RTP_LLM_CHECK(initialized_);

    // Avoid serializing the potentially large pinning loop. A second check
    // below closes the race with a transfer entering quarantine while pins
    // are acquired; tryBeginTransfer() is the final admission boundary.
    if (quarantineActive()) {
        return nullptr;
    }
    std::unordered_set<BlockKey, BlockKeyHash> pinned;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            if (state->request.source_tier == Tier::HOST) {
                const auto& pool = host_pools_by_tag_.at(handle.tag);
                if (pinned.insert({pool.get(), handle.block}).second) {
                    if (!allocation_owner) {
                        RTP_LLM_CHECK_WITH_INFO(pool->validBlock(handle.block), "invalid worker HOST cache block ID");
                        state->borrowed_pools.push_back(pool);
                        continue;
                    }
                    // Several tags can share one packed HOST block.
                    state->pins.push_back({nullptr, pool, handle.block});
                    try {
                        pool->incTreeRef(handle.block, BlockTreeRefType::STORE);
                    } catch (...) {
                        state->pins.pop_back();
                        throw;
                    }
                }
                continue;
            }
            {
                const auto& pool = devicePool(handle.tag);
                if (pinned.insert({pool.get(), handle.block}).second) {
                    if (allocation_owner) {
                        pool->incRef(handle.block);
                        state->pins.push_back({pool, nullptr, handle.block});
                    } else {
                        RTP_LLM_CHECK_WITH_INFO(pool->validBlock(handle.block), "invalid worker cache block ID");
                        state->borrowed_pools.push_back(pool);
                    }
                }
            }
        }
    }
    if (quarantineActive()) {
        state->finish();
        return nullptr;
    }
    return state;
}

bool StorageBackend::runTransfer(StorageRequest request,
                                  bool allocation_owner,
                                  const std::function<bool(StorageWriteTask)>& transfer) {
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (lifecycle_ != Lifecycle::ACCEPTING) {
            return false;
        }
        ++in_flight_;
    }
    struct Completion {
        StorageBackend* backend;
        bool transfer_started{false};
        ~Completion() {
            if (transfer_started) {
                backend->endTransfer();
            }
            backend->taskFinished();
        }
    } completion{this};
    auto state = prepare(std::move(request), /*allow_host=*/true, allocation_owner);
    if (!state || !tryBeginTransfer()) {
        return false;
    }
    completion.transfer_started = true;
    try {
        return transfer(StorageWriteTask(state));
    } catch (const StorageOperationCompletionUnknown& error) {
        quarantineTask(state, error.sdkCanDrain());
        throw;
    }
}

void StorageBackend::validateRequest(const StorageRequest& request, bool allow_null_blocks, bool allow_host) const {
    RTP_LLM_CHECK_WITH_INFO(request.source_tier == Tier::DEVICE || (allow_host && request.source_tier == Tier::HOST),
                            "unsupported storage source tier: %s",
                            tierName(request.source_tier));
    RTP_LLM_CHECK_WITH_INFO(request.keys != nullptr, "storage request requires cache keys");
    RTP_LLM_CHECK_WITH_INFO(request.handles.size() == request.keys->size(),
                            "storage request key/handle count mismatch: keys=%zu handles=%zu",
                            request.keys->size(),
                            request.handles.size());
    RTP_LLM_CHECK_WITH_INFO(request.host_write_source_paths.empty()
                                || (request.source_tier == Tier::HOST
                                    && request.host_write_source_paths.size() == request.keys->size()),
                            "HOST source-path count must match HOST remote key count");
    for (const HostWriteSourcePath& source_path : request.host_write_source_paths) {
        RTP_LLM_CHECK_WITH_INFO(!source_path.local_cache_keys.empty()
                                    && source_path.coordinate < source_path.local_cache_keys.size()
                                    && source_path.host_eviction_lease_id != 0,
                                "HOST source path has an invalid local coordinate or eviction lease");
    }
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        std::unordered_set<std::string> seen_tags;
        for (const auto& handle : request.handles[key_index]) {
            RTP_LLM_CHECK_WITH_INFO(!handle.tag.empty(), "storage handle has empty tag at key=%zu", key_index);
            (void)topology().group(handle.tag);
            if (request.source_tier == Tier::HOST) {
                RTP_LLM_CHECK_WITH_INFO(host_pools_by_tag_.count(handle.tag) != 0 && host_buffer_resolver_,
                                        "storage has no HOST source for tag=%s",
                                        handle.tag.c_str());
            }
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(handle.tag).second,
                                    "storage request has duplicate tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
            RTP_LLM_CHECK_WITH_INFO(allow_null_blocks || !isNullBlockIdx(handle.block),
                                    "storage request has null block for tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
        }
    }
}

const CacheTopology& StorageBackend::topology() const {
    RTP_LLM_CHECK(topology_ != nullptr);
    return *topology_;
}

const DeviceBlockPoolPtr& StorageBackend::devicePool(const std::string& tag) const {
    return pools_by_tag_.at(tag);
}

const std::shared_ptr<HostBlockPool>& StorageBackend::hostPool(const std::string& tag) const {
    static const std::shared_ptr<HostBlockPool> empty_pool;
    const auto                                 found = host_pools_by_tag_.find(tag);
    return found == host_pools_by_tag_.end() ? empty_pool : found->second;
}

const StorageBackend::HostWriteResolver& StorageBackend::hostWriteResolver() const {
    return host_write_resolver_;
}

const StorageBackend::HostWriteSettlementResolver& StorageBackend::hostWriteSettlementResolver() const {
    return host_write_settlement_resolver_;
}

std::vector<BlockInfo>
StorageBackend::convertIndexToBuffer(int layer_id, const std::string& tag, int block_id, Tier source_tier) const {
    (void)topology().group(tag);
    if (source_tier == Tier::HOST) {
        RTP_LLM_CHECK(host_pools_by_tag_.count(tag) != 0 && host_buffer_resolver_);
        return host_buffer_resolver_(layer_id, tag, block_id);
    }
    RTP_LLM_CHECK(source_tier == Tier::DEVICE && buffer_resolver_);
    return buffer_resolver_(layer_id, tag, block_id);
}

bool StorageBackend::isHandleRequired(size_t key_index, size_t matched_key_count, std::string_view tag) const {
    RTP_LLM_CHECK(key_index < matched_key_count);
    const size_t reuse_count = topology().group(tag).reuseBlockCount(matched_key_count);
    return matched_key_count - key_index <= reuse_count;
}

void StorageBackend::match(StorageRequest request, MatchDone done) {
    RTP_LLM_CHECK(initialized_);
    validateRequest(request, /*allow_null_blocks=*/true);
    dispatch([this, request = std::move(request), done = std::move(done)](Lifecycle outcome) mutable {
        StorageMatchResult result;
        bool               success = outcome == Lifecycle::ACCEPTING;
        if (success) {
            if (quarantineActive()) {
                // Unsubmitted requests can keep their local prefix and recompute.
                result.matched_blocks_num = request.local_matched_blocks_num;
            } else {
                try {
                    result = matchImpl(request);
                } catch (...) { success = false; }
            }
        }
        if (done) {
            done(success ? result.matched_blocks_num : 0, success ? std::move(result.match_meta) : nullptr, success);
        }
    });
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    auto state = prepare(std::move(request));
    if (!state) {
        dispatch([done = std::move(done)](Lifecycle) mutable {
            if (done) {
                done(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                               "storage backend is quarantined after unknown transfer completion"));
            }
        });
        return;
    }
    dispatch([this, state = std::move(state), match_meta = std::move(match_meta), done = std::move(done)](
                 Lifecycle outcome) mutable {
        ErrorInfo error = outcome == Lifecycle::ACCEPTING ?
                              ErrorInfo::OkStatus() :
                              ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "storage backend is not accepting reads");
        bool quarantine = false;
        bool sdk_can_drain = true;
        const bool transfer_started = error.ok() && tryBeginTransfer();
        if (error.ok() && !transfer_started) {
            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                              "storage backend is quarantined after unknown transfer completion");
        }
        if (error.ok()) {
            try {
                readImpl(state->request, match_meta);
            } catch (const StorageOperationTimeout& exception) {
                error      = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what(), /*completion_unknown=*/true);
                quarantine = true;
                sdk_can_drain = exception.sdkCanDrain();
            } catch (const StorageOperationCompletionUnknown& exception) {
                error      = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what(), /*completion_unknown=*/true);
                quarantine = true;
                sdk_can_drain = exception.sdkCanDrain();
            } catch (const std::exception& exception) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
            } catch (...) { error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown storage read failure"); }
        }
        if (quarantine) {
            quarantineTask(state, sdk_can_drain);
        }
        if (transfer_started) {
            endTransfer();
        }
        if (!quarantine) {
            state->finish();
        }
        if (done) {
            done(std::move(error));
        }
    });
}

StorageWriteTask StorageBackend::prepareWrite(StorageRequest request) {
    RTP_LLM_CHECK(initialized_);
    if (request.empty()) {
        return {};
    }
    return StorageWriteTask(prepare(std::move(request), /*allow_host=*/true));
}

bool StorageBackend::write(StorageWriteTask task, Done done) {
    RTP_LLM_CHECK(initialized_);
    RTP_LLM_CHECK(task.state_ != nullptr);
    auto state = std::move(task.state_);
    return dispatch([this, state, done = std::move(done)](Lifecycle outcome) mutable {
        ErrorInfo error = outcome == Lifecycle::ACCEPTING ?
                              ErrorInfo::OkStatus() :
                              ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "storage backend is not accepting writes");
        bool quarantine = false;
        bool sdk_can_drain = true;
        const bool transfer_started = error.ok() && tryBeginTransfer();
        if (error.ok() && !transfer_started) {
            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                              "storage backend is quarantined after unknown transfer completion");
        }
        if (error.ok()) {
            try {
                writeImpl(state->request);
            } catch (const StorageOperationTimeout& exception) {
                error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what(), /*completion_unknown=*/true);
                quarantine = true;
                sdk_can_drain = exception.sdkCanDrain();
            } catch (const StorageOperationCompletionUnknown& exception) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what(), /*completion_unknown=*/true);
                quarantine = true;
                sdk_can_drain = exception.sdkCanDrain();
            } catch (const std::exception& exception) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
            } catch (...) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown storage write failure");
            }
        }
        if (quarantine) {
            quarantineTask(state, sdk_can_drain);
        }
        if (transfer_started) {
            endTransfer();
        }
        if (!quarantine) {
            state->finish();
        }
        if (done) {
            done(std::move(error));
        }
    });
}

}  // namespace rtp_llm
