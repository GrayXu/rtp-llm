#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                               BlockTreeCacheConfig                           config,
                               std::shared_ptr<StorageBackend>                storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>             task_pool,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    metrics_reporter_(std::move(metrics_reporter)),
    evictor_(
        tree_.get(),
        config_.device_eviction_policy,
        config_.host_eviction_policy,
        config_.disk_eviction_policy,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        *metrics_reporter_,
        mutex_,
        config_.host_cache_sync_timeout_ms,
        config_.disk_cache_sync_timeout_ms,
        config_.max_descriptors_per_transfer_batch,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool tree_data_mutated, bool check_watermark) {
            onWorkflowSettledLocked(tree_data_mutated, check_watermark);
        },
        storage_backend_.get(),
        config_.remote_host_eviction_owner),
    loader_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.disk_cache_sync_timeout_ms,
            config_.host_cache_sync_timeout_ms,
            config_.enable_device_cache,
            storage_backend_,
            [this](bool tree_data_mutated, bool check_watermark) {
                onWorkflowSettledLocked(tree_data_mutated, check_watermark);
            }),
    storer_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.host_cache_sync_timeout_ms,
            config_.disk_cache_sync_timeout_ms,
            storage_backend_,
            [this](bool tree_data_mutated, bool check_watermark) {
                onWorkflowSettledLocked(tree_data_mutated, check_watermark);
            }) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.task_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     tree_->groupSets().size(),
                     tree_->reusableGroupCount(),
                     config_.task_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_host_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    if (config_.full_prefix_scan_interval_ms > 0) {
        auto scanner =
            std::make_unique<FullPrefixInvariantScanner>(*tree_, mutex_, config_.full_prefix_scan_interval_ms);
        if (scanner->start()) {
            full_prefix_scanner_ = std::move(scanner);
            RTP_LLM_LOG_INFO("FULL prefix invariant scanner enabled, interval_ms=%ld",
                             config_.full_prefix_scan_interval_ms);
        }
    }
    initialized_ = true;
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    full_prefix_scanner_.reset();
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    loader_.shutdown();
    if (storage_backend_) {
        storage_backend_->shutdown();
    }
    RTP_LLM_LOG_INFO("load tickets closed, stopping store admission and draining cache tasks...");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storer_.stopAdmissionLocked();
    }
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();
    task_pool_->waitForIdle();
    // Task-body idleness does not imply asynchronous RPC/transfer completion.
    // Keep dispatcher and cache state alive until transfer callbacks return,
    // then drain the settlement tasks those callbacks submitted.
    transfer_dispatcher_->drainTransfers();
    task_pool_->waitForIdle();
    transfer_dispatcher_.reset();
    task_pool_.reset();
    RTP_LLM_LOG_INFO("destroyed");
}

bool BlockTreeCache::executeTransfer(TransferTask task) {
    auto context = transfer_dispatcher_->executePerRank(std::move(task));
    context->waitDone();
    if (!context->success()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", context->errorInfo().ToString().c_str());
        return false;
    }
    return true;
}

StorageBackend::HostWriteResolution BlockTreeCache::resolveHostWrite(const CacheKeysType&            keys,
                                                                     const std::vector<std::string>& tags,
                                                                     const std::vector<uint32_t>&    coordinates,
                                                                     int                             timeout_ms,
                                                                     uint64_t                        host_eviction_lease_id) {
    if (!storage_backend_ || keys.empty() || tags.empty() || tags.size() != coordinates.size()) {
        return {};
    }
    std::unique_lock<std::mutex> lock(mutex_);
    bool wait_for_publication = false;
    std::vector<uint64_t> source_generations;
    auto resolve = [&]() -> std::vector<BlockIdxType> {
        wait_for_publication = false;
        source_generations.clear();
        const auto path = tree_->findNode(keys);
        if (path.size() < keys.size()) {
            wait_for_publication = host_eviction_lease_id == 0;
            return {};
        }
        std::vector<BlockIdxType> blocks;
        blocks.reserve(tags.size());
        source_generations.reserve(tags.size());
        for (size_t i = 0; i < tags.size(); ++i) {
            if (coordinates[i] >= path.size()) {
                return {};
            }
            const auto* location = tree_->reusableGroupLocation(tags[i]);
            if (!location) {
                return {};
            }
            const auto& host_pool = tree_->groupSets()[location->group_set_id]->hostPool();
            if (!host_pool || host_pool->hasUncertainRemoteIo()) {
                return {};
            }
            const auto& resource = path[coordinates[i]]->group_set_resources[location->group_set_id];
            const bool idle_source = resource.transfer_state == GroupSetTransferState::IDLE;
            const bool leased_source = host_eviction_lease_id != 0
                                       && resource.transfer_state == GroupSetTransferState::DEMOTING
                                       && resource.host_eviction_lease_id == host_eviction_lease_id
                                       && resource.hasTier(Tier::HOST);
            const bool idle_eviction_source = host_eviction_lease_id != 0 && idle_source
                                              && resource.hasTier(Tier::HOST);
            if (host_eviction_lease_id != 0 && !leased_source && !idle_eviction_source) {
                return {};
            }
            if (idle_source && !resource.hasTier(Tier::HOST)) {
                if (host_eviction_lease_id != 0) {
                    return {};
                }
                wait_for_publication = true;
                return {};
            }
            if ((!idle_source && !leased_source) || isNullBlockIdx(resource.host_block)
                || !host_pool->validBlock(resource.host_block)) {
                return {};
            }
            blocks.push_back(resource.host_block);
            source_generations.push_back(host_pool->blockAllocationGeneration(resource.host_block));
        }
        return blocks;
    };
    auto blocks = resolve();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, timeout_ms));
    while (blocks.empty() && wait_for_publication
           && host_publication_cv_.wait_until(lock, deadline) != std::cv_status::timeout) {
        blocks = resolve();
    }
    if (blocks.empty()) {
        return {};
    }
    std::vector<std::pair<TreeNode*, size_t>> newly_reserved_sources;
    const auto rollback_newly_reserved = [&]() {
        bool restored = false;
        for (const auto& [node, group_set_id] : newly_reserved_sources) {
            (void)evictor_.settleRemoteHostWriteLocked(node,
                                                       group_set_id,
                                                       host_eviction_lease_id,
                                                       HostWriteSettlementOutcome::ROLLBACK,
                                                       restored);
        }
        if (restored) {
            onWorkflowSettledLocked(true, true);
        }
    };
    if (host_eviction_lease_id != 0) {
        const auto path = tree_->findNode(keys);
        for (size_t i = 0; i < tags.size(); ++i) {
            const ReusableGroupLocation* location = tree_->reusableGroupLocation(tags[i]);
            if (location == nullptr) {
                rollback_newly_reserved();
                return {};
            }
            TreeNode* const source_node = path[coordinates[i]];
            const auto existing = std::find_if(newly_reserved_sources.begin(),
                                               newly_reserved_sources.end(),
                                               [&](const auto& entry) {
                                                   return entry.first == source_node
                                                          && entry.second == location->group_set_id;
                                               });
            if (existing != newly_reserved_sources.end()) {
                continue;
            }
            bool newly_reserved = false;
            if (!evictor_.reserveRemoteHostWriteLocked(
                    source_node,
                    location->group_set_id,
                    host_eviction_lease_id,
                    newly_reserved)) {
                rollback_newly_reserved();
                return {};
            }
            if (newly_reserved) {
                newly_reserved_sources.emplace_back(source_node, location->group_set_id);
            }
        }
        if (!newly_reserved_sources.empty()) {
            onWorkflowSettledLocked(true, false);
        }
    }
    auto key_copy = std::make_shared<CacheKeysType>();
    StorageRequest request{key_copy, {}};
    request.source_tier = Tier::HOST;
    key_copy->reserve(tags.size());
    request.handles.reserve(tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        key_copy->push_back(keys[coordinates[i]]);
        request.handles.push_back({StorageBlockHandle{tags[i], blocks[i]}});
    }
    try {
        auto pins = storage_backend_->prepareWrite(std::move(request));
        if (pins) {
            for (size_t i = 0; i < tags.size(); ++i) {
                const ReusableGroupLocation* location = tree_->reusableGroupLocation(tags[i]);
                if (location == nullptr) {
                    pins = StorageWriteTask{};
                    rollback_newly_reserved();
                    return {};
                }
                const auto& host_pool = tree_->groupSets()[location->group_set_id]->hostPool();
                if (!host_pool || !host_pool->validBlock(blocks[i])
                    || host_pool->blockAllocationGeneration(blocks[i]) != source_generations[i]) {
                    pins = StorageWriteTask{};
                    rollback_newly_reserved();
                    return {};
                }
            }
        }
        if (!pins && !newly_reserved_sources.empty()) {
            rollback_newly_reserved();
        }
        return {std::move(pins), std::move(blocks)};
    } catch (...) {
        rollback_newly_reserved();
        throw;
    }
}

bool BlockTreeCache::settleRemoteHostWriteSource(const CacheKeysType&             local_cache_keys,
                                                 const std::vector<std::string>& group_tags,
                                                 uint32_t                        coordinate,
                                                 uint64_t                        host_eviction_lease_id,
                                                 HostWriteSettlementOutcome     outcome) {
    if (local_cache_keys.empty() || group_tags.empty() || host_eviction_lease_id == 0) {
        return false;
    }
    if (outcome == HostWriteSettlementOutcome::RESERVE) {
        auto source = resolveHostWrite(local_cache_keys,
                                       group_tags,
                                       std::vector<uint32_t>(group_tags.size(), coordinate),
                                       /*timeout_ms=*/0,
                                       host_eviction_lease_id);
        return static_cast<bool>(source.pins);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  path = tree_->findNode(local_cache_keys);
    if (path.size() < local_cache_keys.size() || coordinate >= path.size()) {
        return true;
    }
    std::optional<size_t> group_set_id;
    for (const std::string& tag : group_tags) {
        const ReusableGroupLocation* location = tree_->reusableGroupLocation(tag);
        if (!location || (group_set_id && *group_set_id != location->group_set_id)) {
            return false;
        }
        group_set_id = location->group_set_id;
    }
    bool tree_data_mutated = false;
    if (!evictor_.settleRemoteHostWriteLocked(
            path[coordinate],
            *group_set_id,
            host_eviction_lease_id,
            outcome,
            tree_data_mutated)) {
        return false;
    }
    if (tree_data_mutated) {
        onWorkflowSettledLocked(true, outcome != HostWriteSettlementOutcome::UNKNOWN);
    }
    return true;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType&                 cache_keys,
                                           std::shared_ptr<const CacheKeysType> remote_keys) {
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result = loader_.matchLocked(cache_keys, std::move(remote_keys));
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier,
                            std::shared_ptr<const StorageRequest>             remote_write,
                            const std::vector<CacheKeysType>&                 remote_key_groups) {
    (void)insert(cache_keys,
                 resources,
                 target_tier,
                 false,
                 false,
                 std::move(remote_write),
                 remote_key_groups);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident,
                              bool                                              write_remote_from_device,
                              std::shared_ptr<const StorageRequest>             remote_write,
                              const std::vector<CacheKeysType>&                 remote_key_groups) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storage_write = storer_.storeLocked(cache_keys,
                                            resources,
                                            target_tier,
                                            is_resident,
                                            resident_prefix_length,
                                            write_remote_from_device,
                                            std::move(remote_write),
                                            remote_key_groups);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

int BlockTreeCache::evictForGroup(std::string_view group_tag, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return 0;
    }
    const GroupSetPtr& group_set   = tree_->groupSets()[location->group_set_id];
    const auto&        device_pool = group_set->devicePools()[location->member_group_id];

    const size_t initial_free       = device_pool->freeBlocksNum();
    size_t       reclaimed          = 0;
    bool         eviction_triggered = false;
    while (reclaimed < num_blocks) {
        if (!evictor_.dropLocked(location->group_set_id, Tier::DEVICE, true)) {
            break;
        }
        eviction_triggered        = true;
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    if (eviction_triggered) {
        metrics_reporter_->reportEvictionTriggered(Tier::DEVICE, group_set->groupType(), /*force_drop=*/true);
    }
    metrics_reporter_->reportEvictionBlocks(Tier::DEVICE,
                                            group_set->groupType(),
                                            /*force_drop=*/true,
                                            num_blocks,
                                            std::min(reclaimed, num_blocks));
    RTP_LLM_LOG_DEBUG("tag=%.*s group_set[%zu] reclaimed %zu/%zu device blocks",
                      static_cast<int>(group_tag.size()),
                      group_tag.data(),
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(std::string_view                      group_tag,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_tag, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<BlockTreePoolMetricsSnapshot> BlockTreeCache::poolMetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_reporter_->collectPoolMetricsSnapshots(tree_->groupSets());
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_->collectEvictableMetricsSnapshots(tree_->groupSets(), evictor_);
    }
    metrics_reporter_->reportEvictableCandidateCount(snapshots);
    metrics_reporter_->reportQueueBacklog(task_pool_->queueSizes(), "business");
    metrics_reporter_->reportQueueBacklog(transfer_dispatcher_->queueSizes(), "transfer");
}

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                       const std::vector<std::string>& required_group_tags) {
    std::lock_guard<std::mutex> lock(mutex_);
    tree_->setEventPublisher(std::move(publisher), required_group_tags);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->size());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty()) {
        const TreeNode* node = pending.back();
        pending.pop_back();
        const bool reusable = std::any_of(node->group_set_resources.begin(),
                                          node->group_set_resources.end(),
                                          [](const GroupSetResource& resource) { return !resource.is_empty(); });
        if (reusable) {
            snapshot.keys.push_back(node->cache_key);
        }
        for (const auto& [cache_key, child] : node->children) {
            (void)cache_key;
            if (child) {
                pending.push_back(child);
            }
        }
    }
    return snapshot;
}

bool BlockTreeCache::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return loader_.abortPendingLoad(context);
}

void BlockTreeCache::onWorkflowSettledLocked(bool tree_data_mutated, bool check_watermark) {
    if (tree_data_mutated) {
        ++mutation_version_;
        host_publication_cv_.notify_all();
    }
    if (check_watermark) {
        checkWatermark();
    }
}

void BlockTreeCache::checkWatermark() {
    for (Tier tier : {Tier::DISK, Tier::HOST, Tier::DEVICE}) {
        const auto watermark = config_.watermarkForTier(tier);
        if (!config_.isTierEnabled(tier) || !watermark.enabled()) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark);
    }
}

}  // namespace rtp_llm
