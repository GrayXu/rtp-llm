#pragma once

#include <unordered_map>
#include <atomic>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

struct StorageBlockHandle {
    std::string  tag;
    BlockIdxType block{NULL_BLOCK_IDX};
    Tier         tier{Tier::DEVICE};
};

struct StorageBackendMatchMeta {
    virtual ~StorageBackendMatchMeta() = default;
};

struct StorageMatchResult {
    size_t                                   matched_blocks_num{0};
    std::shared_ptr<StorageBackendMatchMeta> match_meta;
};

struct StorageRequest {
    std::shared_ptr<const CacheKeysType> keys;
    // handles[i] contains every group block associated with (*keys)[i].
    std::vector<std::vector<StorageBlockHandle>> handles;
    // Match requests expose the complete key sequence. Keys before this
    // boundary are already available from Device/Host/Disk.
    size_t local_matched_blocks_num{0};
    // Write sources may be DEVICE or HOST. Match/read requests stay on DEVICE.
    Tier source_tier{Tier::DEVICE};
    // Shared HOST writes retain pins while local payload completion is unknown.
    std::shared_ptr<std::atomic<bool>> host_payload_dispatched;
    // Complete original fixed-token-block keys at the remote CP boundary.
    // Local tree keys may each represent one full CP round.
    std::shared_ptr<const CacheKeysType> remote_keys;
    // Explicit global-key write assembled before CP local-tree projection.
    bool keys_are_global{false};

    bool empty() const {
        for (const auto& key_handles : handles) {
            if (!key_handles.empty()) {
                return false;
            }
        }
        return true;
    }
};

namespace storage_backend_detail {
struct StorageTaskState;
}  // namespace storage_backend_detail

class StorageWriteTask {
public:
    StorageWriteTask()                            = default;
    StorageWriteTask(StorageWriteTask&&) noexcept = default;
    StorageWriteTask& operator=(StorageWriteTask&&) noexcept = default;

    StorageWriteTask(const StorageWriteTask&) = delete;
    StorageWriteTask& operator=(const StorageWriteTask&) = delete;

    explicit operator bool() const {
        return state_ != nullptr;
    }
    void quarantine();

private:
    explicit StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state);
    std::shared_ptr<storage_backend_detail::StorageTaskState> state_;
    friend class StorageBackend;
};

// Asynchronous facade for synchronous derived I/O. Owners must call shutdown
// before derived backend state starts destruction.
class StorageBackend {
public:
    using MatchDone      = std::function<void(
        size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success)>;
    using Done           = std::function<void(bool success)>;
    using PoolsByTag     = std::unordered_map<std::string, DeviceBlockPoolPtr>;
    using HostPoolsByTag = std::unordered_map<std::string, std::shared_ptr<HostBlockPool>>;
    using BufferResolver = std::function<std::vector<BlockInfo>(int layer_id, const std::string& tag, int block_id)>;
    struct HostPoolBinding {
        std::shared_ptr<HostBlockPool> pool;
        std::shared_ptr<HostBlockPool> read_pool;
        size_t                         group_set_id{0};
        size_t                         member_index{0};
        size_t                         member_count{0};
        size_t                         offset_bytes{0};
        size_t                         payload_bytes{0};
    };
    using HostBindingsByTag = std::unordered_map<std::string, HostPoolBinding>;
    using HostToDevice = std::function<bool(std::vector<TransferDescriptor>, std::vector<HostBufferView>, int timeout_ms)>;
    struct HostWriteResolution {
        StorageWriteTask             pins;
        std::vector<BlockIdxType>    local_blocks;
    };
    using HostWriteResolver = std::function<HostWriteResolution(const CacheKeysType&,
                                                                 const std::vector<std::string>&,
                                                                 const std::vector<uint32_t>&,
                                                                 int timeout_ms)>;

    // An injected executor may be observed by its owner but belongs to only
    // one backend; init rejects binding the same instance a second time.
    explicit StorageBackend(std::shared_ptr<StorageBackendExecutor> executor = nullptr);
    virtual ~StorageBackend();

    // Initialization is single-attempt. A failed start may permanently stop
    // an injected executor; create a fresh backend/executor to retry.
    bool init(std::shared_ptr<const CacheTopology> topology,
              PoolsByTag pools_by_tag,
              BufferResolver buffer_resolver,
              HostPoolsByTag host_pools_by_tag = {},
              BufferResolver host_buffer_resolver = {},
              HostBindingsByTag host_bindings_by_tag = {},
              HostToDevice host_to_device = {},
              HostWriteResolver host_write_resolver = {});
    virtual bool requiresSharedHostMemory() const { return false; }
    virtual bool canInitiateHostWrite() const { return false; }
    void match(StorageRequest request, MatchDone done);
    void read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done);
    StorageWriteTask prepareWrite(StorageRequest request);
    // Returns admission only, not the I/O result; never waits for completion.
    // Source pins live until execution or rejection.
    bool write(StorageWriteTask task);
    // Must not be called from backend I/O or completion callbacks.
    void shutdown();

protected:
    // The resolver must own its backing state when retained by asynchronous I/O.
    BufferResolver bufferResolver(Tier source = Tier::DEVICE) const {
        return source == Tier::HOST ? host_buffer_resolver_ : buffer_resolver_;
    }
    const PoolsByTag& devicePools() const { return pools_by_tag_; }
    const HostPoolsByTag& hostPools() const { return host_pools_by_tag_; }
    const CacheTopology&      topology() const;
    const DeviceBlockPoolPtr& devicePool(const std::string& tag) const;
    const std::shared_ptr<HostBlockPool>& hostPool(const std::string& tag) const;
    const HostBindingsByTag& hostPoolsByTag() const;
    const HostToDevice& hostToDevice() const;
    const HostWriteResolver& hostWriteResolver() const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, const std::string& tag, int block_id, Tier source_tier = Tier::DEVICE) const;
    // Match queries contain every possible group handle. Derived matchers use
    // this predicate for each candidate prefix; the core applies the same rule
    // before allocating read targets.
    bool isHandleRequired(size_t key_index, size_t matched_key_count, std::string_view tag) const;
    // RPC execution participates in shutdown/drain. Worker ranks borrow the
    // registered pool span; TP0 additionally holds allocator references.
    bool runTransfer(StorageRequest request,
                     bool allocation_owner,
                     const std::function<bool(StorageWriteTask)>& transfer);

    virtual bool               initImpl()                                                           = 0;
    virtual StorageMatchResult matchImpl(const StorageRequest& request)                             = 0;
    virtual void               readImpl(const StorageRequest&                           request,
                                        const std::shared_ptr<StorageBackendMatchMeta>& match_meta) = 0;
    virtual void               writeImpl(const StorageRequest& request)                             = 0;
    virtual void               shutdownImpl() noexcept {}

private:
    enum class Lifecycle {
        CREATED,
        ACCEPTING,
        STOPPING,
        FINALIZING,
        STOPPED
    };
    using Operation = std::function<void(Lifecycle outcome)>;

    std::shared_ptr<storage_backend_detail::StorageTaskState>
         prepare(StorageRequest request, bool allow_host = false, bool allocation_owner = true);
    void validateRequest(const StorageRequest& request, bool allow_null_blocks, bool allow_host = false) const;
    bool                                 dispatch(Operation operation);
    void                                 taskFinished();
    std::shared_ptr<const CacheTopology> topology_;
    PoolsByTag                           pools_by_tag_;
    HostBindingsByTag                    host_bindings_by_tag_;
    HostToDevice                          host_to_device_;
    HostWriteResolver                     host_write_resolver_;
    BufferResolver                       buffer_resolver_;
    HostPoolsByTag                          host_pools_by_tag_;
    BufferResolver                          host_buffer_resolver_;
    std::shared_ptr<StorageBackendExecutor> executor_;
    bool                                    init_attempted_{false};
    bool                                    initialized_{false};

    std::mutex              lifecycle_mutex_;
    std::condition_variable lifecycle_cv_;
    Lifecycle               lifecycle_{Lifecycle::CREATED};
    size_t                  in_flight_{0};

    friend class LoadAsyncContext;
};

}  // namespace rtp_llm
