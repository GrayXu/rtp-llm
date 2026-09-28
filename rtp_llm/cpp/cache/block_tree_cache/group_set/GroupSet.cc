#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

std::vector<BlockIdList> collectDeviceBlocks(const MultiNodeResource& resource, size_t pool_count) {
    std::vector<BlockIdList> blocks_by_pool(pool_count);
    for (const auto& [_, blocks] : resource.node_blocks) {
        for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
            blocks_by_pool[pool_index].push_back(blocks[pool_index]);
        }
    }
    return blocks_by_pool;
}

BlockIdList collectBlocks(const MultiNodeResource& resource) {
    BlockIdList result;
    for (const auto& [_, blocks] : resource.node_blocks) {
        result.insert(result.end(), blocks.begin(), blocks.end());
    }
    return result;
}

template<typename Operation>
void applyToResourcePools(const MultiNodeResource&               resource,
                          const std::vector<DeviceBlockPoolPtr>& device_pools,
                          const std::shared_ptr<HostBlockPool>&  host_pool,
                          const BlockTreeDiskBlockPoolPtr&       disk_pool,
                          Operation&&                            operation) {
    if (resource.tier == Tier::DEVICE) {
        const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools.size());
        for (size_t pool_index = 0; pool_index < device_pools.size(); ++pool_index) {
            operation(*device_pools[pool_index], blocks_by_pool[pool_index]);
        }
    } else if (resource.tier == Tier::HOST && host_pool) {
        operation(*host_pool, collectBlocks(resource));
    } else if (resource.tier == Tier::DISK && disk_pool) {
        operation(*disk_pool, collectBlocks(resource));
    }
}

}  // namespace

GroupSet::GroupSet(std::vector<DeviceBlockPoolPtr> device_pools,
                   std::shared_ptr<HostBlockPool>  host_pool,
                   BlockTreeDiskBlockPoolPtr       disk_pool):
    device_pools_(std::move(device_pools)), host_pool_(std::move(host_pool)), disk_pool_(std::move(disk_pool)) {}

void GroupSet::initialize(size_t                               group_set_id,
                          std::shared_ptr<const CacheTopology> topology,
                          std::vector<std::string>             group_tags,
                          size_t                               physical_payload_bytes) {
    RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "GroupSet requires a topology");
    RTP_LLM_CHECK_WITH_INFO(!group_tags.empty(), "GroupSet requires at least one member tag");
    RTP_LLM_CHECK_WITH_INFO(device_pools_.empty() || device_pools_.size() == group_tags.size(),
                            "GroupSet tag/pool count mismatch: tags=%zu pools=%zu",
                            group_tags.size(),
                            device_pools_.size());
    std::unordered_set<std::string> seen_tags;
    for (size_t member_index = 0; member_index < group_tags.size(); ++member_index) {
        const auto& tag = group_tags[member_index];
        RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second, "GroupSet has duplicate member tag=%s", tag.c_str());
        (void)topology->group(tag);
        if (!device_pools_.empty()) {
            RTP_LLM_CHECK_WITH_INFO(
                device_pools_[member_index] != nullptr, "GroupSet has null device pool for tag=%s", tag.c_str());
        }
    }
    uses_physical_payload_geometry_ = physical_payload_bytes > 0;
    if (physical_payload_bytes == 0) {
        for (const auto& tag : group_tags) {
            const size_t group_bytes = topology->blockSizeBytesForGroup(tag);
            RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - physical_payload_bytes,
                                    "GroupSet payload size overflow at tag=%s",
                                    tag.c_str());
            physical_payload_bytes += group_bytes;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(physical_payload_bytes > 0, "GroupSet physical payload must be positive");

    group_set_id_  = group_set_id;
    topology_      = std::move(topology);
    group_tags_    = std::move(group_tags);
    payload_bytes_ = physical_payload_bytes;
    std::lock_guard<std::mutex> lock(host_layout_mutex_);
    host_buffer_layouts_.clear();
}

bool GroupSet::hasAllocatedDeviceBlocks(const std::vector<BlockIdxType>& blocks) const {
    if (blocks.size() != device_pools_.size()) {
        return false;
    }
    for (size_t pool_index = 0; pool_index < blocks.size(); ++pool_index) {
        if (!device_pools_[pool_index]->isAllocated(blocks[pool_index])) {
            return false;
        }
    }
    return true;
}

std::vector<BlockInfo>
GroupSet::convertHostIndexToBuffer(int layer_id, const std::string& tag, BlockIdxType block) const {
    RTP_LLM_CHECK(host_pool_ != nullptr && device_pools_.size() == group_tags_.size());
    const auto host = host_pool_->blockBuffer(block);
    RTP_LLM_CHECK_WITH_INFO(host.payload_bytes == payload_bytes_, "HOST payload size mismatch for tag=%s", tag.c_str());
    std::lock_guard<std::mutex> lock(host_layout_mutex_);
    if (host_buffer_layouts_.empty()) {
        HostBufferLayouts layouts;
        size_t            host_offset = 0;
        for (size_t member = 0; member < group_tags_.size(); ++member) {
            const auto& member_tag   = group_tags_[member];
            const auto& member_group = group(member_tag);
            const auto  layers       = topology_->layerIdsForGroup(member_tag);
            for (size_t local_layer = 0; local_layer < layers.size(); ++local_layer) {
                // Resolve layout metadata once; never read or retain DEVICE addresses.
                auto buffers = device_pools_[member]->convertIndexToBuffer(static_cast<int>(local_layer), 0);
                if (!uses_physical_payload_geometry_) {
                    const size_t scale_bytes = member_group.kvScaleStrideBytes();
                    RTP_LLM_CHECK(buffers.size() >= (scale_bytes == 0 ? 1u : 2u));
                    buffers[0].size_bytes = member_group.kvBlockStrideBytes();
                    if (scale_bytes != 0) {
                        buffers[1].size_bytes = scale_bytes;
                    }
                    buffers.resize(scale_bytes == 0 ? 1 : 2);
                }
                auto& regions = layouts[member_tag][layers[local_layer]];
                for (auto buffer : buffers) {
                    RTP_LLM_CHECK_WITH_INFO(host_offset <= host.payload_bytes
                                                && buffer.size_bytes <= host.payload_bytes - host_offset,
                                            "HOST payload offset exceeds block size for tag=%s",
                                            member_tag.c_str());
                    buffer.is_cuda      = false;
                    buffer.device_index = 0;
                    buffer.addr         = nullptr;
                    regions.push_back({host_offset, buffer});
                    host_offset += buffer.size_bytes;
                }
            }
        }
        RTP_LLM_CHECK_WITH_INFO(host_offset == host.payload_bytes, "HOST payload layout size mismatch");
        host_buffer_layouts_ = std::move(layouts);
    }
    const auto tag_layout = host_buffer_layouts_.find(tag);
    RTP_LLM_CHECK_WITH_INFO(tag_layout != host_buffer_layouts_.end(), "unknown HOST storage tag=%s", tag.c_str());
    const auto layer_layout = tag_layout->second.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(layer_layout != tag_layout->second.end(),
                            "layer_id=%d does not belong to HOST storage tag=%s",
                            layer_id,
                            tag.c_str());
    std::vector<BlockInfo> buffers;
    buffers.reserve(layer_layout->second.size());
    for (const auto& region : layer_layout->second) {
        buffers.push_back(region.info);
        buffers.back().addr = static_cast<uint8_t*>(host.addr) + region.offset;
    }
    return buffers;
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
    for (size_t pool_index = 0; pool_index < device_pools_.size(); ++pool_index) {
        device_pools_[pool_index]->incRef(blocks_by_pool[pool_index]);
    }
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource) const {
    if (resource.tier != Tier::DEVICE) {
        return;
    }
    const auto blocks_by_pool = collectDeviceBlocks(resource, device_pools_.size());
    for (size_t pool_index = 0; pool_index < device_pools_.size(); ++pool_index) {
        device_pools_[pool_index]->decRef(blocks_by_pool[pool_index]);
    }
}

void GroupSet::referenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, device_pools_, host_pool_, disk_pool_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.incTreeRef(blocks, ref_type);
        });
}

void GroupSet::unreferenceBlocks(const MultiNodeResource& resource, BlockTreeRefType ref_type) const {
    applyToResourcePools(
        resource, device_pools_, host_pool_, disk_pool_, [ref_type](IBlockPool& pool, const BlockIdList& blocks) {
            pool.decTreeRef(blocks, ref_type);
        });
}

BlockIdxType GroupSet::allocateSingleBlock(Tier tier, BlockTreeRefType ref_type) {
    auto blocks = allocateBlocks(1, tier, ref_type);
    return blocks.has_value() ? blocks->front() : NULL_BLOCK_IDX;
}

std::optional<BlockIdList> GroupSet::allocateBlocks(size_t n, Tier tier, BlockTreeRefType ref_type) {
    IBlockPool* pool = nullptr;
    if (tier == Tier::HOST) {
        pool = host_pool_.get();
    } else if (tier == Tier::DISK) {
        pool = disk_pool_.get();
    }
    if (!pool)
        return std::nullopt;
    auto blocks = pool->malloc(n);
    if (!blocks.has_value())
        return std::nullopt;
    pool->incTreeRef(*blocks, ref_type);
    return blocks;
}

void GroupSet::releaseBlocks(Tier tier, const BlockIdList& blocks, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decTreeRef(blocks, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decTreeRef(blocks, ref_type);
    }
}

void GroupSet::releaseSingleBlock(Tier tier, BlockIdxType block, BlockTreeRefType ref_type) const {
    if (tier == Tier::HOST) {
        if (host_pool_)
            host_pool_->decTreeRef(block, ref_type);
    } else if (tier == Tier::DISK) {
        if (disk_pool_)
            disk_pool_->decTreeRef(block, ref_type);
    }
}

}  // namespace rtp_llm
