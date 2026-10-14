#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

struct TreeNode {
    CacheKeyType                                cache_key{0};
    size_t                                      index{0};
    std::vector<int>                            token_ids;
    std::unordered_map<CacheKeyType, TreeNode*> children;
    TreeNode*                                   parent{nullptr};
    std::vector<GroupSetResource>               group_set_resources;
    // Original full CP keys for this local canonical node, when remote storage
    // is enabled. Empty means the node key is already the remote key.
    CacheKeysType                               remote_key_group;
    bool                                        is_resident{false};  // Must never change from true to false.
};

}  // namespace rtp_llm
