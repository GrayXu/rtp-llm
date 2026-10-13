### ReuseCache
In multi-turn conversation scenarios, multiple prompts often share common prefixes. The KV cache corresponding to these prefix tokens is identical, and reusing KV cache can reduce computation time for these repeated parts, lowering First Token Latency. Enable KV cache reuse by setting the environment variable `REUSE_CACHE=1`. The startup logs will show "reuse_cache: True" when enabled.
The environment variable `SEQ_SIZE_PER_BLOCK` specifies the number of sequences corresponding to each KV cache block.
**Note: ReuseCache cannot currently use flash attention due to mismatched lengths between Q and KV, requiring `--reuse_cache true` to be added in the CMD**

``` python
import os
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# This query can reuse the KV cache corresponding to the previous query.
# The 'reuse len' in the response's aux info indicates how much KV cache was reused. The reuse length is always an integer multiple of SEQ_SIZE_PER_BLOCK.
for res in pipeline("hello, what's your name? how old are you?", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```

## Cache Tiers (L1 / L2 / L3)

Reused KV cache can live in three local tiers, each controlled by its own independent switch:

| Tier | Location | Switch | Required companion settings |
|------|----------|--------|-----------------------------|
| L1 | GPU device memory | `ENABLE_DEVICE_CACHE` (default on) | — |
| L2 | Pinned host memory | `ENABLE_MEMORY_CACHE` (default off) | `MEMORY_CACHE_SIZE_MB` |
| L3 | Local disk | `ENABLE_DISK_CACHE` (default off) | `DISK_CACHE_SIZE_MB`, `DISK_CACHE_PATHS` |

Device free-block headroom is managed by `BLOCK_TREE_DEVICE_EVICT_HIGH_WATERMARK_RATIO` (default 0.90)
and `BLOCK_TREE_DEVICE_EVICT_LOW_WATERMARK_RATIO` (default 0.82). Reaching the high occupancy watermark
triggers eviction toward the low watermark. Allocations can consume this headroom and reclaim cached
blocks on demand; blocks still referenced by requests or transfers may delay physical reclamation.
Scheduler admission separately uses `RESERVE_BLOCK_RATIO` (default 5%) to preserve headroom for
running requests to grow. This admission reserve is independent of the cache eviction watermarks.

All eight combinations are valid, including L2-only and L3-only deployments. Enabling a tier
without its capacity or path settings is a startup error rather than a silent downgrade, so a
misconfiguration never degrades quietly into a smaller cache than intended.

When MULTI_TASK_PROMPT is configured, the server automatically enables `REUSE_CACHE` and L1 to
preserve the existing static system-prompt behavior. Disable MULTI_TASK_PROMPT when testing a pure
L2-only or L3-only configuration.

`REUSE_CACHE` remains the deployment master switch: with it off, no tier is consulted or written.
By default, a request must also set `reuse_cache=true` (the default) to use prefix-cache lookup
or store its KV for later reuse.

### Lookup and store targets

The deployment switches select the local tiers used for lookup and storage. The per-request
`enable_device_cache`, `enable_memory_cache` and `enable_disk_cache` fields remain accepted for
protocol compatibility, but do not change this local-tier policy.

- **Lookup** may use any enabled local tier, plus the configured remote backend.
- **Storage after successful completion** selects one local target: L1 (DEVICE), then L2
  (HOST), then L3 (DISK). Lower-tier demotion is a separate operation.

With cache reuse enabled, the local-tier rules are:

| Enabled local tiers (deployment) | Lookup may use | Request-completion store target |
|---------------------------------|----------------|---------------------------------|
| None | None locally | None |
| L1 | L1 | L1 |
| L2 | L2 | L2 |
| L3 | L3 | L3 |
| L1, L2 | L1, L2 | L1 |
| L1, L3 | L1, L3 | L1 |
| L2, L3 | L2, L3 | L2 |
| L1, L2, L3 | L1, L2, L3 | L1 |

Remote caching requires a backend enabled and initialized by the deployment configuration.
Every DEVICE insert also prepares a remote upload when that backend exists, including
resident inserts and duplicate keys. HOST/DISK inserts do not upload. Remote writes are
asynchronous, submitted outside the tree lock, and do not delay insert until I/O completes;
a failed upload does not roll back the DEVICE cache entry.

The request's `enable_remote_cache` field remains accepted but does not control this DEVICE
upload path or remote lookup admission. A remote-only deployment can still look up remote
entries, but has no request-completion store target. Direct REMOTE inserts are unsupported.
Backend availability does not bypass successful completion, `reuse_cache`, or allocator
resource eligibility checks required to reach a DEVICE insert.

### KVCM remote cache

KVCM remote caching requires compatible SDK/Manager dependencies and a build with remote-cache support.
Enable it with `--enable_remote_cache 1` and configure the Manager with `--kvcm_server_address`.
For PACE, set `--kvcm_model_sdk_config '[{"type":"pace"}]'` (`RECO_MODEL_SDK_CONFIG`).

The following arguments have equivalent upper-case environment variables:

| Argument | Default | Meaning |
|---|---:|---|
| `--kvcm_remote_layout` | `legacy` | Rank-local storage; `canonical_v1` supports compatible TP/CP layouts |
| `--kvcm_read_backend_type` | 0 | 0=regular query; 1=3fs, 2=mooncake, 3=PACE, 4=NFS, 5=VCNS 3fs |
| `--kvcm_min_replica_count` | 0 | Minimum readable replicas for StartWrite; the server treats 0 as 1 |

Processes exchanging `KVCacheConfig` must use the same build version. The config uses pickle
v9 and can read v1–v8. See [KV cache event publisher](kv_cache_event_publisher.md) for HBM reporting.

#### GPU dual writes and explicit HOST writes

Explicit inserts of ready HOST content use ordinary CPU IOVs and the transfer client for the
original business tag. Each rank resolves HOST blocks by its local cache key/coordinate and
holds STORE references during I/O. This path requires neither HOST fd registration nor a
dedicated HOST client.

For sharded CP, explicit HOST inserts use the existing `remote_write` parameter. The
`StorageRequest` supplies `source_tier=HOST`, local round endpoint keys/handles, and the original
global fixed-block keys in `remote_keys`. It uploads complete rounds stored in the local HOST
tree. Workers use the same `CPSlotMapper` to project global keys/coordinates into local rounds;
the original keys cannot be recovered from endpoint hashes. A TP replica with no write shard
completes as an empty task.

Remote reads always target GPU IOVs. SDK/PACE selects direct, landing or staging; RTP does not
allocate a separate HOST read pool. The first transfer client registers all required GPU spans
before the first PACE INIT, and subsequent clients borrow those registrations. Shutdown drains
borrowers before closing the registration owner.

DEVICE and explicit HOST STORE references follow the storage task lifetime. On SDK timeout or
unknown remote completion, affected references remain held until process termination. SDK task
return and destruction do not prove physical I/O completion. An unknown read completion
terminates that request to avoid recomputing into the same target blocks; determinate failures
retain the existing Prefill fallback. After quarantine, requests that have not submitted I/O
keep local hits and recompute the missing portion.

With `PACE_ENABLE_GATHERSCATTER=ON`, compatible equal-size IOVs within each block retain the
accelerated path. Heterogeneous IOV sizes use the SDK's order-preserving Normal path, keeping
the same remote byte order for CPU and GPU.

#### Multiple pools and canonical layouts

Remote keys across cache groups use a common token stride. Each location must contain every
spec in the selected groups and a nonempty URI. Each pool's payload must match its declared
byte size. Generated legacy Instance IDs include the heterogeneous physical layout. Custom
configurations must match token-block size, specs and groups; spec order within a group does
not affect matching.

Set `--kvcm_remote_layout canonical_v1` on both producer and consumer to reuse KV across
compatible TP/CP partitions. Canonical Instance IDs use the `canonical_v1_` namespace, derived
from the logical KV layout and model identity with deployment TP/CP partitioning normalized.
Legacy and canonical payloads require separate Instances. Change `KVCM_INSTANCE_ID_SALT`
when weights or other cache semantics change.

Canonical supports typed MHA/GQA/MQA, MLA FULL, and typed FULL+LINEAR state. TP partitions
logical heads. Sharded CP places global block `i` on rank `i % CP` at local slot `i / CP`.
Reads reuse complete consumer CP rounds. Request-completion writes preserve complete token
blocks within a partial CP round and exclude the final incomplete token block.

Sharded CP requires `canonical_v1`. ALL_GATHER, ALL_GATHER_WITH_OVERLAP and ALLTOALL use
the same CP page plan. Decode's `PREFILL_CP` describes the producer topology; payloads are
placed according to Decode's own attention TP. Worker I/O uses allocation generation and
an access fence to keep transfers and computation mutually exclusive.

Canonical requires DEVICE groups and `pp_size=1`. Compact CP SWA rings, opaque state layouts,
incompatible block coverage and orthogonal TP×CP execution are unsupported. Typed MHA SWA
retains its window policy and supports TP head conversion. Canonical GPU and explicit HOST
CPU IOV writes use the same logical shard slicing. Remote reads target GPU memory, with the
transfer strategy selected by SDK/PACE.

### Ignoring request cache switches

Setting `RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES=1` (default off) ignores the request's
`reuse_cache` switch for lookup and storage. Local tiers and DEVICE uploads already follow
deployment policy regardless of this setting. It does not enable a tier disabled by the
deployment, and deployment-level `REUSE_CACHE=0` still disables reuse entirely.

# MultiTaskPrompt
Create static cache for long-text System Prompts, directly reading KV cache from static cache in each request instead of recomputing. This method can significantly reduce the model's First Token Latency.

## Usage
### MultiTaskPrompt
rtp-llm specifies the system prompt information file that needs static caching through the `--multi_task_prompt` parameter. The format is similar to the following:
``` json
[
    {"task_id": 1, "prompt": " <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>"},
    {"task_id": 2, "prompt": "你是一个严谨的程序员，你接下来需要非常谨慎的思考并回答以下问题:"}
]
```
You can also pass the above JSON through the `multi_task_prompt_str` environment variable.

After startup, the model will run the above system prompts and cache the KV cache in GPU memory. During subsequent runs, if a task_id is specified, this prefix can be used. Demo is as follows:
**Note: MultiTaskPrompt cannot currently use flash attention due to mismatched lengths between Q and KV, requiring the environment variable `export ENABLE_FMHA=OFF` to be configured before running the code**

``` python
import os
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory, ModelConfig

os.environ["MULTI_TASK_PROMPT"] = "/path/to/file"
# os.environ["MULTI_TASK_PROMPT_STR"] = "{json str}"
model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

# Using system prompt with task_id=1 to concatenate the request
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
    "task_id": "1"
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# Not using system prompt
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```

### Note:
MULTI_TASK_PROMPT is served out of L1. When configured, it automatically enables `REUSE_CACHE`
and `ENABLE_DEVICE_CACHE`, preserving the existing behavior.
When a task ID is specified, the system prompt of the task_id is used to concatenate the request, and the longest matching historical request is found in the KV cache to reuse the KV cache.
When no task ID is specified, the user's prompt is used to find the longest matching historical request in the KV cache to reuse the KV cache.
