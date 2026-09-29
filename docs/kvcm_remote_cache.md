# KVCM remote cache

## Dependencies and artifacts

`deps/kvcm.bzl` pins the internal KVCM, public SDK/Manager, and PACE revisions. Build the client RPM and server archive from the same source combination, with matching SDK headers and shared libraries. The updated virtual interfaces and StartWrite arguments change the ABI; legacy RPMs and the public PACE stub are incompatible.

The built-in artifact records pin the published SDK RPMs and x86 Manager archive from build `77672840`, including their URLs, SHA256 hashes, and matching `KVCM_SOURCE_ID`. The source tuple is internal `1c24aeac35c819c544316e9753eec0186ea53bd3`, public SDK/Manager `a71117d9745d7228f92aaf64c5414737878bd153`, and PACE `770bd4df361f86cd937f9144e910d202e1a7401f`. No external manifest is required for these artifacts.

Client selection follows the BUILD configuration. The standalone Manager is shared across x86 CUDA configurations:

| Build configuration | Client manifest variant | Server manifest variant |
|---|---|---|
| CUDA 12 x86 | `cuda` | `server` (x86) |
| CUDA 12.9 x86 | `cuda129_x86` | `server` (x86) |
| CUDA 13 x86 | `cuda130_x86` | `server` (x86) |
| CUDA 13 ARM | `cuda130_arm` | No local ARM Manager package; use an external Manager |

CUDA 12.9 and CUDA 13 variants are selected by Bazel and are not overridden by `KVCM_CLIENT_VARIANT`. Other configurations default to the CUDA 12 x86 SDK. No CPU-only SDK is included in this release; `--repo_env=KVCM_CLIENT_VARIANT=cpu` requires an explicit paired CPU artifact in an override manifest. Targets that launch the packaged Manager are restricted to x86; the ARM SDK can communicate with an external Manager.

The generated `KVCM_CLIENT_VARIANT` marker preserves the complete CUDA variant. The legacy manifest key `cuda` produces `cuda12_x86`; the other CUDA variants retain the names above. PACE model smoke targets tagged `L20_CU13` or `H20_CU13` require `cuda130_x86` and reject other SDK variants before starting the Manager.

To override the published artifacts, pass `--repo_env=KVCM_ARTIFACT_MANIFEST=/absolute/path/MANIFEST.json`. The manifest requires exactly one entry for the selected client variant and the `server` variant. All source IDs must match `internal_commit:opensource_commit:pace_commit` from the source lock:

```json
{
  "source_id": "<internal commit>:<opensource commit>:<pace commit>",
  "artifacts": [
    {"variant": "cuda", "source_id": "<same source_id>", "url": "<CUDA 12 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda129_x86", "source_id": "<same source_id>", "url": "<CUDA 12.9 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda130_x86", "source_id": "<same source_id>", "url": "<CUDA 13 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda130_arm", "source_id": "<same source_id>", "url": "<CUDA 13 ARM RPM URL>", "sha256": "<SHA256>"},
    {"variant": "server", "source_id": "<same source_id>", "url": "<x86 Manager archive URL>", "sha256": "<SHA256>"}
  ]
}
```

Bazel validates the source IDs and download hashes. The server archive must also contain the matching `KVCM_SOURCE_ID` marker. Override manifests are tracked as Bazel file inputs, so in-place edits invalidate the affected artifact repositories, including on Bazel 6.4.

The `remote_cache_pace_contract` and `remote_cache_pace_ssd_contract` targets exercise CPU buffers using the selected SDK; their `smoke_kvcm_p1_cpu*` suite names describe the buffer type, not a CPU-only SDK requirement. With the published SDKs, the matching CUDA runtime must be available. Model smoke tests still require a CUDA SDK. CUDA 13 keeps remote cache opt-in: place `--config=remote_kv_cache` after `--config=cuda13` or `--config=cuda13_arm`. An external `KVCM_PACE_FIXTURE` must carry the updated source ID; the PACE provider and consumer revision remains unchanged.

SDK packaging must isolate its internal autil/gRPC symbols from RTP to avoid symbol interposition and duplicate destruction. Link with `-Wl,-Bsymbolic` and a version script exporting only the KVCM API (`_ZN16kv_cache_manager*`, `_ZNK16kv_cache_manager*`, `_ZTVN16kv_cache_manager*`, `_ZTIN16kv_cache_manager*`, and `_ZTSN16kv_cache_manager*`), with all other symbols local. Update the RPM hash in the manifest after relinking.

## Configuration

| Argument / environment variable | Default | Meaning |
|---|---:|---|
| `kvcm_remote_layout` / `KVCM_REMOTE_LAYOUT` | `legacy` | Use `canonical_v1` on both ends for asymmetric TP/CP; see [layout and topology requirements](#asymmetric-tpcp) |
| `kvcm_default_query_type` / `KVCM_DEFAULT_QUERY_TYPE` | 2 | Instance default: 1=batch, 2=prefix, 3=SWA, 4=Mamba |
| `kvcm_query_type` / `KVCM_QUERY_TYPE` | 0 | Request mode; 0 uses the Instance default |
| `kvcm_sw_size` / `KVCM_SW_SIZE` | 0 | SWA window in cache keys/blocks; must be positive for SWA |
| `kvcm_read_backend_type` / `KVCM_READ_BACKEND_TYPE` | 0 | 0=regular query; 1=3fs, 2=mooncake, 3=PACE DRAM, 4=NFS, 5=VCNS 3fs, 9=PACE SSD |
| `kvcm_min_replica_count` / `KVCM_MIN_REPLICA_COUNT` | 0 | Minimum readable replicas for StartWrite; the server treats 0 as 1 |

Explicit `KVCM_CLIENT_CONFIG` JSON takes precedence over the generated Instance configuration; an omitted `default_query_type` defaults to 2. Requests can override it with `kvcm_query_type`. Each backend binds to one default Instance. Backend-specific queries use batch mode and accept only `kvcm_query_type=0` or `1`; use `kvcm_read_backend_type=0` for regular Mamba/SWA queries.

Custom `block_size`, `location_spec_infos`, and `location_spec_groups` must match the local key token stride, group payload sizes, and group/TP rank mapping. Spec lists within a group may be reordered; a single group may omit the explicit group map.

Set `--kvcm_model_sdk_config` (environment variable `RECO_MODEL_SDK_CONFIG`) for one data backend:

- DRAM: `[{"type":"pace","sdk_log_level":"INFO"}]`.
- SSD: `[{"type":"pace_ssd","sdk_log_level":"INFO"}]`. Reads can select `kvcm_read_backend_type=9`; the server must use `ST_TAIRMEMPOOL_SSD` and `media_type=5`.

The server storage configuration supplies addresses and media. Regular write candidates should contain only the selected data backend. Configure event storage separately in `event_report_storage_candidates`.

The pinned PACE revision (`770bd4df`) supports TENT TCP transfers, but defaults to AFT. To use TCP, both provider and consumer sidecars must use that revision and receive:

```sh
export TAIR_MEMPOOL_ENABLE_TENT=1
export MC_TENT_CONF='{"transports":{"tcp":{"enable":true},"aft":{"enable":false},"rdma":{"enable":false},"barex":{"enable":false},"shm":{"enable":false}},"policy":[{"name":"tcp_default","segment_type":"memory","transports":["tcp"]}]}'
```

TENT uses an RDMA device slot, so `--no_rdma` disables it. Updating the SDK dependency alone does not change the transport.

Batch/SWA misses preserve their original key positions. For layouts containing SWA groups, internal payload matching translates SWA queries to batch queries so FULL locations outside the window remain available; explicit metadata queries retain the requested mode. Reuse requires a complete FULL prefix, final LINEAR state, and complete SWA window across every TP rank; missing URIs do not count as hits. Mixed LINEAR+SWA writes may store the FULL+LINEAR portion first, but reads still require the complete SWA window. Existing IOV, pool/group, FULL+LINEAR, and same-layout TP support is retained.

Groups in one Instance must share `cacheKeyTokenStride()`, used as the registered token `block_size`; mismatches reject initialization. Each location spec describes its group's fixed KV/scale payload bytes, which may differ between groups. Multiple standard FULL MHA/MLA groups remain subject to model-layout validation.

In `legacy` mode, generated Instance identities include the default query mode and registered group configuration. Multi-group identities also include tag-sorted per-layer physical layouts and layer ownership, creating a new cache namespace. `canonical_v1` uses a topology-independent namespace and the base token block size; sharded CP requires this format. Single-group identities remain unchanged when their registered token stride is unchanged.

Canonical Instance IDs start with `canonical_v1_` and exclude runtime TP/CP/DP partition sizes. Both ends must agree on model weights, KV dtype, block size, layer/group layout, draft configuration, `CHECKPOINT_PATH`, `BIZ_NAME`, `kvcm_model_extra_info`, and salt. Use distinct `kvcm_model_extra_info` values for different weight quantization or other KV-affecting settings, including weights changed in place: the checkpoint path hash does not verify weight contents.

Custom canonical configuration requires a `canonical_v1_` Instance ID and matching logical specs, groups, block size, and storage `model_deployment`; mismatches fail initialization. Legacy and canonical payloads cannot be mixed. Payload RPCs carry the format and logical shard, and canonical workers reject requests missing these fields.

Custom IDs must match the existing server configuration. `KVCacheConfig` uses pickle version 10 with 76 items and reads versions 1-9. Older states without the request-finish write flag default it to disabled; communicating processes must use the same build.

## Asymmetric TP/CP

Set `--kvcm_remote_layout canonical_v1` on both producer and consumer. It stores logical KV components independently of runtime partitions; the default `legacy` format retains rank-based TP storage. Sharded CP requires `canonical_v1`.

| Layout | Logical storage and topology support |
|---|---|
| Typed MHA/GQA/MQA FULL, including INT8/FP8 scales | One spec per global KV head per group, named `v1_F<tag>_h<head>`. Each layer stores K, V, K-scale, then V-scale; kernel pages may span multiple IOVs. Supports TP head conversion and CP round-robin blocks. |
| Typed MLA FULL | One complete latent/rope record with embedded scales, with one designated writer. Supports replicated components across TP and round-robin CP blocks. |
| Typed FULL+LINEAR | One state group per global key head, including its value-head SSM state and Q/K/V convolution history. Supports TP state conversion and replicated LINEAR checkpoints with round-robin FULL blocks under CP. |
| Typed MHA SWA | Converts heads while retaining the window policy. Compact CP SWA rings are rejected at initialization. |
| Opaque KV/state and DSV4 compressed or fixed-ring components | Rejected at initialization. |

Storage registration uses `model_deployment.tp_size=dp_size=pp_size=1`; RTP assigns payload work to physical workers. Each pool registers its complete span. The SDK self-spec identifies one logical spec readable by the worker; transfers use the pool's logical specs, URIs, and IOVs. This identifier does not establish cross-topology CP host-state/P2P ownership.

Ordinary TP partitions heads by `gcd(global_kv_heads, attention_tp)`, matching weight loading. Contiguous ranks may replicate the same heads: the first replica writes each head and reads populate every replica. CP uses the existing TP worker group with attention TP size 1; arbitrary orthogonal TP×CP execution is unsupported.

FULL sharded CP assigns global block `i` to owner `i % CP` and local slot `i / CP`. The local tree retains virtual blocks covering `block_size * CP` tokens; remote metadata uses fixed token-block keys. Read hits are rounded down to complete consumer CP rounds. Request-finish writes construct global keys before local-tree projection, retaining complete token blocks in a partial CP round and excluding incomplete token blocks. The request-finish write gate also applies to canonical writes.

In the Qwen CP path, rank 0 writes the replicated LINEAR state and every reader replica receives the final checkpoint. Decode's `PREFILL_CP` setting describes the producer topology; Decode uses its own attention TP for FULL/LINEAR storage. P/D transfer partitions independent MHA K/V records by `gcd(global_kv_heads, decode_workers)` and shares partitions across GQA replicas.

`CPKVCachePlan` maps sharded pages for ALL_GATHER, ALL_GATHER_WITH_OVERLAP, and ALLTOALL. Prefix collection precedes asynchronous new-KV communication; only owned pages are written back to persistent sharded pools. This requires temporary buffers and prefix communication.

Canonical layout rejects non-DEVICE groups, incompatible key/physical-block coverage, and PP. Request-finish remote writes use DEVICE sources; shared HOST remote I/O remains a `legacy` path. MTP resolves child-owned physical specs; layers sharing a tag must agree on logical shard count and partition semantics. Existing [HBM event publisher topology restrictions](backend/kv_cache_event_publisher.md) continue to apply.

## RPC interface

Metadata requests follow `RpcService/ExecuteFunction` -> `KVCacheManager::executeFunction` -> `KVCMStorageBackend::execute`. They must target TP0 and use this backend's Instance. Payload I/O runs on the corresponding TP rank. SDK errors return a non-OK RPC status.

| `RemoteOperationRequestPB.op` | SDK method | Result |
|---|---|---|
| `REMOTE_OPERATION_MATCH_LOCATION_LEN` | `MatchLocationLen` | `matched_blocks`, in blocks |
| `REMOTE_OPERATION_MATCH_META` | `MatchMeta` | `locations` and the original `metas` string |
| `REMOTE_OPERATION_REMOVE_CACHE` | `RemoveCache` | Success or failure |
| `REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND` | `GetCacheLocationsByBackend` | Key-aligned `backend_locations`, including empty entries, type, spec size, and URI |
| `REMOTE_OPERATION_GET_HOST_CACHE_STATE` | `GetHostCacheState` | Hosts, local length, P2P fetch length, and final length |
| `REMOTE_OPERATION_MATCH_LOCATION` | `MatchLocation` | Original locations, preserving empty batch/SWA entries |

Example protobuf JSON:

```json
{"remote_request":{"op":"REMOTE_OPERATION_MATCH_LOCATION_LEN","trace_id":"cache-length","metadata":{"query_type":2,"block_keys":[101,102,103]}}}
```

`metadata` accepts tokens, offset/bool masks, window size, detail level, backend, spec names, medium, and P2P host count. Backend queries support batch mode only; nonempty spec-name lists must align with the keys. Host-state queries support prefix and Mamba modes and return metadata.

With `kvcm_read_backend_type` set, locations are mapped by group/rank into TP payload requests, and their URIs are passed to `TransferClient::LoadKvCaches`. Event URIs describe locations and are not used as payload backends by this integration.

## HOST shared-memory I/O

With KVCM remote cache and HOST cache capacity enabled, HOST pools use fd-backed CUDA-pinned shared memory. Each group set registers its main cache region. A separate bounded region serves remote reads; its blocks, including reserved block 0, count against the existing HOST budget. Temporary reads therefore do not change main-pool block allocation order. HOST cache remains disabled by default, and GPU-only configurations allocate no shared HOST regions.

Shared HOST writes use completed, tree-admitted blocks. Each rank resolves the full key path and pins its own HOST block; rank 0 block indices are not identities on other ranks. Only rank 0 initiates metadata writes. Remote reads enter the separate HOST region and then use the existing HOST-to-DEVICE transfer. Missing registration, insufficient read capacity, or incomplete group members fall back to DEVICE reads. If HOST registration fails, writes try CPU IOVs through the original client and fail if the backend does not support them.

Tags map to contiguous portions of packed HOST blocks in group-set member order. CPU writes reuse registered clients; registration alone does not guarantee a direct transfer. PACE capability, tiering, cache state, and fallback still determine the actual path. This does not remove GPU-to-DRAM transfers or reduce the PACE consumer's reserved pools.

When local HOST payload completion is uncertain, its block references remain charged to the pool. Pre-submission failures and known local success release local pins even if another rank fails. An uncertain main pool rejects subsequent HOST transfers and writes; an uncertain read pool falls back to DEVICE reads. Uncertain mappings and file descriptors remain alive until process exit. The shared read region is part of the existing budget, not a reduction in physical memory usage.

## I/O lifetime

When a request reaches `FINISHED` successfully and cache reuse is allowed, complete KV blocks are stored in the highest enabled local tier (DEVICE, then HOST, then DISK). Request-finish remote writes require both `ENABLE_REMOTE_CACHE=1` and `ENABLE_REMOTE_CACHE_WRITE_ON_FINISH=1`; the latter defaults to false and can also be set with `--enable_remote_cache_write_on_finish true`. Remote-only deployments use the same write gate. Disabling it preserves remote lookup/read and explicit writes. Per-request `reuse_cache` and `RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES` retain their existing behavior.

The submission includes available complete blocks, including reused prefixes; the final partial block is not published. Remote writes honor the offset/bool mask returned by `StartWrite` to skip blocks with enough replicas. Local asynchronous stores and remote tasks retain independent source references after request release, releasing them on completion or rejection.

When request-finish remote writes are enabled and the local target is HOST, single-rank KVCM reuses the completed HOST copy for remote I/O. HOST allocation, queue, or copy failure falls back to DEVICE sources pinned before request release. Remote failure does not roll back published HOST cache. TP greater than one, other local tiers, and backends without HOST reuse retain DEVICE writes. Ordinary local DEVICE-to-HOST stores do not trigger remote writes.

Request finish triggers submission; it does not guarantee task admission or remote durability. DEVICE publication is synchronous, while HOST/DISK and remote writes complete independently and asynchronously. Rejection releases temporary references. Remote I/O failure follows the existing `FinishWrite` abort rules without rolling back local cache or changing request success.

RTP requires `sdk_config.drain_on_timeout=true`. Caller waits are bounded by `kvcm_get_broadcast_timeout` (read/metadata) and `kvcm_put_broadcast_timeout` (write), including single-rank calls and shutdown. Both must be positive and default to 15s. Allow room above the SDK budgets (12s by default) for metadata and TP dispatch. Timeout returns failure without waiting for submitted I/O; failed reads are not published as reusable cache entries.

On timeout or uncertain completion (SDK transfer errors or failed TP payload RPCs), RTP retains the operation's resources for the process lifetime: controller allocation pins prevent block reuse, and followers retain the backing pools. The affected backend rejects new operations, and shutdown does not wait for retained work. Even transient transfer errors are treated conservatively; restoring remote-cache capacity requires replacing the instance. All TP ranks must run the same version. This bounds caller waits without cancelling backend I/O or releasing memory it may still access.

Worker mirror pools use `WorkerCacheIOFence` to serialize payload access with computation and block copies. GPU completion events order producers before SDK access, allocation generations reject delayed RPCs for recycled block IDs, and payload RPCs participate in backend shutdown accounting. The fence wait budget is the greater of 30 seconds and twice the maximum SDK/RPC timeout (60 seconds with defaults). A timeout fails the operation without bypassing outstanding I/O drain.

Each StartWrite location must contain exactly the selected group's specs (logical shards in `canonical_v1`), all with nonempty URIs. FinishWrite commits only after every worker succeeds and all actual URIs are present. Writes map offset/bool masks back to the original keys and fill actual URIs into specs in the same order. Writes known to have failed abort through FinishWrite; uncertain writes leave their sessions for server expiry rather than immediately recycling destinations; empty sessions are closed on a best-effort basis, with server expiry as a fallback. PACE fallback preserves the hostname. DRAM uses `PREFER_LOCAL` (0) to avoid colliding with the legacy `ONLY_REMOTE` value 2; SSD uses `LOC_DEFAULT | MEDIA_TYPE_LOCALSSD` (5).

## Server event configuration

See [KV cache event publisher](backend/kv_cache_event_publisher.md) for publisher lifecycle and topology limits. Add the event storage to the Instance Group's `event_report_storage_candidates`:

```json
{"global_unique_name":"rtp_hbm_events","storage_type":"ST_EVENT_REPORT_L1P5","event_report":{"heartbeat_timeout_ms":30000,"cleanup_grace_ms":300000,"liveness_check_interval_ms":5000,"snapshot_min_interval_ms":1000},"check_storage_available_when_open":false}
```

## Supported scope

The integration supports SDK query/management interfaces, backend-specific reads, replica controls, same-layout TP payload routing, cache event reporting, explicit CPU/HOST-source writes, and request-finish local/remote writes. Asymmetric TP/CP requires opt-in `canonical_v1`; see [supported layouts and restrictions](#asymmetric-tpcp). Shared HOST I/O uses `legacy`; direct transfer depends on backend support. GDR is outside this integration. See [HOST-source writes](backend/remote_cache_host_write.md) for the CPU IOV and lifetime contract.

Models require matching client/server artifacts and an attention backend and page size supported by the target GPU. Publisher topology limits are documented in [KV cache event publisher](backend/kv_cache_event_publisher.md).

## Historical acceptance

The source record dated 2026-09-29 describes request-finish commit `0d388aca33`, integrated into the former P1/HOST branch as `4fd756e7bb`. Under CUDA 13.2/SM103, seven component targets reported 209 passing cases with no failures, errors, or skips and with test-result caching disabled: `block_tree_storer_test` (28), `stream_cache_resource_test` (42), `storage_backend_test` (35), `kvcm_mock_only_full_test` (26), `client_wrapper_test` (19), `kvcm_internal_test` (30), and `multi_rank_block_transfer_engine_test` (29).

Those cases covered DEVICE/HOST/DISK/remote-only finish paths, complete blocks and reused prefixes, masks, asynchronous failures, and independent reference release. The historical tree also covered ready HOST and automatic CPU writes after independent DEVICE-to-HOST stores; that automatic trigger was subsequently removed by `a388bc6605`. Request-finish cases checked that HOST completion did not duplicate the remote submission.

The source record also reports an incremental model-library build and ten Qwen2.5-0.5B requests across remote-only, HOST, HOST TP2, DISK, and DEVICE scenarios using the former P1 source lock and paired KVCM/PACE artifacts. HOST/DISK logs confirmed local publication. After watermark eviction removed local reuse, warm requests reused 640 remote tokens; cold reuse and warm local/memory/disk reuse were zero, and outputs matched the strict cold-start baseline.

DISK in that record means the RTP local disk tier; remote storage used PACE/DRAM. Remote SSD, real SDK slow I/O/queue saturation, P/D, cross-layout TP/CP, performance, and full CI were outside that acceptance. These historical results do not validate this transplant or its current V2 artifacts. No compilation or tests have been run for this transplant. See [P1 smoke](kvcm_remote_cache_smoke.md) for smoke entry points and artifact requirements.

## Test entry points

The component suite is `//rtp_llm/test/smoke:smoke_kvcm_p2_multi_pool`. The manual `//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:kvcm_multi_pool_pace_test` target requires `KVCM_P2_SERVER_ADDRESS` and `KVCM_P2_INSTANCE_GROUP` for a configured KVCM/PACE environment; it verifies payloads through matching and byte-for-byte readback.

Canonical component targets under `//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test` are `canonical_cache_layout_test` (logical layout and namespace compatibility), `kvcm_canonical_test` (registration, CP ownership, partial rounds, missing shards, and RPC validation), and `worker_cache_io_fence_test` (access serialization, generations, and producer completion).

`//rtp_llm/test/smoke:smoke_kvcm_p2_gpu` uses the PACE fixture and P/D runner for TP1→TP2, TP2→TP1, and CP2/4→TP1/2. Assertions check structured JSON output and cold/warm cache attribution; they do not measure cross-topology model-quality tolerance. Use the artifact and fixture configuration described above.
