# HOST/CPU-source remote cache writes

## Entry points

`BlockTreeCache::insert(..., Tier::HOST)` accepts a populated HOST-only `GroupSetResource::host_block`; the caller must hold a valid reference in the corresponding pool during the call. A local DEVICE-to-HOST store only copies and publishes local cache data; it does not trigger a remote write.

At request finish, a successful request sets the internal `InsertInfo::write_remote_from_device` flag and independently submits one DEVICE-source remote write. Completion of its HOST copy does not submit another CPU write. Local copy failure does not affect an already submitted DEVICE remote task. Direct ready-HOST insertion retains the explicit CPU-source path. See the [request-finish contract and historical acceptance](../kvcm_remote_cache.md).

Direct `StorageBackend` callers provide tag-bound HOST pools and a HOST buffer resolver during initialization, set `StorageRequest::source_tier = Tier::HOST`, then call `prepareWrite` and `write`. Each request uses one source tier. Match/read requests still accept DEVICE handles only.

A packed HOST block can contain multiple groups. Each group resolves CPU addresses in layer and KV/scale order, matching the physical layout used by `DeviceHostTransferExecutor`. This preserves heterogeneous layer sizes and MTP physical sizes and excludes pool alignment padding from remote payloads. Each TP rank uses its own shard and aligned block indices. Callers directly submitting HOST data must ensure that the corresponding payload is ready on every rank. All TP peers must use the same protocol implementation.

The target backend is PACE/TairMempool. A storage configuration containing Mooncake cannot create the HOST client because `regist_span=nullptr` does not meet its registration requirement; client creation fails and the write session is aborted with a diagnostic. GPU clients are unaffected. This change does not add Mooncake CPU transfers or establish support for other backends.

## Writes and lifetime

- Explicit HOST writes take STORE references before local publication, duplicate-block cleanup, or watermark eviction. References are deduplicated by `(pool, block)`, so tags sharing a packed block share one reference.
- Each rank resolves real CPU IOVs from its HOST pool and copies them into a transfer-owned CPU snapshot, preserving IOV count and byte order. CPU traces bypass SDK GPU-kernel address hashing. GPU payloads retain the DEVICE resolver, persistent transfer client, and check path.
- CPU writes create a temporary transfer client configured for the tag/spec. Its destruction drains SDK workers before the rank-local snapshot is released; an SDK timeout result alone does not permit snapshot release. Draining can exceed the timeout. TP payload broadcasts wait for peers to finish after the budget expires, then report failure and release controller references. The default factory serializes client creation within a process to protect shared PACE initialization state and its random generator.
- GroupSet caches HOST layout offsets. Later resolution looks up the tag/layer and adds the current block's CPU base address instead of rescanning the layout for every layer.
- `StartWrite -> SaveKvCaches -> FinishWrite`, offset/sparse masks, actual URI propagation, group/pool routing, and multiple TP/group URIs per key are preserved. RTP does not reimplement SDK `GroupBySdk`.

## Dependencies and protocol

The current transplant retains the V2 source lock and published artifacts in `deps/kvcm.bzl` unchanged. See [KVCM dependencies and artifacts](../kvcm_remote_cache.md#dependencies-and-artifacts) for the current SDK/Manager/PACE tuple, platform selection, and packaging requirements.

Metadata remains protobuf field 7; the write-only `host_source` flag uses field 8. Followers validate DEVICE/HOST physical pool ranges and retain the backing pool. The controller holds allocation references until I/O completes on every rank. All TP peers must use the same protocol implementation.

The original implementation was based on internal KVCM `719f7cb2e31d6db9b8d95f541c8df4b4b2f68436`, its SDK `0bc254bcf9dc44cbff47ced8104f5df845e5d3de`, and PACE `da631aadf9f926fc46d402a5838c6db3a6b0a8f5`, checked on 2026-09-28 against RTP main `382e155932e3bec7c58dc5c6b220773d66db9d9a`. Public main `6c463548c22b5780088a186e032056373bf5b544` was used only for interface comparison. The internal `TairMempoolSdk::Put` passed CPU IOVs to `DramToMP`, using `pace_copy_batch_async` and `pace_synchronize`; the public stub was not transfer evidence.

Original source commit `ca56e898e5` did not update the old RPM. Its former P1/ST integration was `69a55bca02`, using internal KVCM `32dc3162ec4f9f981617f8d82f9696a4faa2fe5b`, SDK `6015fca48a091dc18ea9497518138cb58959c3f2`, and PACE `770bd4df361f86cd937f9144e910d202e1a7401f`. Those are historical versions, not the current V2 dependency tuple.

## Coverage and historical acceptance

The retained cases cover ready-HOST insertion, local HOST stores without remote submission, CPU/GPU resolver selection, multiple IOVs, packed groups/scales/physical sizes, shared HOST pin release on completion/failure/rejection/shutdown/unsubmitted tasks, SDK timeout draining, write masks, and actual URI propagation. Existing GPU, multiple-URI, and independent-pool cases remain.

The source record reports validation on 2026-09-29 under CUDA 13.2/Torch 2.11.0+cu130/SM103 with 200 GiB DRAM and TENT TCP:

- Twelve related C++ targets reported 217 passing results. They covered ready HOST, the then-enabled automatic DEVICE-to-HOST CPU writes, multiple layers/groups/scales, snapshots, temporary-client timeout draining, HOST references, and TP source/URI propagation. Targets with unchanged inputs reused that run's Bazel test cache.
- The incremental model-library build passed and updated C++/Python protobuf bindings loaded.
- Qwen2.5-0.5B single-GPU and TP2 scenarios disabled DEVICE cache and enabled 64 MiB HOST cache. Logs confirmed HOST publication; watermark eviction removed local reuse. Cold reuse was zero, warm remote reuse was 640 tokens, local/memory reuse was zero, and outputs matched.
- Original DEVICE single-GPU and TP2 regressions also passed. The four scenarios comprised eight requests with block size 16 and strict cold/warm inputs.

The original test repair corrected namespaces, resolver signatures, an invalid sparse topology fixture, and the then-obsolete HOST submission assertion. It retained asynchronous-return and reference checks without relaxing byte, source, output, or reuse assertions. `a388bc6605` subsequently removed automatic writes after local HOST copies and updated the relevant cases. Its behavior must not be inferred from the earlier model results.

The historical run did not validate Mooncake HOST, remote SSD, cross-layout TP/CP, real SDK slow I/O/queue saturation, performance, or full CI. Its model coverage exercised the former automatic CPU write after DEVICE-to-HOST copying; direct ready HOST was covered by component cases. None of those results validates this transplant or the current V2 artifacts. This transplant has only static inspection; no compilation or tests have been run.
