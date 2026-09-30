# HOST/CPU 源 remote cache 写入

## 接入入口

`BlockTreeCache::insert(..., Tier::HOST)` 支持已经填好的 HOST-only `GroupSetResource::host_block`；caller 在调用期间持有对应 pool 的有效引用。本地 DEVICE→HOST 存储仅完成拷贝与缓存发布，不触发 remote 写入。

集成 request finish 双写后，成功请求设置内部 `InsertInfo::write_remote_from_device`，从 DEVICE 独立提交一次 remote 写入；该请求的 HOST 拷贝完成后不重复提交 CPU 写入。本地拷贝失败不影响已提交的 DEVICE remote 任务。直接 ready HOST 和独立 DEVICE→HOST 存储仍保持上述 CPU 源路径，详见 [request finish 契约与验收](../kvcm_remote_cache.md)。

直接使用 `StorageBackend` 时，初始化同时传入按 tag 绑定的 HOST pools 和 HOST buffer resolver，设置 `StorageRequest::source_tier = Tier::HOST`，再调用 `prepareWrite` 和 `write`。一个请求使用同一种来源。match/read 仍只接收 DEVICE handles。

HOST pool 中一个 packed block 可包含多个 group；每个 group 按 layer、KV/scale 顺序解析多个 CPU 地址。采用与 `DeviceHostTransferExecutor` 一致的实际布局，保留异构 layer 大小及 MTP 物理大小，不把 pool 对齐 padding 写入 remote。各 TP rank 继续使用自身 shard 和对齐的 block index；直接提交 HOST 数据的 caller 需要保证各 rank 对应 payload 已就绪。HOST RPC 标志遵循已有 same-build KV 协议，所有 TP peer 需使用同一实现。

目标后端是 PACE/TairMempool。storage config 包含 Mooncake 时，HOST client 的 `regist_span=nullptr` 不满足其注册要求，创建失败并 abort 写会话；日志提示该限制。GPU client 不受影响。本次不扩展 Mooncake 的 CPU 传输，也不推断其他后端的支持范围。

## 写入与生命周期

- 显式 HOST 写任务在发布本地数据、清理重复 block 或检查 watermark 前取得 STORE 引用，按 `(pool, block)` 去重；一个 packed block 的多个 tag 共享这一次引用。
- 每个 rank 从自己的 HOST pool 生成真实 CPU IOV，再复制到该次传输持有的 CPU snapshot，IOV 数量和字节顺序保持不变。CPU trace 避开 SDK 的 GPU kernel 地址哈希检查。GPU 数据仍走原有 DEVICE resolver、长驻 transfer client 和检查路径。
- CPU 写入使用按 tag/spec 配置创建的临时 transfer client。按 pinned SDK 源码契约，其析构停止并等待 SDK worker，随后才能销毁 snapshot；SDK 超时返回本身不是释放依据。该收尾可能超过 SDK 返回超时的时刻。集成 P1 后，TP payload 广播在超出预算时等待 peer 收尾，再判失败并释放调度 rank 的引用。默认 factory 的 client 创建在进程内串行，保护真实 PACE SDK 初始化中的共享状态和随机数生成器。
- HOST 布局偏移按 GroupSet 缓存，后续按 tag/layer 查找并代入当前 block 的 CPU 基址，避免每个 layer 重复扫描全部布局。
- 保留 `StartWrite → SaveKvCaches → FinishWrite`、offset/sparse mask、actual URI 回填、group/pool 路由和一个 key 对应多个 TP/group URI；SDK 自带的 `GroupBySdk` 没有在 RTP 重写。

## 来源与当前集成版本

2026-09-28 通过远端 refs 核对：

| 对象 | commit |
|---|---|
| RTP-LLM 基线 main | `382e155932e3bec7c58dc5c6b220773d66db9d9a` |
| KVCM 内源目标 | `719f7cb2e31d6db9b8d95f541c8df4b4b2f68436` |
| 内源固定的开源 SDK | `0bc254bcf9dc44cbff47ced8104f5df845e5d3de` |
| 内源固定的 PACE/tair-mempool | `da631aadf9f926fc46d402a5838c6db3a6b0a8f5` |
| 单独核对的开源 main | `6c463548c22b5780088a186e032056373bf5b544` |

实现依据是内源目标及其固定子模块，开源 main 仅用于接口对比。真实内源 `TairMempoolSdk::Put` 将 CPU IOV 交给 `DramToMP`，通过 `pace_copy_batch_async` 和 `pace_synchronize` 搬运数据。公开 stub 不作为真实传输依据。

源任务在 `ca56e898e5` 提交时未更新旧 RPM。该提交现已 cherry-pick 到 P1/ST 分支，集成提交为 `69a55bca02`，沿用 `deps/kvcm.bzl` 的来源锁：KVCM 内源 `32dc3162ec4f9f981617f8d82f9696a4faa2fe5b`、SDK `6015fca48a091dc18ea9497518138cb58959c3f2`、PACE `770bd4df361f86cd937f9144e910d202e1a7401f`。验收使用通过 URL/SHA256/source_id 门禁的配套制品，以及已经隔离内部 autil/gRPC 符号的 SDK RPM；制品要求见 [P1 smoke 说明](../kvcm_remote_cache_smoke.md)。

冲突解决保留 P1 的 metadata 字段 7，新增 `host_source` 使用字段 8。Follower 按来源校验 DEVICE/HOST 物理池范围并保留 backing pool；分配引用由调度 rank 保持至所有 rank 的 I/O 收尾。所有 TP peer 仍需使用相同的协议实现。

## 验证范围

补充用例覆盖 HOST 直接入口、本地 HOST 存储不触发 remote 写入、CPU/GPU resolver 选择、多 IOV、packed 多 group/scale/物理大小、共享 HOST pin 的完成/失败/拒绝/停机/未提交释放、SDK 超时 drain、写入 mask 及 actual URI 回填。原有 GPU、多 URI 和独立 pool 用例继续保留。

2026-09-29 在独立 CUDA 13.2/Torch 2.11.0+cu130/SM103、200 GiB DRAM/TENT TCP 环境验证：

- 12 个相关 C++ 测试目标、217 项结果通过，覆盖 HOST ready 入口、DEVICE→HOST 成功/失败提交、多 layer/group/scale、snapshot、临时客户端超时 drain、HOST 引用和 TP 来源/URI 传递。最终未改变输入的目标复用本次运行的 Bazel 测试缓存。
- 模型库增量构建通过，更新后的 C++/Python protobuf binding 可装载。
- 真实 Qwen2.5-0.5B 单卡及 TP2 测试关闭 DEVICE cache、开启 64 MiB HOST cache，日志确认存入 HOST；HOST watermark 强制清除本地复用。冷请求复用为 0，热请求远端复用 640 token，local/memory 复用均为 0，输出一致。
- 原 DEVICE 单卡与 TP2 路径的模型回归通过。上述四个场景合计 8 个请求，使用 block size 16 的严格冷/热输入。

首次运行修正了测试中的命名空间、resolver 签名、非法稀疏拓扑夹具，以及旧的“HOST 不触发 remote 写入”断言；保留了异步返回与 remote 完成前后引用检查。仅修正测试所需的夹具和预期，没有放宽字节、来源、输出或复用断言。

Mooncake HOST、SSD、跨布局 TP/CP、真实 SDK 慢 I/O/队列饱和、性能与完整 CI 未获本次运行验证。真实模型覆盖的是 DEVICE→HOST 后的 CPU 源写入；ready HOST 直接入口由组件用例验证。
