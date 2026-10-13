# KVCM GDR 与 ADV 接入方案

## 1. 目标与边界

RTP 的远端 KV Cache 支持多 group/pool、请求结束 GPU 双写、显式 HOST 写入、
GPU GDR，以及 `canonical_v1` 非对称 TP/CP 布局。传输复用 SDK/PACE 的既有
IOV、注册和 direct/landing/staging 机制。

保留默认 prefix、直接 `MatchLocation`、`forward_context.sw_size` 和已支持的
元数据查询类型。配置使用 v9，参与通信的 rank 使用同一构建版本。

## 2. 数据路径

| 场景 | 行为 |
| --- | --- |
| 请求成功结束 | 按配置写入本地缓存，同时从 DEVICE 数据源异步写 remote，复用 #1490 的 GDR 路径。远端写入不等待本地 D2H copy，不复用 HOST copy。 |
| 显式 HOST→remote | 调用方显式插入已就绪 HOST 内容时，使用普通 CPU IOV；持有 HOST STORE 引用，不依赖共享 HOST 注册或专用 HOST transfer client。 |
| 远端读取 | 目标始终为 GPU IOV。是否走 direct、landing 或 staging 由 SDK/PACE 决定，RTP 不分配独立 HOST 读池。 |
| 多 group/pool | 不同 group 可有不同 block 字节数，同组内一致；同一 remote key 对应相同 token 范围，保留 IOV 逻辑顺序。 |
| canonical | 按逻辑 KV shard 和全局 token block 映射本地 TP/CP；GPU 路径及显式 HOST CPU IOV 使用同一 shard 语义。 |

普通本地 D2H copy、迁移和驱逐不额外触发 remote 写入。上传失败不回滚本地缓存，
也不改变已经成功结束的请求。

Canonical Instance 使用独立命名空间，不能与 legacy payload 共用 Instance。
CP 读取复用完整 consumer round；写入保留部分 CP round 内完整的 token block，
排除未完成的最后一个 token block。支持范围及布局拒绝条件见远端缓存使用说明。

普通 MHA/GQA/MQA 的 P/D CacheStore 按 kernel page 发布完整 head 范围的
K/V/scale 连续 buffer，使用 `_page_<index>` key。ALLOCATE 携带真实来源的 CP
拓扑和按 tag 的 physical/kernel token span，Decode 将元数据转发至加载 worker。
主模型与参与传输的 MTP 模块按既有同 tag token-span 约束使用各自物理 buffer。
两端 physical token span 相同、kernel page 大小不同时，以最大公约数划分每个
源 head 的 token 片段，直接写入目标 page/head/token 地址。各连续分片按
`(partition_count, partition_id)` 分桶调用既有 `loadBuffers`，单次调用的 key 唯一。
全部 context 成功后按原 physical block/group 统计完成量；缺失几何、CP 拓扑冲突
或不同 physical token span 在提交前失败。通信端使用相同构建版本，多 group、
opaque、MLA 和 LINEAR 沿既有整组件路线传输。

## 3. SDK 接口与注册

配套 C++ SDK 同时提供 `GetHostCacheState`、`GetCacheLocationsByBackend`、
`StartWrite(..., min_replica_count)`，以及 #367 的 GPU 注册重载。

首个 transfer client 在第一次 PACE INIT 前提交全部所需 GPU span，后续 client
借用这些注册。关闭时先 drain 借用方，再关闭注册持有方。保留 SDK 既有
`host`、`gpu` 和 Create 接口，不扩展多 HOST 注册。

显式 HOST 写入使用原业务 tag 的普通 transfer client 和 CPU IOV；各 rank 按
本地 cache key/coordinate 解析 HOST block 并持有自己的 STORE 引用。
不能将 rank 0 的 HOST block 编号作为其他 rank 的分配身份。CPU IOV 使用已有
trace 控制，避免按 GPU 地址计算调试 hash。

分片 CP 的显式 HOST 插入通过已有 `remote_write` 保留原始全局 `remote_keys`，
同时提供 HOST 来源、本地 round endpoint keys 和 handles。上传本地 HOST 树
保存的完整 round，worker 复用 `CPSlotMapper` 投影全局 key/coordinate；空写入
shard 的 TP replica 安全跳过。GPU 请求结束上传继续保留部分 round 的完整 block。

SDK 并行调用只有全部任务已确认结束时才返回确定失败。未排空的任务或超时结果
使用已有超时状态，不能被普通错误遮蔽。开启 `PACE_ENABLE_GATHERSCATTER=ON` 时，
每个 block 内的异构尺寸 IOV 使用 Normal 保序路径；兼容的等尺寸加速继续保留。

## 4. 引用与完成状态

DEVICE 与显式 HOST STORE 引用统一纳入 storage task，保留 worker I/O fence、
GPU 生产者事件和 allocation generation。HOST 与 DEVICE 身份分别处理，跨 rank
DEVICE generation 按 group tag 对齐。

| 结果 | 处理 |
| --- | --- |
| 确定成功或确定未提交的失败 | 正常释放引用。 |
| KVCM SDK 超时或远端完成状态不明 | 隔离相关引用、禁止新传输准入，并保留至进程终止。 |
| 读取完成未知 | 终止该次请求，不使用同一目标 block 重计算；确定失败保留既有 Prefill fallback。 |

隔离后尚未提交的请求保留本地命中，缺失部分重新计算。

SDK 任务返回或析构不能代替物理 DMA/RDMA 完成证明。应用层保留
TIMEOUT/COMPLETION_UNKNOWN 状态及原 ErrorInfo code/message。
`RemoteOperationResponsePB` 保留 reserved 字段及名称，保留字段 6、7，
`transfer_status` 使用字段 8。

## 5. 配置与制品

内部 PACE sidecar 先于 RTP CLI 解析，GDR 启用条件统一通过环境变量传入。
有效 GDR 要求 external master、memory cache、remote cache 和 GDR 开关开启。
共享 HOST zero-copy 不属于该接入方式；HOST 容量仅用于既有本地缓存。
纯 CLI 启动条件覆盖不在该环境变量模式的范围内。

普通非 FFN 分离、DP1、跨节点 TP 的 GDR 使用有界空闲控制轮。rank 0 在远端
BlockTree 与 GDR 均启用时，将 FIFO、ratio、Batch 空闲等待限制为 10 ms，复用
empty streams 的 `skip_run` shape-header 通信，使此前提交的 NCCL 等待可完成。
不增加 FakeStream 或模型 forward，不以在途传输计数控制轮询停止。
代价是空闲时约每秒 100 个空调度轮和更多调度指标上报。Normal 每轮 1 个
shape-header；MTP PDFUSION 每轮在 prefill、decode 各 1 个，共 2 个 shape-header。
空控制轮产生 CPU、GPU 和网络开销，随拓扑与通信耗时变化。DP>1 与 FFN 分离保留
既有行为，不扩大其 FakeStream 或模型空跑范围。

配套公共 SDK 同时提供所需元数据接口、#367 GPU 注册和传输完成状态。
内部 KVCM 适配对应公共 SDK，PACE 选择兼容 GDR 契约的版本。
RTP 内部 gitlink 指向对应公共整合版本。
配套 RPM 在 [KVCacheManager CI](https://code.alibaba-inc.com/tair_kvcache/KVCacheManager/ci/runs) 构建。

每个制品明确公共 SDK、内部 KVCM、PACE 三个源码提交以及 URL、SHA256。
保留 CUDA12/12.9/13 x86/ARM 的平台选择，并更新实际生效的依赖入口。
包内接口、源码组合和平台 ABI 必须一致。

验收覆盖 GPU/HOST IOV 字节顺序、GPU 注册与生命周期、非对称 TP/CP shard、
默认 prefix/SWA 元数据行为，以及跨 rank 提交前拒绝、确定完成、超时和完成未知。
GPU GDR、模型复用和性能结论需要实际运行结果支撑；制品打包不能代替这些验收。
