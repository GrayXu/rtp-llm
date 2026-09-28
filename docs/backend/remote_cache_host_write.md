# HOST/CPU 源 remote cache 写入

## 接入入口

`BlockTreeCache::insert(..., Tier::HOST)` 支持已经填好的 HOST-only `GroupSetResource::host_block`；caller 在调用期间持有对应 pool 的有效引用。原有 DEVICE 源存入 HOST 的流程则在各 rank 拷贝成功后提交 HOST remote 写入。失败、超时或停机回滚的本地拷贝不提交 remote 写入。自动迁移及驱逐流程没有新增写入动作。

直接使用 `StorageBackend` 时，初始化同时传入按 tag 绑定的 HOST pools 和 HOST buffer resolver，设置 `StorageRequest::source_tier = Tier::HOST`，再调用 `prepareWrite` 和 `write`。一个请求使用同一种来源。match/read 仍只接收 DEVICE handles。

HOST pool 中一个 packed block 可包含多个 group；每个 group 按 layer、KV/scale 顺序解析多个 CPU 地址。采用与 `DeviceHostTransferExecutor` 一致的实际布局，保留异构 layer 大小及 MTP 物理大小，不把 pool 对齐 padding 写入 remote。各 TP rank 继续使用自身 shard 和对齐的 block index；直接提交 HOST 数据的 caller 需要保证各 rank 对应 payload 已就绪。HOST RPC 标志遵循已有 same-build KV 协议，所有 TP peer 需使用同一实现。

目标后端是 PACE/TairMempool。storage config 包含 Mooncake 时，HOST client 的 `regist_span=nullptr` 不满足其注册要求，创建失败并 abort 写会话；日志提示该限制。GPU client 不受影响。本次不扩展 Mooncake 的 CPU 传输，也不推断其他后端的支持范围。

## 写入与生命周期

- HOST 写任务在发布本地数据、清理重复 block 或检查 watermark 前取得 STORE 引用，按 `(pool, block)` 去重；一个 packed block 的多个 tag 共享这一次引用。
- 每个 rank 从自己的 HOST pool 生成真实 CPU IOV，再复制到该次传输持有的 CPU snapshot，IOV 数量和字节顺序保持不变。CPU trace 避开 SDK 的 GPU kernel 地址哈希检查。GPU 数据仍走原有 DEVICE resolver、长驻 transfer client 和检查路径。
- CPU 写入使用按 tag/spec 配置创建的临时 transfer client。按 pinned SDK 源码契约，其析构停止并等待 SDK worker，随后才能销毁 snapshot；SDK 超时返回本身不是释放依据。该收尾可能超过 SDK 返回超时的时刻，仍沿用已有 TP 广播 deadline 及失败收尾。默认 factory 的 client 创建在进程内串行，保护真实 PACE SDK 初始化中的共享状态和随机数生成器。
- HOST 布局偏移按 GroupSet 缓存，后续按 tag/layer 查找并代入当前 block 的 CPU 基址，避免每个 layer 重复扫描全部布局。
- 保留 `StartWrite → SaveKvCaches → FinishWrite`、offset/sparse mask、actual URI 回填、group/pool 路由和一个 key 对应多个 TP/group URI；SDK 自带的 `GroupBySdk` 没有在 RTP 重写。

## 核对版本与制品限制

2026-09-28 通过远端 refs 核对：

| 对象 | commit |
|---|---|
| RTP-LLM 基线 main | `382e155932e3bec7c58dc5c6b220773d66db9d9a` |
| KVCM 内源目标 | `719f7cb2e31d6db9b8d95f541c8df4b4b2f68436` |
| 内源固定的开源 SDK | `0bc254bcf9dc44cbff47ced8104f5df845e5d3de` |
| 内源固定的 PACE/tair-mempool | `da631aadf9f926fc46d402a5838c6db3a6b0a8f5` |
| 单独核对的开源 main | `6c463548c22b5780088a186e032056373bf5b544` |

实现依据是内源目标及其固定子模块，开源 main 仅用于接口对比。真实内源 `TairMempoolSdk::Put` 将 CPU IOV 交给 `DramToMP`，通过 `pace_copy_batch_async` 和 `pace_synchronize` 搬运数据。公开 stub 不作为真实传输依据。

目前 `deps/http.bzl` 仍固定 2026-04-29 的 client RPM 和配套 server tar。已重新核对缓存 RPM 的 SHA256 为 `8a50e27c6c009bb2e9d55c7ff44ccef53268cc0b67559b95fd7e22221f1e9600`，与声明一致；包内头文件有 CPU/GPU IOV 和当前使用的接口，ELF 内 `TairMempoolSdk::NormalPut` 可见真实 PACE copy/synchronize 调用。这些是静态证据，不证明旧包具备目标 SDK 的全部收尾修复。

**制品阻塞项：尚未取得上述固定版本组合对应的新 client/server URL 和 SHA256，因此没有更新依赖声明，也没有触发构建。**不能把此次源码适配视为依赖升级完成。

## 静态验收范围

补充用例覆盖 HOST 直接入口、拷贝成功/失败入口、CPU/GPU resolver 选择、多 IOV、packed 多 group/scale/物理大小、共享 HOST pin 的完成/失败/拒绝/停机/未提交释放、SDK 超时 drain、写入 mask 及 actual URI 回填。原有 GPU、多 URI 和独立 pool 用例继续保留。

仅做静态检查。**未编译、未运行测试、smoke、benchmark 或 CI。**真实 PACE、GPU 回归、TP 多 rank、超时释放及性能未验证；worker 收尾依赖目标制品的线程池 stop/join 契约。
