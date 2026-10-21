# Automatic local PACE smoke

PACE-specific tests (`PACE_REQUIRED=true`) start a private memory-backed Meta
and one foreground `consumer|provider` data process when `KVCM_PACE_FIXTURE` is
unset. The existing KVCM fixture then configures PACE storage and starts KVCM;
the SDK contract and model smoke use the same client environment.

This tests local DRAM payload read/write through the real PACE backend. It does
not verify cross-node TCP, RDMA, GDR, or SSD. No extra Docker service, network
namespace, or PACE source change is needed. Do not separately start a Consumer
sidecar in this local mode.
The SDK maps the combined server's shared memory and copies payload through its
local DRAM device. Readiness includes Meta's asynchronous virtual segment assignment.

## Runtime and artifacts

The initial automatic runtime uses the pinned Linux x86_64 Meta/Server RPMs in
`deps/http.bzl`. `//3rdparty/pace_test_runtime:runtime` extracts the binaries and
`libtent_shared.so.4` into test runfiles. The test environment must provide their
ELF dependencies (including the compatible glibc, Boost, Prometheus, glog,
OpenSSL, hiredis, MySQL client and jemalloc libraries), `rpm2cpio`, and `cpio`.
The release RPMs are downloaded directly from `rtp-maga/pace/server/` using the
`oss-cn-zhangjiakou.aliyuncs.com` endpoint, with
`cn-zhangjiakou.oss.aliyuncs.com` as an alternative for TH environments.
Already-downloaded release RPMs can also be supplied with Bazel's
`--distdir=/path/to/release-rpms` for offline builds.
The CUDA13 KVCM server and SDK are paired from published CI release `79404320`.
That CUDA SDK initializes a CUDA device even for the CPU-buffer contract, so
run it in the normal GPU test environment.

The runtime allocates a 2 GiB pool including 1 GiB of shared metadata. Its
`/dev/shm` must have enough capacity, with a suitable memlock limit. Only one
local PACE fixture may own the fixed PACE SHM filenames in that environment.
PACE cases use Bazel's `exclusive` tag to run serially. Independently launched
local fixtures fail explicitly on contention, and existing SHM is not overwritten.
Use a dedicated development/test environment for automatic mode.

For a manually supplied runtime, set `--test_env=KVCM_PACE_BIN_DIR=/path/to/bin`.
That directory must contain both service binaries and their corresponding
private shared libraries. No installation or compilation happens inside a test.

## Run

Use the repository build configuration matching the SDK and test machine:

```bash
unset KVCM_PACE_FIXTURE
bazel test --config=cuda13 --config=remote_kv_cache --test_strategy=standalone \
  //rtp_llm/test/smoke:remote_cache_pace_contract

# Requires the model files and GPU expected by this existing smoke case.
bazel test --config=cuda13 --config=remote_kv_cache --test_strategy=standalone \
  //rtp_llm/test/smoke:remote_cache_pace_basic
```

Keep `--test_env=KVCM_PACE_FIXTURE=/path/to/fixture.json` to attach an existing
PACE environment instead. Its processes remain caller-owned; their topology
and transport are described by that supplied fixture.

Local processes use test-owned process groups and stop in reverse order when
the test finishes, startup fails, or SIGTERM unwinds the test. `stop_server()` intentionally still stops
only KVCM for the existing `kill_remote` fault case; `close()` releases the
complete test environment. Meta/data service logs are saved under
`kvcm_logs/pace_logs` in the test's undeclared outputs.
