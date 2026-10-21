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

The automatic runtime uses the pinned PACE Meta/Server RPMs in
`deps/http.bzl`. `//3rdparty/pace_test_runtime:runtime` extracts the binaries and
`libtent_shared.so.5` into test runfiles for both x86_64 and aarch64.
The test environment must provide their
ELF dependencies (including the compatible glibc, Boost, Prometheus, glog,
OpenSSL, hiredis, MySQL client and jemalloc libraries), `rpm2cpio`, and `cpio`.
The Meta and Server RPMs are downloaded from `rtp-maga/pace/server/`.
The Meta and Server RPMs use the TH endpoint first, with the standard OSS
endpoint as an alternative.
Already-downloaded release RPMs can also be supplied with Bazel's
`--distdir=/path/to/release-rpms` for offline builds.
The CUDA13 SDK RPMs use `2026_10_25_15_21` for x86_64 and aarch64. The KVCM
server archives come from CI run `83066354`, built from internal commit
`99e071b0` and external commit `a297b3f9`. Each architecture uses the original
server tarball extracted from its corresponding production image, mirrored to
`rtp-maga/kv_cache_manager/server/ci-83066354/` with a pinned SHA256.
Both Manager binaries require glibc 2.32 or newer, libstdc++ with
`GLIBCXX_3.4.26`, `libibverbs.so.1`, and `librdmacm.so.1` in the test environment.
That CUDA SDK initializes a CUDA device even for the CPU-buffer contract, so
run it in the normal GPU test environment.

The Meta RPMs come from CI run `79257320`, source commit `b4fb6b6b`: x86_64 uses
`3.1.21_2026_10_13_15_39`, and aarch64 uses `3.1.21_2026_10_13_15_45`.
Both PACE Server architectures use the CSSDV Server RPMs from CI run
`83025525`, source commit `26b89c97`, mirrored to
`rtp-maga/pace/server/ci-83025525/` with pinned SHA256 checksums.
The Manager archives provide x86_64 and aarch64 binaries from the same CI run.
Meta, PACE Server, and KVCM Server are selected for the configured architecture.

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
