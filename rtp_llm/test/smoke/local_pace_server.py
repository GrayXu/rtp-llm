"""A single local DRAM PACE server shared by the SDK and model smoke tests."""

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from rtp_llm.test.smoke.pace_fixture import PaceFixture, runfile
from rtp_llm.test.utils.port_util import PortsContext


class LocalPaceServer:
    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.logs_dir = self.work_dir / "pace_logs"
        self.logs_dir.mkdir()
        self.fixture = None
        self._processes = []
        self._streams = []
        self._ports = None
        self._lock = None
        self._socket_dir = None
        self._shm_files = {}
        self._sigterm_handler = lambda signum, frame: sys.exit(128 + signum)
        self._previous_sigterm_handler = None

    def _binary(self, name):
        directory = os.environ.get("KVCM_PACE_BIN_DIR")
        if directory:
            path = Path(directory) / name
        else:
            path = runfile(
                os.environ["TEST_WORKSPACE"], "3rdparty/pace_test_runtime/" + name
            )
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"PACE test binary is not executable: {path}")
        return str(path.resolve())

    def _spawn(self, name, args, env):
        output = (self.logs_dir / f"{name}.log").open("w")
        self._streams.append(output)
        logging.info("Starting local PACE %s: %s", name, " ".join(args))
        process = subprocess.Popen(
            args, env=env, cwd=self.logs_dir, stdout=output, stderr=output,
            start_new_session=True,
        )
        self._processes.append((name, process))
        return process

    def _remember_shm(self):
        for name in ("shm_mps", "shm_mps_meta"):
            path = Path("/dev/shm") / name
            if path not in self._shm_files and path.exists():
                stat = path.stat()
                self._shm_files[path] = (stat.st_dev, stat.st_ino)

    def _wait_ready(self, process, url, deadline, data_server=False):
        while time.monotonic() < deadline:
            if data_server:
                self._remember_shm()
            if process.poll() is not None:
                raise RuntimeError(f"Local PACE exited before ready; see {self.logs_dir}")
            try:
                with urlopen(url, timeout=1) as response:
                    status = json.load(response)
                if status.get("status") == "success":
                    if not data_server:
                        return
                    if (
                        status.get("data", {}).get("total_bytes", 0) > 0
                        and (self._socket_dir / "mp.sock").exists()
                        and (self._socket_dir / "mp_data.sock").exists()
                    ):
                        # Meta assigns virtual segment IDs asynchronously after
                        # the HTTP server and SDK sockets have started.
                        with urlopen(url.rsplit("/", 1)[0] + "/segment", timeout=1) as response:
                            segments = json.load(response).get("data") or []
                        if segments and all(
                            segment.get("virt_segment_id", 0xFFFFFFFF) != 0xFFFFFFFF
                            for segment in segments
                        ):
                            return
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Local PACE readiness timed out; see {self.logs_dir}")

    def start(self, timeout=120):
        if self.fixture is not None:
            return self.fixture
        meta_bin = self._binary("tair_mempool_meta_service")
        server_bin = self._binary("tair_mempool_server")
        try:
            self._previous_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
            # Production PACE binaries have fixed SHM names. Cooperating tests
            # serialize within the same /dev/shm instead of changing PACE.
            self._lock = Path("/dev/shm/rtp_llm_pace_test.lock").open("a+")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another local PACE test is using this /dev/shm") from exc
            for name in ("shm_mps", "shm_mps_meta"):
                if (Path("/dev/shm") / name).exists():
                    raise RuntimeError(
                        "Existing PACE shared memory found; use a dedicated test "
                        "environment or attach with KVCM_PACE_FIXTURE"
                    )

            self._ports = PortsContext(num_ports=6)
            meta_port, meta_monitor, admin_port, rpc_port, http_port, monitor_port = (
                self._ports.__enter__()
            )
            # Keep Unix socket names below the kernel's sockaddr_un limit even
            # when Bazel's TEST_TMPDIR is long.
            self._socket_dir = Path(tempfile.mkdtemp(prefix="pace-", dir="/tmp"))
            client_env = {
                "TAIR_MEMPOOL_SOCK": str(self._socket_dir / "mp.sock"),
                "TAIR_MEMPOOL_DATA_SOCK": str(self._socket_dir / "mp_data.sock"),
                "TAIR_MEMPOOL_ENABLE_TENT": "0",
                "TAIR_MEMPOOL_ENABLE_GDR_ZERO_COPY": "0",
                "TAIR_MEMPOOL_ENABLE_EXTERNAL_ZERO_COPY": "0",
            }
            env = os.environ.copy()
            env.update(client_env)
            env["ENABLE_TAIR_MEMPOOL_HUGEPAGES"] = "OFF"
            # The SDK's 3FS runfiles carry another OpenSSL build. Services
            # need the OS libraries used by their MySQL/Kerberos dependencies.
            env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
                str(Path(server_bin).parent), str(Path(meta_bin).parent),
                "/usr/lib64",
                env.get("LD_LIBRARY_PATH", ""),
            )))
            domain = f"http://127.0.0.1:{meta_port}"
            status_url = f"http://127.0.0.1:{http_port}/api/status"
            deadline = time.monotonic() + timeout
            meta = self._spawn("meta", [
                meta_bin, "-b", "memory", "-l", "no", "-p", str(meta_port),
                "-M", str(meta_monitor), "-P", "2", "--client_io_threads", "2",
                "--meta_thread_pool_size", "2",
            ], env)
            self._wait_ready(meta, domain + "/v1/api/healthy", deadline)
            data = self._spawn("server", [
                server_bin, "--role", "consumer|provider", "--no_rdma",
                "--tiering_mode", "off",
                "--metaservice", domain, "--metric_type", "local",
                "-s", "2048", "--shm_meta_size_mb", "1024", "-R", "0", "-c",
                "-p", "2", "--ctrl_threads", "2", "-a", str(admin_port),
                "-r", str(rpc_port), "-t", str(http_port), "-m", str(monitor_port),
            ], env)
            self._wait_ready(data, status_url, deadline, data_server=True)
            config = self.work_dir / "pace-fixture.json"
            config.write_text(json.dumps({
                "domain": domain, "provider_status_urls": [status_url],
                "client_env": client_env,
            }), encoding="utf-8")
            self.fixture = PaceFixture(config)
            return self.fixture
        except BaseException:
            try:
                self.close()
            except Exception:
                logging.exception("Failed to clean up PACE after startup failure")
            raise

    def close(self):
        had_data_server = any(name == "server" for name, _ in self._processes)
        if had_data_server:
            self._remember_shm()
        for name, process in reversed(self._processes.copy()):
            logging.info("Stopping local PACE %s process group %d", name, process.pid)
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        process.poll()
                        os.killpg(process.pid, 0)
                        time.sleep(0.1)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                self._processes.remove((name, process))
            except Exception:
                logging.exception("Failed to stop local PACE %s", name)
        if self._processes:
            raise RuntimeError(f"PACE processes are still running; see {self.logs_dir}")
        if had_data_server:
            self._remember_shm()
        if self._previous_sigterm_handler is not None:
            if signal.getsignal(signal.SIGTERM) is self._sigterm_handler:
                signal.signal(signal.SIGTERM, self._previous_sigterm_handler)
            self._previous_sigterm_handler = None
        for output in self._streams:
            output.close()
        self._streams.clear()
        for path, identity in self._shm_files.items():
            if path.exists():
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == identity:
                    path.unlink()
        self._shm_files.clear()
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir)
            self._socket_dir = None
        if self._ports is not None:
            self._ports.__exit__(None, None, None)
            self._ports = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        self.fixture = None
