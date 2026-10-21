import fcntl
import io
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from rtp_llm.test.smoke.local_pace_server import LocalPaceServer


class LocalPaceServerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.shm_dir = self.directory / "shm"
        self.shm_dir.mkdir()
        self.socket_dir = self.directory / "sockets"
        self.socket_dir.mkdir()
        self.events = []
        self.meta = Mock(pid=41001)
        self.data = Mock(pid=41002)
        self.meta.poll.return_value = None
        self.data.poll.return_value = None

        def local_path(*parts):
            path = Path(*parts)
            if path == Path("/dev/shm"):
                return self.shm_dir
            if Path("/dev/shm") in path.parents:
                return self.shm_dir / path.relative_to("/dev/shm")
            return path

        self.patch("Path", side_effect=local_path)
        self.patch("tempfile.mkdtemp", return_value=str(self.socket_dir))
        self.ports = self.patch("PortsContext")
        self.ports.return_value.__enter__.return_value = [12001, 12002, 12003, 12004, 12005, 12006]
        self.popen = self.patch("subprocess.Popen", side_effect=self.spawn)
        self.http = self.patch("urlopen", side_effect=self.response)
        self.killpg = self.patch("os.killpg", side_effect=ProcessLookupError)
        self.patch("time.sleep")
        binary = patch.object(LocalPaceServer, "_binary", side_effect=lambda name: "/runtime/" + name)
        binary.start()
        self.addCleanup(binary.stop)
        self.server = LocalPaceServer(self.directory)
        self.addCleanup(self.server.close)

    def patch(self, name, **kwargs):
        patcher = patch("rtp_llm.test.smoke.local_pace_server." + name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def spawn(self, args, **kwargs):
        if args[0].endswith("tair_mempool_meta_service"):
            self.events.append("meta started")
            return self.meta
        self.events.append("data started")
        for name in ("shm_mps", "shm_mps_meta"):
            (self.shm_dir / name).write_text("test-owned", encoding="utf-8")
        for name in ("mp.sock", "mp_data.sock"):
            (self.socket_dir / name).touch()
        return self.data

    def response(self, url, **kwargs):
        if url.endswith("/v1/api/healthy"):
            self.events.append("meta ready")
            return io.StringIO(json.dumps({"status": "success"}))
        if url.endswith("/api/segment"):
            return io.StringIO(json.dumps({"data": [{"virt_segment_id": 1}]}))
        self.events.append("data ready")
        return io.StringIO(json.dumps({"status": "success", "data": {"total_bytes": 1024}}))

    def test_combined_role_starts_after_meta_and_supplies_local_client_settings(self):
        with patch.dict("os.environ", {"LD_LIBRARY_PATH": "/test-sdk/libs"}):
            fixture = self.server.start()
        self.assertEqual(self.events, ["meta started", "meta ready", "data started", "data ready"])
        self.assertEqual(self.popen.call_count, 2)
        meta_args = self.popen.call_args_list[0].args[0]
        data_call = self.popen.call_args_list[1]
        data_args = data_call.args[0]
        self.assertEqual(meta_args[meta_args.index("-b") + 1], "memory")
        self.assertEqual(data_args[data_args.index("--role") + 1], "consumer|provider")
        self.assertIn("--no_rdma", data_args)
        self.assertEqual(data_args[data_args.index("--tiering_mode") + 1], "off")
        self.assertEqual(data_args[data_args.index("-R") + 1], "0")
        self.assertEqual(data_args[data_args.index("--metaservice") + 1], "http://127.0.0.1:12001")
        self.assertTrue(all(entry.kwargs["start_new_session"] for entry in self.popen.call_args_list))
        library_dirs = data_call.kwargs["env"]["LD_LIBRARY_PATH"].split(":")
        self.assertLess(library_dirs.index("/usr/lib64"), library_dirs.index("/test-sdk/libs"))
        client_env = fixture.client_env()
        self.assertEqual(client_env["TAIR_MEMPOOL_ENABLE_TENT"], "0")
        self.assertEqual(client_env["TAIR_MEMPOOL_SOCK"], str(self.socket_dir / "mp.sock"))
        self.assertEqual(client_env["TAIR_MEMPOOL_DATA_SOCK"], str(self.socket_dir / "mp_data.sock"))
        for name in ("TAIR_MEMPOOL_SOCK", "TAIR_MEMPOOL_DATA_SOCK"):
            self.assertEqual(data_call.kwargs["env"][name], client_env[name])
        self.assertEqual(fixture.config["provider_status_urls"], ["http://127.0.0.1:12005/api/status"])
        self.assertIs(self.server.start(), fixture)
        self.assertEqual(self.popen.call_count, 2)

    def test_meta_readiness_failure_rolls_back_before_starting_data(self):
        self.meta.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited before ready"):
            self.server.start()
        self.assertEqual(self.events, ["meta started"])
        self.http.assert_not_called()
        self.killpg.assert_called_once_with(self.meta.pid, signal.SIGTERM)
        self.meta.wait.assert_called_once_with(timeout=5)
        self.ports.return_value.__exit__.assert_called_once_with(None, None, None)
        self.assertIsNone(self.server._lock)
        self.assertFalse(self.socket_dir.exists())
        self.assertTrue(self.popen.call_args.kwargs["stdout"].closed)

    def test_sigterm_unwinds_the_test_and_close_restores_the_previous_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        self.server.start()
        with self.assertRaises(SystemExit) as caught:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
        self.server.close()
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_failed_data_stop_still_stops_meta_and_can_be_retried(self):
        self.server.start()
        self.killpg.side_effect = [PermissionError(), ProcessLookupError()]
        with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "still running"):
            self.server.close()
        self.assertEqual(self.server._processes, [("server", self.data)])
        self.meta.wait.assert_called_once_with(timeout=5)
        self.data.wait.assert_not_called()
        self.assertTrue((self.shm_dir / "shm_mps").exists())
        self.assertIsNotNone(self.server._lock)
        self.killpg.side_effect = ProcessLookupError
        self.server.close()
        self.assertFalse((self.shm_dir / "shm_mps").exists())

    def test_data_readiness_requires_allocated_memory_and_both_client_sockets(self):
        self.server._socket_dir = self.socket_dir
        (self.socket_dir / "mp.sock").touch()

        def data_status(url, **kwargs):
            if url.endswith("/api/segment"):
                return io.StringIO(json.dumps({"data": [{"virt_segment_id": 1}]}))
            attempt = self.http.call_count
            if attempt == 3:
                (self.socket_dir / "mp_data.sock").touch()
            capacity = 0 if attempt == 1 else 1024
            return io.StringIO(json.dumps({"status": "success", "data": {"total_bytes": capacity}}))

        self.http.side_effect = data_status
        with patch(
            "rtp_llm.test.smoke.local_pace_server.time.monotonic",
            side_effect=[0, 0.1, 0.2, 1],
        ):
            self.server._wait_ready(self.data, "http://127.0.0.1/api/status", 0.3, data_server=True)
        self.assertEqual(self.http.call_count, 4)

    def test_data_readiness_waits_for_meta_to_assign_virtual_segments(self):
        self.server._socket_dir = self.socket_dir
        for name in ("mp.sock", "mp_data.sock"):
            (self.socket_dir / name).touch()
        self.http.side_effect = [
            io.StringIO(json.dumps({"status": "success", "data": {"total_bytes": 1024}})),
            io.StringIO(json.dumps({"data": [{"virt_segment_id": 0xFFFFFFFF}]})),
            io.StringIO(json.dumps({"status": "success", "data": {"total_bytes": 1024}})),
            io.StringIO(json.dumps({"data": [{"virt_segment_id": 1}]})),
        ]
        with patch(
            "rtp_llm.test.smoke.local_pace_server.time.monotonic", side_effect=[0, 0.1],
        ):
            self.server._wait_ready(self.data, "http://127.0.0.1/api/status", 1, data_server=True)
        self.assertEqual(self.http.call_count, 4)

    def test_data_readiness_failure_rolls_back_both_processes_and_shared_memory(self):
        self.data.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited before ready"):
            self.server.start()
        self.assertEqual(
            self.killpg.call_args_list,
            [call(self.data.pid, signal.SIGTERM), call(self.meta.pid, signal.SIGTERM)],
        )
        self.data.wait.assert_called_once_with(timeout=5)
        self.meta.wait.assert_called_once_with(timeout=5)
        self.assertFalse((self.shm_dir / "shm_mps").exists())
        self.assertFalse((self.shm_dir / "shm_mps_meta").exists())
        self.assertIsNone(self.server._lock)
        self.assertIsNone(self.server.fixture)
        self.assertTrue(all(entry.kwargs["stdout"].closed for entry in self.popen.call_args_list))

    def test_existing_shared_memory_is_preserved_and_no_process_is_started(self):
        existing = self.shm_dir / "shm_mps_meta"
        existing.write_text("existing service", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Existing PACE shared memory"):
            self.server.start()
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing service")
        self.popen.assert_not_called()
        self.ports.assert_not_called()
        self.killpg.assert_not_called()
        self.assertIsNone(self.server._lock)

    def test_existing_test_lock_rejects_start_and_preserves_the_owner_lock(self):
        with (self.shm_dir / "rtp_llm_pace_test.lock").open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another local PACE test"):
                self.server.start()
            with (self.shm_dir / "rtp_llm_pace_test.lock").open("a+") as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.popen.assert_not_called()
        self.ports.assert_not_called()
        self.assertIsNone(self.server._lock)

    def test_close_is_idempotent_and_removes_only_owned_shared_memory(self):
        unrelated = self.shm_dir / "another-service"
        unrelated.write_text("unrelated", encoding="utf-8")
        self.server.start()
        replaced = self.shm_dir / "shm_mps"
        replaced.rename(self.shm_dir / "old-shm-mapping")
        replaced.write_text("replacement", encoding="utf-8")
        self.server.close()
        self.server.close()
        self.assertEqual(replaced.read_text(encoding="utf-8"), "replacement")
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "unrelated")
        self.assertFalse((self.shm_dir / "shm_mps_meta").exists())
        self.assertFalse(self.socket_dir.exists())
        self.assertEqual(
            self.killpg.call_args_list,
            [call(self.data.pid, signal.SIGTERM), call(self.meta.pid, signal.SIGTERM)],
        )
        self.ports.return_value.__exit__.assert_called_once_with(None, None, None)
        self.assertTrue(all(entry.kwargs["stdout"].closed for entry in self.popen.call_args_list))
        self.assertIsNone(self.server.fixture)
        with (self.shm_dir / "rtp_llm_pace_test.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
