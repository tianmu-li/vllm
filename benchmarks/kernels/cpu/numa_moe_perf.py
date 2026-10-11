# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped perf-stat collection for CPU NUMA MoE replay variants."""

from __future__ import annotations

import errno
import os
import select
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any


class PerfCounters:
    """Collect system-wide perf counters for a bounded CPU set and time window."""

    ACK_TIMEOUT_SECONDS = 10.0
    STOP_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        output_prefix: Path,
        cpus: tuple[int, ...],
        events: list[str],
    ) -> None:
        if not cpus or any(cpu < 0 for cpu in cpus):
            raise ValueError("perf CPU set must contain nonnegative CPU IDs")
        if not events or any(not event for event in events):
            raise ValueError("perf event list must contain nonempty event names")
        if len(set(cpus)) != len(cpus):
            raise ValueError("perf CPU set must not contain duplicates")
        self.output_prefix = Path(output_prefix)
        self.cpus = cpus
        self.events = events
        self.csv_path = self.output_prefix.with_name(
            f"{self.output_prefix.name}.perf.csv"
        )
        self.stderr_path = self.output_prefix.with_name(
            f"{self.output_prefix.name}.perf.stderr"
        )
        self.command: list[str] = []
        self.result: dict[str, Any] = {
            "raw_csv_path": str(self.csv_path),
            "stderr_path": str(self.stderr_path),
            "command": self.command,
            "pid": None,
            "enabled_wall_seconds": None,
            "parsed_counters": [],
            "status": "not_started",
        }
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None
        self._ack_fd: int | None = None
        self._control_fd: int | None = None
        self._ack_buffer = b""
        self._started_at: float | None = None
        self._disable_acknowledged = False
        self._fifo_paths: list[Path] = []

    def __enter__(self) -> PerfCounters:
        perf = shutil.which("perf")
        if perf is None:
            raise RuntimeError("perf executable is unavailable")
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists() or self.stderr_path.exists():
            raise FileExistsError(
                "perf output already exists: "
                f"{self.csv_path if self.csv_path.exists() else self.stderr_path}"
            )
        fifo_id = f"{os.getpid()}.{uuid.uuid4().hex}"
        control_path = self.output_prefix.with_name(
            f"{self.output_prefix.name}.{fifo_id}.ctl"
        )
        ack_path = self.output_prefix.with_name(
            f"{self.output_prefix.name}.{fifo_id}.ack"
        )
        try:
            for path in (control_path, ack_path):
                os.mkfifo(path, 0o600)
                self._fifo_paths.append(path)
            self.command = [
                perf,
                "stat",
                "-a",
                "-C",
                ",".join(str(cpu) for cpu in self.cpus),
                "-D",
                "-1",
                "--control",
                f"fifo:{control_path},{ack_path}",
                "-x",
                ";",
                "-o",
                str(self.csv_path),
                "-e",
                ",".join(self.events),
                "--",
                "sleep",
                "3600",
            ]
            self.result["command"] = self.command
            self._stderr_file = self.stderr_path.open("wb")
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
                start_new_session=True,
            )
            self.result["pid"] = self._process.pid
            self._open_control(control_path, ack_path)
            self._send_control("enable")
            self._started_at = time.monotonic()
            self.result["status"] = "running"
            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        failure: BaseException | None = None
        if self._started_at is not None:
            self.result["enabled_wall_seconds"] = max(
                0.0, time.monotonic() - self._started_at
            )
            try:
                self._send_control("disable")
                self._disable_acknowledged = True
            except BaseException as error:
                failure = error
        self._cleanup()
        try:
            self._finish_result()
        except Exception as error:
            self.result["status"] = "unavailable"
            self.result["report_error"] = str(error)
            if exc_type is None:
                raise RuntimeError(
                    f"Unable to finalize perf counters: {error}"
                ) from error
        if failure is not None and exc_type is None:
            raise RuntimeError(
                f"Unable to disable perf counters: {failure}"
            ) from failure
        if self.result["status"] != "measured" and exc_type is None:
            raise RuntimeError(
                f"perf counter measurement did not complete: {self.result}"
            )
        return False

    def _open_control(self, control_path: Path, ack_path: Path) -> None:
        deadline = time.monotonic() + self.ACK_TIMEOUT_SECONDS
        self._ack_fd = os.open(ack_path, os.O_RDONLY | os.O_NONBLOCK)
        while self._control_fd is None:
            try:
                self._control_fd = os.open(control_path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
                self._check_startup(deadline)
                time.sleep(0.01)
        self._check_startup(deadline)

    def _check_startup(self, deadline: float) -> None:
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError(
                "perf exited before its control FIFO became ready: "
                f"{self._stderr_contents()}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out opening perf control FIFO")

    def _send_control(self, command: str) -> None:
        if self._control_fd is None or self._ack_fd is None:
            raise RuntimeError("perf control FIFOs are not open")
        os.write(self._control_fd, f"{command}\n".encode())
        deadline = time.monotonic() + self.ACK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self._ack_fd], [], [], 0.1)
            if not readable:
                if self._process is not None and self._process.poll() is not None:
                    break
                continue
            chunk = os.read(self._ack_fd, 4096)
            if not chunk:
                if self._process is not None and self._process.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            self._ack_buffer += chunk
            while b"\n" in self._ack_buffer:
                line, self._ack_buffer = self._ack_buffer.split(b"\n", 1)
                response = line.strip(b"\x00\r \t")
                if response == b"ack":
                    return
                if response:
                    raise RuntimeError(
                        "unexpected perf control response: "
                        f"{response.decode(errors='replace')}"
                    )
        raise TimeoutError(
            f"timed out waiting for perf {command} acknowledgement: "
            f"{self._stderr_contents()}"
        )

    def _cleanup(self) -> None:
        process = self._process
        if process is not None:
            self._signal_process_group(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=self.STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._signal_process_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=self.STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    self._signal_process_group(process.pid, signal.SIGKILL)
                    process.wait()
            self._wait_for_group_exit(process.pid)
        for fd_name in ("_control_fd", "_ack_fd"):
            fd = getattr(self, fd_name)
            if fd is not None:
                try:
                    os.close(fd)
                finally:
                    setattr(self, fd_name, None)
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        for path in self._fifo_paths:
            with suppress(FileNotFoundError):
                path.unlink()
        self._fifo_paths.clear()

    def _signal_process_group(self, process_group: int, signum: int) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process_group, signum)

    def _wait_for_group_exit(self, process_group: int) -> None:
        deadline = time.monotonic() + self.STOP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            time.sleep(0.01)
        self._signal_process_group(process_group, signal.SIGTERM)
        time.sleep(0.05)
        self._signal_process_group(process_group, signal.SIGKILL)

    def _stderr_contents(self) -> str:
        if self._stderr_file is not None:
            self._stderr_file.flush()
        try:
            return self.stderr_path.read_text(errors="replace")
        except OSError:
            return ""

    def _finish_result(self) -> None:
        process = self._process
        returncode = process.wait() if process is not None else None
        self.result["returncode"] = returncode
        self.result["parsed_counters"] = self._parse_csv()
        counters = self.result["parsed_counters"]
        missing_events = [
            event
            for event in self.events
            if not any(
                self._event_matches(event, str(counter["event"]))
                for counter in counters
            )
        ]
        self.result["missing_events"] = missing_events
        unsupported = any(
            str(counter["value"]).strip().lower()
            in {"<not counted>", "<not supported>", ""}
            for counter in counters
        )
        expected_exit = returncode == 0 or (
            returncode == -signal.SIGINT and self._disable_acknowledged
        )
        if expected_exit and counters and not unsupported and not missing_events:
            self.result["status"] = "measured"
        else:
            self.result["status"] = "unavailable"

    @staticmethod
    def _event_matches(requested: str, observed: str) -> bool:
        requested = requested.strip().lower()
        observed = observed.strip().lower()
        if requested == observed:
            return True
        requested_pmu, separator, requested_config = requested.partition("/")
        observed_pmu, observed_separator, observed_config = observed.partition("/")
        if not separator or not observed_separator:
            return False
        if requested_config != observed_config:
            return False
        if observed_pmu == requested_pmu:
            return True
        instance = observed_pmu.removeprefix(f"{requested_pmu}_")
        return instance.isdecimal()

    def _parse_csv(self) -> list[dict[str, str | None]]:
        if not self.csv_path.exists():
            return []
        counters = []
        for line in self.csv_path.read_text(errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(";")
            if len(fields) < 3:
                continue
            counters.append(
                {
                    "value": fields[0].strip(),
                    "unit": fields[1].strip() or None,
                    "event": fields[2].strip(),
                    "runtime": fields[3].strip() if len(fields) > 3 else None,
                    "percent_running": (fields[4].strip() if len(fields) > 4 else None),
                }
            )
        return counters
