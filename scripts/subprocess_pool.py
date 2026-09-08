from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class WorkerProc:
    proc: subprocess.Popen
    out_q: "queue.Queue[dict[str, Any]]"
    err_q: "queue.Queue[str]"


def _reader_json_lines(stream, out_q: "queue.Queue[dict[str, Any]]") -> None:
    for line in iter(stream.readline, ""):
        line = line.strip()
        if not line:
            continue
        try:
            out_q.put(json.loads(line))
        except Exception:  # noqa: BLE001
            out_q.put({"ok": False, "error": f"failed to parse json: {line[:200]}"})


def _reader_text(stream, err_q: "queue.Queue[str]") -> None:
    for line in iter(stream.readline, ""):
        err_q.put(line.rstrip("\n"))


def start_workers(cmd: list[str], n: int) -> list[WorkerProc]:
    workers: list[WorkerProc] = []
    for _ in range(n):
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        out_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        err_q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=_reader_json_lines, args=(proc.stdout, out_q), daemon=True).start()  # type: ignore[arg-type]
        threading.Thread(target=_reader_text, args=(proc.stderr, err_q), daemon=True).start()  # type: ignore[arg-type]
        workers.append(WorkerProc(proc=proc, out_q=out_q, err_q=err_q))
    return workers


def stop_workers(workers: list[WorkerProc]) -> None:
    for w in workers:
        try:
            if w.proc.stdin:
                w.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
    for w in workers:
        try:
            w.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def submit_task(worker: WorkerProc, task: dict[str, Any]) -> None:
    if worker.proc.stdin is None:
        raise RuntimeError("worker stdin is closed")
    worker.proc.stdin.write(json.dumps(task, ensure_ascii=False) + "\n")
    worker.proc.stdin.flush()


def poll_results(workers: list[WorkerProc], timeout_s: float = 0.1) -> dict[str, Any] | None:
    """
    Poll any available worker result without head-of-line blocking.

    旧实现会对每个 worker 依次做 blocking get(timeout=timeout_s)，当 workers>1 时会导致：
    - 即使某个 worker 已经有结果，也可能被前面的 empty worker 阻塞住；
    - master 取结果过慢，worker stdout pipe 可能被写满，进而让 worker 停在 IO 上；
    - 表现为 CPU 空闲、吞吐显著下降。
    """
    # Fast path: non-blocking sweep
    for w in workers:
        try:
            return w.out_q.get_nowait()
        except queue.Empty:
            continue

    if timeout_s <= 0:
        return None

    # Wait up to timeout_s, polling periodically so we don't block on any single worker.
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        for w in workers:
            try:
                return w.out_q.get_nowait()
            except queue.Empty:
                continue
        time.sleep(0.01)
    return None
