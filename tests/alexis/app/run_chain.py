"""Run a job's steps in order, detached from the interface (started by core.start_job).

    python app/run_chain.py app_data/jobs/<task>/<kind>.json

Each step's output is appended to the job's log. A step with "background": true is started and left running
(e.g. the policy server); the chain waits for its "ready" text in the log, and kills it at the end.
Status in the job file: running -> done | failed (+ returncode, failed step).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    jp = Path(sys.argv[1])
    st = json.loads(jp.read_text(encoding="utf-8"))
    log = Path(st["log"])
    bg: list[subprocess.Popen] = []

    def save(**kw):
        cur = json.loads(jp.read_text(encoding="utf-8"))
        cur.update(kw)
        jp.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    status, rc = "done", 0
    save(pid=os.getpid(), status="running")
    try:
        with open(log, "a", encoding="utf-8", errors="replace") as f:
            for i, step in enumerate(st["steps"]):
                save(step=i, step_name=step["name"])
                f.write(f"\n===== [{i + 1}/{len(st['steps'])}] {step['name']} =====\n")
                f.flush()
                env = {**os.environ, "PYTHONUNBUFFERED": "1", **step.get("env", {})}
                p = subprocess.Popen(step["cmd"], cwd=step.get("cwd"), env=env, stdout=f, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
                if step.get("background"):
                    bg.append(p)
                    t0 = time.time()
                    while step.get("ready") and step["ready"] not in log.read_text(encoding="utf-8", errors="replace"):
                        if p.poll() is not None or time.time() - t0 > step.get("ready_timeout", 600):
                            status, rc = "failed", p.poll() if p.poll() is not None else -1
                            break
                        time.sleep(1)
                    if status == "failed":
                        save(failed_step=step["name"])
                        break
                    continue
                rc = p.wait()
                if rc != 0:
                    status = "failed"
                    save(failed_step=step["name"])
                    break
    except Exception as exc:  # noqa: BLE001
        status, rc = "failed", -1
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n[run_chain] error: {exc!r}\n")
    finally:
        for p in bg:
            if p.poll() is None:
                p.kill()
        save(status=status, returncode=rc, ended=time.time())


if __name__ == "__main__":
    main()
