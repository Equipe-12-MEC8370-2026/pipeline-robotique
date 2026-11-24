"""Shared helpers of the CTA interface: paths, tasks, YAML that keeps its comments, background jobs, log parsing.

The interface only drives the pipeline's scripts (no pipeline logic here). Long work (demo generation, training,
evaluation) runs as a DETACHED job (app/run_chain.py): it keeps going when the browser or the interface is closed,
and the pages re-read its log to show progress.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
from ruamel.yaml import YAML

REPO = Path(__file__).resolve().parents[1]
APP_DATA = REPO / "app_data"
JOBS = APP_DATA / "jobs"
CONFIGS = REPO / "configs"
PY_LEARN = REPO / ".venv-learn" / "Scripts" / "python.exe"
PY_SIM = REPO.parent / "isaacsim-env" / "Scripts" / "python.exe"
PY_PERC = REPO / ".venv-perception" / "Scripts" / "python.exe"
PY_APP = Path(sys.executable)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120


# ---------------------------------------------------------------- YAML (round trip: comments kept)

def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return _yaml.load(f)


def save_yaml(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- tasks

def task_configs() -> dict[str, Path]:
    """{task_id: configs/tasks/<file>.yaml} for every task created in the interface."""
    out = {}
    for p in sorted((CONFIGS / "tasks").glob("*.yaml")):
        if p.name.startswith("so101_"):
            continue
        try:
            d = load_yaml(p)
        except Exception:  # noqa: BLE001
            continue
        # only the tasks created in the interface (``created_in_app: true``): the project's reference tasks
        # (profiles, angle block) stay out of the client's view
        if isinstance(d, dict) and d.get("task_id") and isinstance(d.get("source"), dict) and d.get("created_in_app"):
            out[str(d["task_id"])] = p
    return out


def pair_suffix(pair: str) -> str:
    return pair.rsplit("_", 1)[-1]


def slug(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t or "tache"


def create_task(name_fr: str, name_en: str = "") -> str:
    """New empty task config (no video, no part yet): configs/tasks/cta_<slug>.yaml. Returns its task_id."""
    tid = "cta_" + slug(name_fr)[:40]
    k, base = 2, tid
    while (CONFIGS / "tasks" / f"{tid}.yaml").exists():
        tid, k = f"{base}_{k}", k + 1
    data = _yaml.load(f"""# Task created in the CTA interface ({time.strftime('%Y-%m-%d')}).
source: {{pair: null}}
task_id: {tid}
created_in_app: true
instruction:   # written by a human (validated at the TaskSpec review)
  fr: ""
  en: ""
objects: {{}}
demos: []
""")
    data["instruction"]["fr"], data["instruction"]["en"] = name_fr, name_en or name_fr
    save_yaml(CONFIGS / "tasks" / f"{tid}.yaml", data)
    return tid


class Task:
    def __init__(self, task_id: str, cfg_path: Path):
        self.id, self.cfg_path = task_id, cfg_path
        self.cfg = load_yaml(cfg_path)
        if self.cfg.get("objects") is None:
            self.cfg["objects"] = {}
        pair = self.cfg["source"].get("pair")
        self.pair = str(pair) if pair else None
        self.taskspec = CONFIGS / "taskspecs" / f"{task_id}_{pair_suffix(self.pair) if self.pair else 'none'}.json"
        self.adapt = CONFIGS / "tasks" / f"so101_adapt_{task_id}.yaml"
        self.layout = CONFIGS / "tasks" / f"so101_{task_id}.yaml"
        self.robot_id = f"{task_id}_so101"
        self.raw = REPO / "data" / "raw" / f"{task_id}_app"
        self.dataset = REPO / "data" / "datasets" / f"{task_id}_app"
        self.cad_dir = REPO / "data" / "cad" / task_id
        self.upload_dir = REPO / "data" / "uploads" / task_id
        self.data = APP_DATA / task_id
        self.eval_dir = REPO / "outputs" / "taskspec_eval" / self.robot_id

    def demos(self) -> list[dict]:
        """Demonstration video pairs of the task: [{"pair", "fpv", "tpv"}]. Older configs: their source pair."""
        d = self.cfg.get("demos")
        if d:
            return [dict(x) for x in d]
        return [{"pair": self.pair}] if self.pair else []

    def set_demos(self, demos: list[dict]) -> None:
        self.cfg["demos"] = demos
        first = next((x.get("pair") for x in demos if x.get("pair")), None)
        self.cfg["source"]["pair"] = first
        self.save()

    @property
    def parts(self) -> list[str]:
        return list(self.cfg["objects"].keys())

    def reload(self) -> None:
        self.cfg = load_yaml(self.cfg_path)

    def save(self) -> None:
        save_yaml(self.cfg_path, self.cfg)

    # approvals (human review steps), kept in app_data
    def approvals(self) -> dict:
        return read_json(self.data / "approvals.json", {})

    def set_approval(self, key: str, value: bool, note: str = "") -> None:
        a = self.approvals()
        a[key] = {"approved": value, "at": time.strftime("%Y-%m-%d %H:%M"), "note": note}
        write_json(self.data / "approvals.json", a)

    def env(self) -> dict:
        """Environment of the simulation scripts for this task."""
        return {"PYTHONPATH": str(REPO / "src"), "CTA_LAYOUT": str(self.layout)}


# ---------------------------------------------------------------- jobs

def job_paths(task_id: str, kind: str) -> tuple[Path, Path]:
    d = JOBS / task_id
    return d / f"{kind}.json", d / f"{kind}.log"


def job_state(task_id: str, kind: str) -> dict | None:
    jp, _ = job_paths(task_id, kind)
    st = read_json(jp)
    if not st:
        return None
    if st.get("status") == "running" and not _alive(st.get("pid")):
        st["status"] = "stopped"   # killed with its console, the PC restarted, ...
    if st.get("status") == "starting":
        st["status"] = "running" if time.time() - st.get("started", 0) < 60 else "stopped"
    return st


def job_log(task_id: str, kind: str, max_chars: int = 400_000) -> str:
    _, lp = job_paths(task_id, kind)
    try:
        with open(lp, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_chars))
            return f.read()
    except OSError:
        return ""


def _alive(pid) -> bool:
    try:
        return bool(pid) and psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def running_jobs() -> list[tuple[str, str]]:
    out = []
    if JOBS.exists():
        for jp in JOBS.glob("*/*.json"):
            st = job_state(jp.parent.name, jp.stem)
            if st and st.get("status") == "running":
                out.append((jp.parent.name, jp.stem))
    return out


def start_job(task_id: str, kind: str, title: str, steps: list[dict]) -> None:
    """steps: [{"name", "cmd": [...], "env": {...}, "background": bool, "ready": "text"}]; run in order by
    app/run_chain.py, detached from the interface."""
    jp, lp = job_paths(task_id, kind)
    jp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("", encoding="utf-8")
    write_json(jp, {"title": title, "status": "starting", "steps": steps, "step": 0, "started": time.time(),
                    "log": str(lp)})
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    # run_chain.py writes its own pid and status (no read-modify-write race with a job that ends at once)
    subprocess.Popen([str(PY_APP), str(Path(__file__).with_name("run_chain.py")), str(jp)], cwd=str(REPO),
                     creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)


def stop_job(task_id: str, kind: str) -> None:
    st = job_state(task_id, kind)
    if not st or not st.get("pid"):
        return
    try:
        root = psutil.Process(int(st["pid"]))
        for p in root.children(recursive=True) + [root]:
            try:
                p.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    st = read_json(job_paths(task_id, kind)[0])
    st["status"] = "stopped"
    write_json(job_paths(task_id, kind)[0], st)


# ---------------------------------------------------------------- log parsing

_ATTEMPT = re.compile(r"^\[attempt\s+(\d+)\]\s+(.+?)\s+steps=\s*(\d+).*?successes=(\d+)/(\d+)", re.M)
_TRIAL = re.compile(r"^\[trial\s+(\d+)\s+(\S+)\s*\]\s+(\S+).*?successes=(\d+)/(\d+) of (\d+)", re.M)
_TQDM = re.compile(r"Training:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*)")
_LOSS = re.compile(r"step:(\d+(?:\.\d+)?)([KMB]?)\s.*?loss:([0-9.eE+-]+)")


def parse_generation(log: str) -> dict:
    rows = _ATTEMPT.findall(log)
    if not rows:
        return {"attempts": 0, "successes": 0, "failures": []}
    last = rows[-1]
    fails = [(int(a), o) for a, o, *_ in rows if o.strip() != "success"]
    return {"attempts": int(last[4]), "successes": int(last[3]), "failures": fails[-10:]}


def parse_eval(log: str) -> dict:
    rows = _TRIAL.findall(log)
    if not rows:
        return {"done": 0, "total": 0, "successes": 0, "outcomes": {}}
    last = rows[-1]
    out: dict[str, int] = {}
    for r in rows:
        out[r[2]] = out.get(r[2], 0) + 1
    return {"done": int(last[4]), "total": int(last[5]), "successes": int(last[3]), "outcomes": out}


def parse_training(log: str) -> dict:
    m = None
    for m in _TQDM.finditer(log):
        pass
    losses = []
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}
    for s, suf, loss in _LOSS.findall(log):
        try:
            losses.append((float(s) * mult[suf], float(loss)))
        except ValueError:
            pass
    losses = sorted(dict(losses).items())   # one point per step (a step can be logged twice)
    if not m:
        return {"step": int(losses[-1][0]) if losses else 0, "total": 0, "elapsed": "", "remaining": "", "losses": losses}
    return {"step": int(m.group(1)), "total": int(m.group(2)), "elapsed": m.group(3).strip(),
            "remaining": m.group(4).strip(), "losses": losses}


def tail_lines(log: str, n: int = 12, keep=None) -> str:
    lines = [ln for ln in log.replace("\r", "\n").splitlines() if ln.strip()]
    if keep:
        lines = [ln for ln in lines if keep(ln)]
    return "\n".join(lines[-n:])


def ffmpeg() -> str:
    """ffmpeg executable (PATH, else the WinGet install used by the project, see cta_learn.video.ingest.ffbin)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    hits = sorted(Path(os.environ.get("LOCALAPPDATA", "")).glob("Microsoft/WinGet/Packages/Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
    return str(hits[-1]) if hits else "ffmpeg"
