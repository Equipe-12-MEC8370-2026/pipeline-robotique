"""P2.4a — download the two perception models from their official sources, check their sha256, write a manifest.

    .venv-perception\\Scripts\\python.exe scripts\\perception\\fetch_perception_models.py
-> data/pretrained/perception/{hand_landmarker.task, sam2.1_hiera_small.pt, SOURCE.json}

Neither Google nor Meta publishes checksums for these files: the sha256 below were recorded at the first
download (2026-11-13) and are enforced from then on, so a silently changed file is refused.
Already-present files are only re-checked, never re-downloaded.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cta_learn import REPO_ROOT

DEST = REPO_ROOT / "data" / "pretrained" / "perception"
MODELS = {
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "publisher": "Google MediaPipe (Hand Landmarker, float16, version 1)",
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    },
    "sam2.1_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "publisher": "Meta FAIR (SAM 2.1 Hiera-Small, release 2024-09-28; URL from facebookresearch/sam2 "
                     "checkpoints/download_ckpts.sh at commit 2b90b9f5)",
        "sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cta-learn-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while b := r.read(1 << 20):
            f.write(b)
    tmp.replace(dst)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest, ok = {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}, True
    for name, m in MODELS.items():
        dst = DEST / name
        if not dst.exists():
            print(f"[fetch] downloading {name} from {m['url']}")
            download(m["url"], dst)
        digest = sha256(dst)
        good = m["sha256"] is None or digest == m["sha256"]
        ok &= good
        print(f"[fetch] {name}: {dst.stat().st_size / 1e6:.1f} MB sha256 {digest} "
              f"{'(recorded now: pin it in this script)' if m['sha256'] is None else ('OK' if good else 'MISMATCH')}")
        manifest["files"][name] = {"url": m["url"], "publisher": m["publisher"], "bytes": dst.stat().st_size,
                                   "sha256": digest, "pinned": m["sha256"] is not None}
    (DEST / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
