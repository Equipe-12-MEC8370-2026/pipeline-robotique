"""Copy the video measurements of the parts that have no size yet into the task config (interface job step).

    python app/apply_measures.py <task config yaml> <pair>
Reads outputs/p2_4/size_<pair>_<part>.json (scripts/perception/measure_part.py); comments of the config are kept.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402


def main() -> None:
    cfg_path, pair = Path(sys.argv[1]), sys.argv[2]
    cfg = core.load_yaml(cfg_path)
    for pid, o in cfg["objects"].items():
        if o.get("size_m") and o.get("method") not in (None, "video"):
            continue
        d = core.read_json(core.REPO / "outputs" / "p2_4" / f"size_{pair}_{pid}.json")
        if d:
            o["size_m"] = d["size_m_longest_first"]
            o["method"] = o.get("method") or "video"
            print(f"[measure] {pid}: {[round(v * 1000, 1) for v in o['size_m']]} mm (video, IoU {d['mean_iou']})")
        elif not o.get("size_m"):
            sys.exit(f"{pid}: no size (no CAD, no typed dimensions, no video measurement)")
    core.save_yaml(cfg_path, cfg)


if __name__ == "__main__":
    main()
