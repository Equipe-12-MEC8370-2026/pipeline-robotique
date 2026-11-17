"""Action smoothness of trained policies vs the expert (offline, .venv-learn; HF_HUB_OFFLINE set inside)."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
from cta_learn import REPO_ROOT
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class

CK = {"act": "outputs/train/act_p0/checkpoints/last/pretrained_model",
      "smolvla": "outputs/train/smolvla_p1/checkpoints/last/pretrained_model"}
EPS, FRAMES, H = [0, 37, 75, 112, 149], [30, 90, 150, 210], 50
ds = LeRobotDataset("cta/p0_pickplace", root=REPO_ROOT / "data/datasets/p0_pickplace", episodes=EPS, video_backend="pyav",
                    delta_timestamps={"action": [i / 30 for i in range(H + 50)]})
ep_idx = np.asarray(ds.hf_dataset["episode_index"])
items = []
for ep in EPS:
    rows = np.flatnonzero(ep_idx == ep)
    for f in FRAMES:
        if f + 50 < len(rows):
            items.append((ds[int(rows[f])], ds[int(rows[f + 25])], ds[int(rows[f + 50])]))


def hf(a):  # high-frequency content: mean |2nd difference| per step, degrees
    return np.rad2deg(np.abs(np.diff(a, n=2, axis=0))).mean()


exp = [hf(it[0]["action"][:H].numpy()) for it in items]
print(f"expert           2nd-diff {np.mean(exp):.3f} deg/step^2")
dev = "cuda"
for name, ck in CK.items():
    ck = REPO_ROOT / ck
    pol = get_policy_class(PreTrainedConfig.from_pretrained(ck).type).from_pretrained(ck).to(dev).eval()
    pre, post = make_pre_post_processors(pol.config, pretrained_path=str(ck), preprocessor_overrides={"device_processor": {"device": dev}})

    def chunk(item, seed):
        torch.manual_seed(seed)
        obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
        obs["task"] = [item["task"]]
        with torch.no_grad():
            return post(pol.predict_action_chunk(pre(obs))).squeeze(0).float().cpu().numpy()[:H]

    s2, spread, jump = [], [], {25: [], 50: []}
    for it0, it25, it50 in items:
        a, b = chunk(it0, 0), chunk(it0, 1)
        s2.append(hf(a))
        spread.append(np.rad2deg(np.abs(a - b)).mean())  # same image, other random seed
        for n, itn in ((25, it25), (50, it50)):
            nxt = chunk(itn, 2)
            prev_end = a[n - 1] if n <= H else a[-1]
            jump[n].append(np.rad2deg(np.abs(nxt[0] - prev_end)).mean())  # step at a replanning boundary
    print(f"{name:8s} 2nd-diff {np.mean(s2):.3f} deg/step^2 | same obs, other seed: {np.mean(spread):.3f} deg | "
          f"jump at replan n25 {np.mean(jump[25]):.2f} deg, n50 {np.mean(jump[50]):.2f} deg")
    del pol
    torch.cuda.empty_cache()
gt_jump = np.mean([np.rad2deg(np.abs(it[0]["action"][1].numpy() - it[0]["action"][0].numpy())).mean() for it in items])
print(f"expert one-step change (reference for jumps) {gt_jump:.2f} deg")
