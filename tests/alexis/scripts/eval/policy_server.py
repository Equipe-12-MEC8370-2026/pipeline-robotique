r"""Serve a trained LeRobot checkpoint (ACT, SmolVLA, ...) to the simulator (run with .venv-learn, start BEFORE the sim client).

    .\.venv-learn\Scripts\python.exe scripts\eval\policy_server.py
    .\.venv-learn\Scripts\python.exe scripts\eval\policy_server.py --n_action_steps 25
    .\.venv-learn\Scripts\python.exe scripts\eval\policy_server.py --temporal_ensemble 0.01          # ACT only
    .\.venv-learn\Scripts\python.exe scripts\eval\policy_server.py --ckpt outputs\train\smolvla_p1\checkpoints\last\pretrained_model

The policy type comes from the checkpoint. Language-conditioned policies use the training dataset's
task text unless --task is given. Listens on 127.0.0.1:50765 (cta_learn.eval.protocol). Stops when a
client sends "shutdown".
"""

import argparse

from cta_learn import REPO_ROOT
from cta_learn.eval.protocol import DEFAULT_ADDRESS, serve
from cta_learn.learn.lerobot_policy import LeRobotPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/train/act_p0/checkpoints/last/pretrained_model")
    p.add_argument("--n_action_steps", type=int, default=None, help="replan every k steps (default: checkpoint's)")
    p.add_argument("--temporal_ensemble", type=float, default=None, help="ACT temporal ensembling coefficient")
    p.add_argument("--task", default=None, help="language instruction (default: the training dataset's task)")
    p.add_argument("--port", type=int, default=DEFAULT_ADDRESS[1])
    a = p.parse_args()
    ckpt = REPO_ROOT / a.ckpt
    policy = LeRobotPolicy(ckpt, n_action_steps=a.n_action_steps, temporal_ensemble_coeff=a.temporal_ensemble,
                           task=a.task)
    print(f"[server] {policy.info()}", flush=True)
    serve(policy.handler, ("127.0.0.1", a.port), on_ready=lambda addr: print(f"[server] READY on {addr}", flush=True))
    print(f"[server] shutdown after {policy.calls} act calls", flush=True)


if __name__ == "__main__":
    main()
