"""D12 trial: amplify seed demos with Isaac Lab Mimic in CTA-SO101-ProfileStack-Mimic-v0 (no cameras, 1 env).

Same flow as IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py (unchanged library code:
setup_env_config, setup_async_generation, env_loop), with our task registered by cta_learn.sim.tasks.
Successful generated demos only -> --output (Isaac Lab HDF5: actions, initial_state, states). Render them with
cameras afterwards: scripts/sim/render_mimic_demos.py.

    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\sim\\mimic_generate.py --headless `
        --input data\\mimic\\p3_seeds.hdf5 --output data\\mimic\\p3_generated.hdf5 --num 200
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num", type=int, default=10, help="successful demos to generate (attempts with --by_attempts)")
parser.add_argument("--by_attempts", action="store_true", help="stop after --num attempts (to measure the success rate)")
parser.add_argument("--max_failures", type=int, default=None)
parser.add_argument("--keep_failed", action="store_true", help="also export failed attempts (<output>_failed.hdf5) for diagnosis")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--select_demos", type=str, nargs="*", default=None, help="use only these seed demos (demo_0 ...)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import asyncio  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_mimic.datagen import generation as G  # noqa: E402
from isaaclab_mimic.datagen.utils import setup_output_paths  # noqa: E402

import cta_learn.sim.tasks  # noqa: E402,F401
from cta_learn import REPO_ROOT  # noqa: E402
from cta_learn.sim.tasks.profile_stack_mimic import TASK_ID  # noqa: E402


def resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


out_dir, out_name = setup_output_paths(resolve(args.output))
env_cfg, success_term = G.setup_env_config(env_name=TASK_ID, output_dir=out_dir, output_file_name=out_name, num_envs=1,
                                           device=args.device, generation_num_trials=args.num)
env_cfg.datagen_config.seed = args.seed
env_cfg.datagen_config.generation_guarantee = not args.by_attempts
if args.keep_failed:
    from isaaclab.managers import DatasetExportMode
    env_cfg.datagen_config.generation_keep_failed = True
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
if args.max_failures is not None:
    env_cfg.datagen_config.max_num_failures = args.max_failures
env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset()

input_file = resolve(args.input)
if args.select_demos:  # keep only some seed demos: copy them into a temporary file
    import h5py
    sub = Path(out_dir) / f"_seeds_{'_'.join(args.select_demos)}.hdf5"
    with h5py.File(input_file, "r") as src, h5py.File(sub, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        g = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
        for name in args.select_demos:
            src.copy(src["data"][name], g, name)
    input_file = str(sub)

t0 = time.time()
comp = G.setup_async_generation(env=env, num_envs=1, input_file=input_file, success_term=success_term)
tasks = asyncio.ensure_future(asyncio.gather(*comp["tasks"]))
try:
    G.env_loop(env, comp["reset_queue"], comp["action_queue"], comp["info_pool"], comp["event_loop"], data_gen_tasks=tasks)
finally:
    tasks.cancel()
    try:
        comp["event_loop"].run_until_complete(tasks)
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    print(f"MIMIC_RESULT successes={G.num_success} attempts={G.num_attempts} "
          f"rate={100 * G.num_success / max(G.num_attempts, 1):.1f}% elapsed={time.time() - t0:.0f}s -> {resolve(args.output)}",
          flush=True)
    env.close()
app.close()
