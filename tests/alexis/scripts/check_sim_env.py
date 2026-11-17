"""Check that the simulation runtime (isaacsim-env) sees cta_learn without installing it.

Usage (PowerShell, from the repo root):
    $env:PYTHONPATH = "$PWD\\src"; & ..\\isaacsim-env\\Scripts\\python.exe scripts\\check_sim_env.py
"""

import sys
from importlib.metadata import version

import cta_learn
from cta_learn.core import from_rad, load_embodiment

emb = load_embodiment("so101")
print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"cta_learn   {cta_learn.__version__}  from {cta_learn.__file__}")
print(f"embodiment  {emb.name}: {emb.joint_names}")
print(f"lerobot(0)  {from_rad([0.0] * emb.num_joints, 'lerobot', emb).round(2).tolist()}")
for pkg in ("isaacsim", "isaaclab", "torch", "numpy"):
    print(f"{pkg:<11} {version(pkg)}")
print("OK")
