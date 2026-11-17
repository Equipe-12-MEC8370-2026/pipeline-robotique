#!/usr/bin/env bash
# D12 trial, whole chain (Git Bash, from cta-learn/): seeds -> Mimic generation -> render with cameras -> LeRobot
# dataset -> ACT training -> checkpoint check -> 50-trial evaluation. Logs in outputs/p3_mimic/.
# Also measures Mimic's success rate with 1 and 3 seed demos (40 attempts each), in parallel with the conversion.
# Usage: bash scripts/pipelines/p3_mimic_trial.sh [N_GENERATED] [VERSION_SUFFIX]   (default 200, none)
# A suffix (e.g. _v2) keeps a previous run: every output name gets it; the seeds are shared.
set -euo pipefail
N=${1:-200}
V=${2:-}
SIM="../isaacsim-env/Scripts/python.exe"
LEARN="./.venv-learn/Scripts/python.exe"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
L=outputs/p3_mimic${V}
mkdir -p "$L" data/mimic
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/chain.txt"; }

stamp "1/8 seeds (10 expert demos)"
[ -f data/mimic/p3_seeds.hdf5 ] || $SIM scripts/sim/record_mimic_seeds.py --headless --seeds 10 --max_attempts 20 \
  --out data/mimic/p3_seeds.hdf5 > "$L/1_seeds.txt" 2>&1
grep -hE "SEEDS" "$L/1_seeds.txt" 2>/dev/null | tee -a "$L/chain.txt" || echo "seeds reused: data/mimic/p3_seeds.hdf5" | tee -a "$L/chain.txt"

stamp "2/8 Mimic generation: $N successes from 10 seeds"
[ -f data/mimic/p3_gen10${V}.hdf5 ] || $SIM scripts/sim/mimic_generate.py --headless --input data/mimic/p3_seeds.hdf5 \
  --output data/mimic/p3_gen10${V}.hdf5 --num "$N" > "$L/2_generate.txt" 2>&1
grep -E "MIMIC_RESULT" "$L/2_generate.txt" | tee -a "$L/chain.txt"

stamp "3/8 render with cameras"
[ -d data/raw/p3_mimic${V} ] || $SIM scripts/sim/render_mimic_demos.py --headless --enable_cameras \
  --input data/mimic/p3_gen10${V}.hdf5 --out data/raw/p3_mimic${V} --report "$L/3_render.json" > "$L/3_render.txt" 2>&1
grep -E "RENDER_SUMMARY" "$L/3_render.txt" | tee -a "$L/chain.txt"

stamp "4/8 LeRobot conversion (+ in parallel: Mimic success rate with 1 and 3 seeds)"
(
  $SIM scripts/sim/mimic_generate.py --headless --input data/mimic/p3_seeds.hdf5 --output data/mimic/p3_gen1_rate${V}.hdf5 \
    --num 40 --by_attempts --select_demos demo_0 > "$L/4_rate1.txt" 2>&1 || true
  $SIM scripts/sim/mimic_generate.py --headless --input data/mimic/p3_seeds.hdf5 --output data/mimic/p3_gen3_rate${V}.hdf5 \
    --num 40 --by_attempts --select_demos demo_0 demo_1 demo_2 > "$L/4_rate3.txt" 2>&1 || true
) &
RATE_PID=$!
[ -d data/datasets/p3_mimic${V} ] || $LEARN scripts/data/raw_to_lerobot.py --raw data/raw/p3_mimic${V} --out data/datasets/p3_mimic${V} \
  --repo_id cta/p3_mimic${V} > "$L/4_convert.txt" 2>&1
wait $RATE_PID || true
grep -hE "MIMIC_RESULT" "$L/4_rate1.txt" "$L/4_rate3.txt" | sed 's/^/rate: /' | tee -a "$L/chain.txt" || true

stamp "5/8 dataset check"
$LEARN scripts/data/check_dataset.py --root data/datasets/p3_mimic${V} --raw data/raw/p3_mimic${V} --repo_id cta/p3_mimic${V} \
  --video 0 --out "$L" 2>&1 | grep -E "PASS|FAIL" | tee -a "$L/chain.txt"

stamp "6/8 ACT smoke test + training (100K steps)"
$LEARN scripts/train/train_act.py --config configs/pipelines/train_act_p3_mimic${V}.yaml --smoke > "$L/6_smoke.txt" 2>&1
grep -E "SMOKE" "$L/6_smoke.txt" | tee -a "$L/chain.txt"
[ -d outputs/train/act_p3_mimic${V} ] || $LEARN scripts/train/train_act.py --config configs/pipelines/train_act_p3_mimic${V}.yaml \
  > "$L/6_train.txt" 2>&1
grep -E "End of training" outputs/train/act_p3_mimic${V}.log | tee -a "$L/chain.txt"

stamp "7/8 checkpoint check"
$LEARN scripts/train/check_checkpoint.py --ckpt outputs/train/act_p3_mimic${V}/checkpoints/last/pretrained_model \
  --root data/datasets/p3_mimic${V} --episodes 0 40 80 120 --frames 0 150 300 450 600 \
  --out "$L/checkpoint_check.json" 2>&1 | grep -E "PASS|FAIL|CHECKPOINT" | tee -a "$L/chain.txt"

stamp "8/8 evaluation, 50 trials (seed 7), n_action_steps 25"
$LEARN scripts/eval/policy_server.py --ckpt outputs/train/act_p3_mimic${V}/checkpoints/last/pretrained_model \
  --n_action_steps 25 > "$L/8_server.txt" 2>&1 &
for _ in $(seq 1 60); do grep -q READY "$L/8_server.txt" && break; sleep 5; done
$SIM scripts/sim/eval_profile_policy.py --headless --enable_cameras --agent policy --tag act_p3_mimic${V}_n25 \
  --shutdown_server > "$L/8_eval.txt" 2>&1
$LEARN -c "import json; r=json.load(open('outputs/p3_5/act_p3_mimic${V}_n25/report.json')); \
print({k: (v['successes'], v['n']) for k, v in r['summary'].items()}, r.get('paired_with_expert'))" | tee -a "$L/chain.txt"
stamp "DONE"
