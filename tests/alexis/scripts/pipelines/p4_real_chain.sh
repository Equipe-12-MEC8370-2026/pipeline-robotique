#!/usr/bin/env bash
# P4 (2026-11-24): whole chain for the "real-looking" cube task (CTA-SO101-PickPlaceReal-v0), Git Bash, from cta-learn/:
# expert demos -> LeRobot dataset -> dataset check -> ACT smoke test + training -> checkpoint check ->
# 50-trial evaluation in sim (expert reference, then policy with replanning every 25 steps, as P0).
# Logs: outputs/p4_1/chain.txt (+ one file per step). Each step is skipped if its output already exists (resume).
# Usage: bash scripts/pipelines/p4_real_chain.sh [N_DEMOS]   (default 200)
set -euo pipefail
N=${1:-200}
SIM="../isaacsim-env/Scripts/python.exe"
LEARN="./.venv-learn/Scripts/python.exe"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
L=outputs/p4_1
RAW=data/raw/p4_pickplace_real
DS=data/datasets/p4_pickplace_real
RUN=act_p4
mkdir -p "$L"
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/chain.txt"; }
# keep Windows from idle-sleeping while the chain runs (P0 v2 training was cut by sleep at 89 976 steps);
# SetThreadExecutionState only holds while this helper lives: no power setting is changed
$LEARN -c "import ctypes, time
ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
while True: time.sleep(60)" &
AWAKE_PID=$!
trap 'kill $AWAKE_PID 2>/dev/null || true' EXIT

stamp "1/7 expert demos: $N successes (seed 1)"
[ -f "$L/gen_report.json" ] || $SIM scripts/sim/gen_demos.py --headless --enable_cameras --variant real --record \
  --successes "$N" --attempts $((N * 13 / 10)) --seed 1 --out "$RAW" --report "$L/gen_report.json" > "$L/1_gen.txt" 2>&1
$LEARN -c "import json; r=json.load(open('$L/gen_report.json')); print('gen', r['successes'], '/', r['attempts'], r['outcomes'], round(r['elapsed_s']/60), 'min')" | tee -a "$L/chain.txt"

stamp "2/7 LeRobot conversion"
[ -d "$DS" ] || $LEARN scripts/data/raw_to_lerobot.py --raw "$RAW" --out "$DS" --repo_id cta/p4_pickplace_real > "$L/2_convert.txt" 2>&1

stamp "3/7 dataset check"
$LEARN scripts/data/check_dataset.py --root "$DS" --raw "$RAW" --repo_id cta/p4_pickplace_real \
  --video 0 $((N / 2)) $((N - 1)) --out "$L" 2>&1 | grep -E "PASS|FAIL|OK" | tee -a "$L/chain.txt" || true

stamp "4/7 ACT smoke test + training (100K steps)"
$LEARN scripts/train/train_act.py --config configs/pipelines/train_act_p4.yaml --smoke > "$L/4_smoke.txt" 2>&1
grep -E "SMOKE" "$L/4_smoke.txt" | tee -a "$L/chain.txt" || true
grep -q "SMOKE OK" "$L/4_smoke.txt" || { stamp "smoke test failed: stop"; exit 1; }
[ -d outputs/train/$RUN ] || $LEARN scripts/train/train_act.py --config configs/pipelines/train_act_p4.yaml > "$L/4_train.txt" 2>&1
grep -E "End of training" outputs/train/$RUN.log | tee -a "$L/chain.txt" || true

stamp "5/7 checkpoint check"
$LEARN scripts/train/check_checkpoint.py --ckpt outputs/train/$RUN/checkpoints/last/pretrained_model \
  --root "$DS" --episodes 0 $((N / 4)) $((N / 2)) $((3 * N / 4)) $((N - 1)) --frames 0 60 120 180 240 \
  --out "$L/checkpoint_check.json" 2>&1 | grep -E "PASS|FAIL|CHECKPOINT" | tee -a "$L/chain.txt" || true

stamp "6/7 expert reference, 50 trials (seed 7)"
[ -d "$L/eval/expert" ] || $SIM scripts/sim/eval_policy.py --headless --enable_cameras --variant real --agent expert \
  --out "$L/eval" > "$L/6_eval_expert.txt" 2>&1

stamp "7/7 ACT evaluation, 50 trials (seed 7), n_action_steps 25"
$LEARN scripts/eval/policy_server.py --ckpt outputs/train/$RUN/checkpoints/last/pretrained_model \
  --n_action_steps 25 > "$L/7_server.txt" 2>&1 &
for _ in $(seq 1 60); do grep -q READY "$L/7_server.txt" && break; sleep 5; done
$SIM scripts/sim/eval_policy.py --headless --enable_cameras --variant real --agent policy --tag act_p4_n25 \
  --out "$L/eval" --shutdown_server > "$L/7_eval.txt" 2>&1
for t in expert act_p4_n25; do
  $LEARN -c "import json; r=json.load(open('$L/eval/$t/report.json')); print('$t', {k: (v['successes'], v['n']) for k, v in r['summary'].items()})" | tee -a "$L/chain.txt"
done
stamp "DONE"
