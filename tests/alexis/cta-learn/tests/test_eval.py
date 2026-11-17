"""P0.5 tests: policy-server protocol, ACT handler on the real checkpoint, eval metrics."""

import threading

import numpy as np
import pytest

from cta_learn import REPO_ROOT
from cta_learn.eval import metrics as M
from cta_learn.eval.protocol import PolicyClient, pack, serve, unpack

CKPT = REPO_ROOT / "outputs" / "train" / "act_p0" / "checkpoints" / "last" / "pretrained_model"


# ------------------------------------------------------------------ protocol

def test_pack_unpack_round_trip():
    arrays = {"img": np.arange(24, dtype=np.uint8).reshape(2, 4, 3), "q": np.array([0.1, -2.0], np.float32),
              "empty": np.zeros((0, 3), np.float64)}
    kind, meta, out = unpack(pack("act", {"x": 1}, arrays))
    assert kind == "act" and meta == {"x": 1}
    for k, v in arrays.items():
        assert out[k].dtype == v.dtype and out[k].shape == v.shape and np.array_equal(out[k], v)


def test_unpack_rejects_truncated():
    blob = pack("act", arrays={"q": np.ones(4, np.float32)})
    with pytest.raises(ValueError):
        unpack(blob + b"x")


def _start(handler):
    ready = threading.Event()
    addr = {}

    def on_ready(a):
        addr["a"] = a
        ready.set()

    t = threading.Thread(target=serve, args=(handler, ("127.0.0.1", 0)), kwargs={"on_ready": on_ready}, daemon=True)
    t.start()
    assert ready.wait(10)
    return t, addr["a"]


def test_server_client_dummy_policy():
    calls = {"reset": 0}

    def handler(kind, meta, arrays):
        if kind == "reset":
            calls["reset"] += 1
            return {}, {}
        if kind == "act":
            return {}, {"action": arrays["observation.state"] * 2}
        if kind == "info":
            return {"policy": "dummy"}, {}
        raise ValueError(f"unknown {kind}")

    t, addr = _start(handler)
    with PolicyClient(addr) as pc:  # first client: close keeps the server alive
        assert pc.info() == {"policy": "dummy"}
        pc.reset()
        np.testing.assert_allclose(pc.act({"observation.state": np.array([1, 2, 3], np.float32)}), [2, 4, 6])
        with pytest.raises(RuntimeError, match="unknown"):
            pc.call("bogus")
    pc = PolicyClient(addr)  # second client, then shutdown
    pc.reset()
    pc.close(shutdown=True)
    t.join(5)
    assert not t.is_alive() and calls["reset"] == 2


# ------------------------------------------------------------------ ACT handler (real checkpoint)

@pytest.mark.skipif(not CKPT.exists(), reason="trained ACT checkpoint not present")
def test_act_policy_handler_real_checkpoint():
    from cta_learn.core import load_embodiment
    from cta_learn.learn.act_policy import ActPolicy

    pol = ActPolicy(CKPT, device="cpu", n_action_steps=5)
    info = pol.info()
    assert info["chunk_size"] == 100 and info["n_action_steps"] == 5
    assert info["image_keys"] == ["observation.images.front", "observation.images.wrist"]
    emb = load_embodiment("so101")
    rng = np.random.default_rng(0)
    obs = {"observation.state": ((emb.lower + emb.upper) / 2).astype(np.float32),
           **{k: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for k in info["image_keys"]}}
    acts = [pol.handler("act", {}, obs)[1]["action"] for _ in range(6)]
    assert all(a.shape == (6,) and a.dtype == np.float32 and np.isfinite(a).all() for a in acts)
    assert pol.policy._action_queue.maxlen == 5
    with pytest.raises(ValueError):
        pol.to_batch({**obs, "observation.images.front": obs["observation.images.front"].astype(np.float32)})
    with pytest.raises(KeyError):
        pol.to_batch({"observation.state": obs["observation.state"]})


# ------------------------------------------------------------------ metrics

def test_make_trials_deterministic_and_split():
    a, b = M.make_trials(40, 10, seed=7), M.make_trials(40, 10, seed=7)
    assert a == b and len(a) == 50 and [t["trial"] for t in a] == list(range(50))
    assert a != M.make_trials(40, 10, seed=8)
    for t in a[:40]:
        assert t["split"] == "train_range" and M._inside(t["cube"], M.TRAIN_CUBE) and M._inside(t["target"], M.TRAIN_TARGET)
    for t in a[40:]:
        assert t["split"] == "out_of_range" and not M._inside(t["cube"], M.TRAIN_CUBE)
        assert M._inside(t["cube"], M.OOD_CUBE) and M._inside(t["target"], M.TRAIN_TARGET)


def test_clip_to_limits():
    lo, hi = np.array([-1.0, 0.0]), np.array([1.0, 2.0])
    a, c = M.clip_to_limits(np.array([0.5, 1.0]), lo, hi)
    assert not c and np.allclose(a, [0.5, 1.0]) and a.dtype == np.float32
    a, c = M.clip_to_limits(np.array([1.2, -0.1]), lo, hi)
    assert c and np.allclose(a, [1.0, 0.0])


def test_classify_outcome():
    z0 = np.full(10, 0.015)
    lifted = np.r_[z0[:3], 0.08, 0.08, 0.015]
    tgt = np.array([0.2, 0.1])
    assert M.classify_outcome(True, False, lifted, tgt, tgt, False) == "success"
    assert M.classify_outcome(False, True, lifted, tgt, tgt, False) == "cube_off_table"
    assert M.classify_outcome(False, False, z0, tgt + 0.2, tgt, False) == "never_lifted"
    assert M.classify_outcome(False, False, lifted, tgt + 0.01, tgt, True) == "in_target_no_success"
    assert M.classify_outcome(False, False, lifted, tgt + 0.2, tgt, True) == "held_at_timeout"
    assert M.classify_outcome(False, False, lifted, tgt + 0.2, tgt, False) == "placed_outside"


def test_wilson_and_summary():
    lo, hi = M.wilson_interval(40, 50)
    assert 0.66 < lo < 0.68 and 0.88 < hi < 0.90  # reference values for 40/50
    assert M.wilson_interval(0, 10)[0] == 0.0 and M.wilson_interval(10, 10)[1] == 1.0
    recs = [{"trial": i, "split": "train_range" if i < 4 else "out_of_range",
             "outcome": "success" if i % 2 == 0 else "never_lifted"} for i in range(6)]
    s = M.summarize(recs)
    assert s["all"]["successes"] == 3 and s["train_range"]["n"] == 4 and s["out_of_range"]["successes"] == 1
    exp = [{"trial": i, "outcome": "success"} for i in range(6)]
    assert M.paired_table(recs, exp) == {"policy_ok__expert_ok": 3, "policy_fail__expert_ok": 3}
