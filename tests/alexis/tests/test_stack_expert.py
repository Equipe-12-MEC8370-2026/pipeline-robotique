"""P3.2 scripted expert: plan derived from the TaskSpec layout + phase machine with ideal tracking."""
import math

import numpy as np
import yaml

from cta_learn import CONFIGS_DIR
from cta_learn.datagen.stack_expert import StackExpert, StackParams, bearing, plan_from_layout, rot2, wrap
from cta_learn.retarget.profile_stack import stack_success

LAYOUT = yaml.safe_load((CONFIGS_DIR / "tasks" / "so101_profile_stack.yaml").read_text(encoding="utf-8"))
DT = 1 / 30
TCP0 = np.array([0.195, 0.021, 0.071])


def quat_yaw(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def run_ideal(expert, short, long_, params=None, miss_first=False, max_steps=4000):
    """Ideal tracking: the TCP lands on the command; a grasped part follows the TCP (keeping its grasp offset)
    and turns with the gripper. ``miss_first``: the first grasp of the short part slips (part stays down)."""
    parts = {"profile_short": [np.array(short[:3], float), short[3]], "profile_long": [np.array(long_[:3], float), long_[3]]}
    tcp = TCP0.copy()
    expert.reset(tcp, {k: (v[0], v[1]) for k, v in parts.items()}, params, arm_q=np.zeros(5))
    held, rel, missed, phases = None, None, False, []
    for _ in range(max_steps):
        obs = {k: (v[0].copy(), v[1], np.array([0.0, 0.0, 1.0])) for k, v in parts.items()}  # as in the sim: + up axis
        cmd = expert.step(tcp, obs, DT)  # arm_q=None: the ready phase tracks its own joint command
        phases.append((expert.step_index, cmd.phase))
        if cmd.joint_goal is not None:
            continue
        tcp = cmd.tcp_goal.copy()
        obj = expert.plan.obj
        if cmd.phase == "close" and held is None:
            if miss_first and expert.regrasps == 0 and obj == "profile_short":
                missed = True  # the whole first close slips: the part stays on the table
            else:
                held = obj
                rel = (rot2(-cmd.yaw, (parts[obj][0] - tcp)[:2]), parts[obj][0][2] - tcp[2], parts[obj][1] - cmd.yaw)
        if held and cmd.phase in ("release", "clear", "slide", "retreat", "approach"):
            held = None
        if held:
            parts[held][0] = np.r_[tcp[:2] + rot2(cmd.yaw, rel[0]), tcp[2] + rel[1]]
            parts[held][1] = cmd.yaw + rel[2]
        if expert.done or expert.failed:
            break
    return parts, tcp, phases


def check_success(parts, tcp):
    s, l_ = parts["profile_short"], parts["profile_long"]
    # parts placed at their release height drop by place_drop: settle them on their supports
    s_pos = s[0].copy(); s_pos[2] = 0.0125
    l_pos = l_[0].copy(); l_pos[2] = 0.025 + 0.0375
    ok, checks = stack_success(s_pos[None], quat_yaw(s[1])[None], l_pos[None], quat_yaw(l_[1])[None], tcp[None],
                               np.array([LAYOUT["goals"]["profile_short"]["xyyaw"][:2]]), LAYOUT, np)
    return bool(ok[0]), {k: bool(v[0]) for k, v in checks.items()}


def test_plan_follows_taskspec_order_and_geometry():
    plans = plan_from_layout(LAYOUT)
    assert [p.obj for p in plans] == LAYOUT["order"] == ["profile_short", "profile_long"]
    short, long_ = plans
    assert short.support is None and long_.support == "profile_short"
    assert math.isclose(short.symmetry, math.pi) and math.isclose(short.grip_yaw, math.pi / 2)  # across the bar
    assert math.isclose(long_.symmetry, math.pi / 2)
    # fingertips (10 mm under the TCP) clear the part's top before descending, and the placed stack after release
    assert long_.approach_z - 0.010 >= long_.height + 0.008 - 1e-9 and long_.approach_z <= 0.095
    # the upright part is left by sliding the open gripper out sideways (fingers beside the part, ~5 cm finger
    # plates), not by rising straight up; the lying part by rising until the fingertips clear its top
    assert short.slide == 0.0 and short.retreat_z - 0.010 > short.height + 0.01
    assert long_.slide >= long_.height * 0 + 0.025 / 2 + 0.04
    # TCP far enough from both part centres after the final retreat (success needs > 5 cm)
    assert math.hypot(long_.slide, long_.retreat_z - (0.025 + 0.0375)) > 0.05
    assert math.isclose(long_.target_yaw, math.radians(LAYOUT["goals"]["profile_long"]["yaw_rel_deg"]))


def test_wrap():
    assert math.isclose(wrap(math.radians(100), math.pi), math.radians(-80))
    assert math.isclose(wrap(math.radians(50), math.pi / 2), math.radians(-40))


def test_ideal_run_reaches_the_stacked_goal():
    expert = StackExpert(plan_from_layout(LAYOUT))
    parts, tcp, phases = run_ideal(expert, (0.28, -0.03, 0.0125, math.radians(20)), (0.19, -0.11, 0.0375, math.radians(-30)))
    assert expert.done and not expert.failed, expert.failure_reason
    ok, checks = check_success(parts, tcp)
    assert ok, checks
    # the short part ends at the goal yaw (bar symmetric under 180 deg)
    assert abs(wrap(parts["profile_short"][1] - LAYOUT["goals"]["profile_short"]["xyyaw"][2], math.pi)) < 1e-6
    # the long part's yaw relative to the short part = the demo's (square section: modulo 90 deg)
    rel = parts["profile_long"][1] - parts["profile_short"][1]
    assert abs(wrap(rel - math.radians(LAYOUT["goals"]["profile_long"]["yaw_rel_deg"]), math.pi / 2)) < 1e-6
    # order: every short-part phase happens before the first long-part phase
    first_long = next(i for i, (k, _) in enumerate(phases) if k == 1)
    assert all(k == 0 for k, _ in phases[:first_long])


def test_long_goal_follows_where_the_short_part_really_is():
    expert = StackExpert(plan_from_layout(LAYOUT))
    expert.reset(TCP0, {"profile_short": (np.array([0.20, 0.11, 0.0125]), 0.3), "profile_long": (np.array([0.19, -0.11, 0.0375]), 0.0)})
    expert.step_index = 1
    expert._grasp = {"xy": np.array([0.19, -0.11]), "bottom": 0.0, "part_yaw": 0.0, "yaw": 0.0}
    place = expert._make_place({"profile_short": (np.array([0.20, 0.11, 0.0125]), 0.3),
                                "profile_long": (np.array([0.19, -0.11, 0.0375]), 0.0)})
    off = LAYOUT["goals"]["profile_long"]["offset_xy_m"]
    assert np.allclose(place["xy"], np.array([0.20, 0.11]) + rot2(0.3, off))
    assert math.isclose(place["z"], 0.025 + expert.plan.grasp_z + expert.params.place_drop)


def test_random_params_and_starts_succeed_with_ideal_tracking():
    rng = np.random.default_rng(0)
    for _ in range(20):
        so, lo = LAYOUT["objects"]["profile_short"], LAYOUT["objects"]["profile_long"]
        s = [so["start_xyyaw"][k] + rng.uniform(-so["range"][k], so["range"][k]) for k in range(3)]
        l_ = [lo["start_xyyaw"][k] + rng.uniform(-lo["range"][k], lo["range"][k]) for k in range(3)]
        expert = StackExpert(plan_from_layout(LAYOUT))
        parts, tcp, _ = run_ideal(expert, (s[0], s[1], 0.0125, s[2]), (l_[0], l_[1], 0.0375, l_[2]), StackParams.sample(rng))
        assert expert.done, expert.failure_reason
        assert check_success(parts, tcp)[0]


def test_missed_grasp_is_retried():
    expert = StackExpert(plan_from_layout(LAYOUT))
    parts, tcp, _ = run_ideal(expert, (0.28, -0.03, 0.0125, 0.0), (0.19, -0.11, 0.0375, 0.0), miss_first=True)
    assert expert.regrasps == 1 and expert.done
    assert check_success(parts, tcp)[0]



def test_upright_part_released_by_clearance_then_sideways_slide_toward_the_base():
    expert = StackExpert(plan_from_layout(LAYOUT))
    parts, tcp, phases = run_ideal(expert, (0.28, -0.03, 0.0125, 0.2), (0.19, -0.11, 0.0375, -0.4))
    seq = [ph for k, ph in phases if k == 1]
    i_rel, i_clear, i_slide = seq.index("release"), seq.index("clear"), seq.index("slide")
    i_ret = seq.index("retreat", i_slide)  # (the first "retreat" with index 1 is the end of step 0)
    assert i_rel < i_clear < i_slide < i_ret
    assert not {"clear", "slide"} & {ph for k, ph in phases if k == 0}   # the lying part: straight up
    pl, r = expert._place, expert._rel
    closing = np.array([math.cos(pl["yaw"]), math.sin(pl["yaw"])])
    # 1) the fixed finger (gripper -x side of the part) moves away from the part: gripper along -closing axis
    assert np.allclose(r["clear_xy"] - r["xy"], -expert.params.clear_dist * closing)
    # 2) then slides in the plane of the fingers, toward the robot base
    move = r["out_xy"] - r["clear_xy"]
    assert abs(move @ closing) < 1e-9 and move @ r["xy"] < 0
    assert np.isclose(np.linalg.norm(move), expert.plan.slide)
    assert np.allclose(tcp[:2], r["out_xy"]) and np.isclose(tcp[2], pl["retreat"])


def test_carried_part_is_aimed_by_its_bottom_not_by_the_tcp():
    """A part held 4 mm off-centre in the fingers still lands on its target (closed loop on the part)."""
    prm = StackParams(grasp_offset_xy=(0.004, -0.003))
    expert = StackExpert(plan_from_layout(LAYOUT))
    parts, tcp, _ = run_ideal(expert, (0.28, -0.03, 0.0125, 0.3), (0.19, -0.11, 0.0375, 0.2), params=prm)
    assert expert.done
    goal_short = np.array(LAYOUT["goals"]["profile_short"]["xyyaw"][:2])
    assert np.linalg.norm(parts["profile_short"][0][:2] - goal_short) < 5e-4
    s_pos, s_yaw = parts["profile_short"][0], parts["profile_short"][1]
    target_long = s_pos[:2] + rot2(s_yaw, LAYOUT["goals"]["profile_long"]["offset_xy_m"])
    assert np.linalg.norm(parts["profile_long"][0][:2] - target_long) < 5e-4


def test_grasp_and_place_yaws_keep_the_wrist_roll_small():
    """The closing axis is chosen among the part's symmetric equivalents closest to the arm's bearing."""
    expert = StackExpert(plan_from_layout(LAYOUT))
    parts = {"profile_short": (np.array([0.28, -0.03, 0.0125]), 0.2),
             "profile_long": (np.array([0.18, -0.13, 0.0375]), math.radians(35))}   # the failing P3.3 case
    expert.reset(TCP0, parts)
    expert.step_index = 1
    g = expert._grasp_yaw(parts)
    b = bearing(parts["profile_long"][0])
    assert abs(g - b) <= math.radians(45) + 1e-9
    assert abs(wrap(g - math.radians(35), math.pi / 2)) < 1e-9       # still aligned with the part's faces
    expert._grasp = {"xy": parts["profile_long"][0][:2], "bottom": 0.0, "part_yaw": math.radians(35), "yaw": g}
    pl = expert._make_place(parts)
    assert abs(pl["yaw"] - math.atan2(pl["xy"][1], pl["xy"][0])) <= math.radians(45) + 1e-9
    # the part ends at the target yaw (modulo its symmetry): short yaw + 14 deg
    final_part = math.radians(35) + (pl["yaw"] - g)
    assert abs(wrap(final_part - (0.2 + math.radians(14)), math.pi / 2)) < 1e-9


def test_gripper_never_turns_more_than_a_quarter_turn_in_transport():
    """The lying bar at yaw ~0 (failing P3.3 case: 180 deg spin up to the wrist roll limit) and random starts."""
    rng = np.random.default_rng(3)
    cases = [((0.2994, -0.047, -0.0021), (0.2058, -0.0936, 0.5282)), ((0.2987, -0.0425, -0.1345), (0.1758, -0.1279, 0.5948))]
    cases += [((0.28 + rng.uniform(-.025, .025), -0.03 + rng.uniform(-.03, .03), rng.uniform(-.52, .52)),
               (0.19 + rng.uniform(-.02, .02), -0.11 + rng.uniform(-.025, .025), rng.uniform(-.79, .79))) for _ in range(40)]
    for s, l_ in cases:
        expert = StackExpert(plan_from_layout(LAYOUT))
        parts = {"profile_short": (np.array([s[0], s[1], 0.0125]), s[2]), "profile_long": (np.array([l_[0], l_[1], 0.0375]), l_[2])}
        expert.reset(TCP0, parts)
        for k in (0, 1):
            expert.step_index = k
            obj = expert.plan.obj
            g = expert._grasp_yaw(parts)
            expert._grasp = {"xy": parts[obj][0][:2], "bottom": 0.0, "part_yaw": parts[obj][1], "yaw": g}
            pl = expert._make_place(parts)
            assert abs(pl["yaw"] - g) <= math.pi / 2 + 1e-9, (s, l_, k)
            if k == 1:   # the long part: grasp within 45 deg of the arm's bearing
                assert abs(wrap(g - bearing(parts[obj][0]), 2 * math.pi)) <= math.pi / 4 + 1e-9
            # final part yaw correct modulo symmetry
            target = LAYOUT["goals"]["profile_short"]["xyyaw"][2] if k == 0 else s[2] + math.radians(14)
            assert abs(wrap(parts[obj][1] + pl["yaw"] - g - target, expert.plan.symmetry)) < 1e-9
