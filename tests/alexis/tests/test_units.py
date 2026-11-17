import math

import numpy as np
import pytest

from cta_learn.core import (
    UnitRangeError,
    check_dataset_stats,
    convert,
    from_rad,
    load_embodiment,
    to_rad,
)

EMB = load_embodiment("so101")


def _random_rad(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(EMB.lower, EMB.upper, size=(n, EMB.num_joints))


@pytest.mark.parametrize("fmt", ["rad", "deg", "lerobot"])
def test_round_trip_is_identity(fmt):
    q = _random_rad()
    back = to_rad(from_rad(q, fmt, EMB), fmt, EMB)
    np.testing.assert_allclose(back, q, atol=1e-12)


def test_lerobot_mapping_endpoints_match_leisaac_convention():
    # Joint limits map to the ends of the LeRobot range; gripper uses [0, 100].
    lo_norm = from_rad(EMB.lower, "lerobot", EMB)
    hi_norm = from_rad(EMB.upper, "lerobot", EMB)
    np.testing.assert_allclose(lo_norm, [-100, -100, -100, -100, -100, 0])
    np.testing.assert_allclose(hi_norm, [100, 100, 100, 100, 100, 100])
    # shoulder_pan limits are symmetric: 0 rad -> 0
    assert from_rad(np.zeros(6), "lerobot", EMB)[0] == pytest.approx(0.0)


def test_lerobot_matches_leisaac_formula_on_a_known_value():
    # LeIsaac: deg = (norm - motor_lo) / motor_range * joint_range + joint_lo
    # elbow_flex: norm 50 -> 0.75 * 190 - 100 = 42.5 deg
    norm = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 50.0])
    q = to_rad(norm, "lerobot", EMB)
    assert math.degrees(q[2]) == pytest.approx(42.5)
    # gripper: norm 50 -> 0.5 * 110 - 10 = 45 deg
    assert math.degrees(q[5]) == pytest.approx(45.0)


def test_convert_deg_to_lerobot_goes_through_validated_radians():
    deg = np.degrees(_random_rad(10))
    np.testing.assert_allclose(convert(deg, "deg", "lerobot", EMB), from_rad(np.radians(deg), "lerobot", EMB))


def test_wrong_vector_length_is_rejected():
    with pytest.raises(ValueError, match="6 joints"):
        to_rad(np.zeros(5), "rad", EMB)


def test_out_of_limit_radians_are_rejected():
    q = np.zeros(6)
    q[1] = math.radians(150)  # shoulder_lift limit is 100 deg
    with pytest.raises(UnitRangeError, match="shoulder_lift"):
        from_rad(q, "lerobot", EMB)


def test_regression_degrees_dataset_declared_as_lerobot_is_caught():
    """Regression for the previous SO-101 attempt.

    Stats of community dataset Jeevesh2009/so101_gray_block_midvar_test (old LeRobot
    calibration, degrees). They were treated as normalized [-100, 100] values, so the
    policy saw out-of-distribution states and the arm dove into the table.
    """
    stats_min = [-82.7, 69.1, 65.9, 36.4, -51.9, 0.6]
    stats_max = [37.7, 189.8, 176.3, 94.2, 50.8, 52.1]
    with pytest.raises(UnitRangeError) as err:
        check_dataset_stats(stats_min, stats_max, "lerobot", EMB)
    msg = str(err.value)
    assert "shoulder_lift" in msg and "elbow_flex" in msg


def test_plausible_lerobot_stats_pass():
    check_dataset_stats([-41.4, -53.2, -100.0, -30.0, 0.0, 9.1], [24.5, 11.1, 81.3, 100.0, 69.5, 57.9], "lerobot", EMB)


def test_batch_shapes_are_preserved():
    q = _random_rad(24).reshape(2, 12, 6)
    assert from_rad(q, "lerobot", EMB).shape == (2, 12, 6)
