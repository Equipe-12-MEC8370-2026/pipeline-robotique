"""P2.4a: perception models are pinned; in .venv-perception, both models load and run (skipped elsewhere)."""
import importlib.util
import re

import numpy as np
import pytest

from cta_learn import REPO_ROOT

MODELS_DIR = REPO_ROOT / "data" / "pretrained" / "perception"


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_pm", REPO_ROOT / "scripts" / "perception" / "fetch_perception_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_models_are_pinned_to_official_sources():
    m = _fetch_module().MODELS
    assert set(m) == {"hand_landmarker.task", "sam2.1_hiera_small.pt"}
    for name, info in m.items():
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), name
        assert info["url"].startswith(("https://storage.googleapis.com/mediapipe-models/",
                                       "https://dl.fbaipublicfiles.com/segment_anything_2/"))


@pytest.mark.skipif(not (MODELS_DIR / "hand_landmarker.task").exists(), reason="models not fetched")
def test_local_model_files_match_pins():
    mod = _fetch_module()
    for name, info in mod.MODELS.items():
        assert mod.sha256(MODELS_DIR / name) == info["sha256"], name


def test_hand_landmarker_runs_on_blank_image():
    mp = pytest.importorskip("mediapipe")  # only in .venv-perception
    if not (MODELS_DIR / "hand_landmarker.task").exists():
        pytest.skip("models not fetched")
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
                                 running_mode=RunningMode.IMAGE, num_hands=2)
    with HandLandmarker.create_from_options(opts) as lm:
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8)))
    assert r.hand_landmarks == []  # no hand on a black image


@pytest.mark.gpu
def test_sam2_segments_a_bright_box():
    pytest.importorskip("sam2")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    img = np.full((480, 640, 3), 90, np.uint8)
    img[200:300, 250:420] = 220  # bright rectangle on a grey background
    pred = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(MODELS_DIR / "sam2.1_hiera_small.pt"), device="cuda"))
    with torch.inference_mode():
        pred.set_image(img)
        masks, _, _ = pred.predict(box=np.array([240, 190, 430, 310]), multimask_output=False)
    m = masks[0].astype(bool)
    inter = m[200:300, 250:420].sum()
    assert inter / m.sum() > 0.9 and inter / (100 * 170) > 0.9  # IoU-like: mask ~= the rectangle
