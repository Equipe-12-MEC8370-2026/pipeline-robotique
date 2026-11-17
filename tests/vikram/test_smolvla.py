"""
Test LeRobot SmolVLA inference
"""
import sys
sys.path.append("../")

import torch
from lerobot.policies import smolvla
from lerobot.policies.utils import build_inference_frame


def test_smolvla():
    """
    SmolVLA test
    """
    ### parameters
    name = "lerobot/smolvla_base"
    task = "move clockwise"

    device = torch.device("cuda")
    policy = smolvla.SmolVLAPolicy.from_pretrained(name)

    #prepp, postp = smolvla.make_smolvla_pre_post_processors(policy.config)

    #obs = prepp("temp/camera_0.png")
    #print(obs)

    #obs_frame = build_inference_frame(
    #    observation=obs, task=task, device=device
    #)



if __name__ == "__main__":
    test_smolvla()