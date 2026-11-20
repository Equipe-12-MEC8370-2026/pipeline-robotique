"""
Test LeRobot SmolVLA inference

https://deepwiki.com/huggingface/lerobot/5.2-observation-and-action-processors
"""
import sys
sys.path.append("../")

import numpy as np
import torch
from lerobot.policies import smolvla
from lerobot.policies.utils import build_inference_frame
from lerobot.lerobot_types import RobotObservation


def test_smolvla():
    """
    SmolVLA test
    """
    ### parameters
    name = "lerobot/smolvla_base"
    task = "move clockwise"

    device = torch.device("cuda")
    policy = smolvla.SmolVLAPolicy.from_pretrained(name)

    prepp, postp = smolvla.make_smolvla_pre_post_processors(policy.config)

    obs = RobotObservation()
    print(obs)


if __name__ == "__main__":
    test_smolvla()