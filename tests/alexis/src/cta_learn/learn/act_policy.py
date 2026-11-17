"""Kept for compatibility: the ACT wrapper became the generic cta_learn.learn.lerobot_policy.LeRobotPolicy (P1.3)."""

from cta_learn.learn.lerobot_policy import LeRobotPolicy as ActPolicy

__all__ = ["ActPolicy"]
