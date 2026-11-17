"""cta_learn: human video -> task spec -> simulated robot demos -> policy -> robot."""

from pathlib import Path

__version__ = "0.1.0"

#: Repository root (…/cta-learn), resolved from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
