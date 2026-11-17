"""Shared types and the single unit-conversion layer."""

from cta_learn.core.embodiment import (
    ArmSpec,
    CameraSpec,
    Embodiment,
    JointSpec,
    embodiment_from_dict,
    load_embodiment,
)
from cta_learn.core.units import (
    FORMATS,
    UnitRangeError,
    check_dataset_stats,
    convert,
    format_bounds,
    from_rad,
    to_rad,
    validate,
)

__all__ = [
    "ArmSpec",
    "CameraSpec",
    "Embodiment",
    "JointSpec",
    "embodiment_from_dict",
    "load_embodiment",
    "FORMATS",
    "UnitRangeError",
    "check_dataset_stats",
    "convert",
    "format_bounds",
    "from_rad",
    "to_rad",
    "validate",
]
