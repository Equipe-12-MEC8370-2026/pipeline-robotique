"""Simplified CAD of the two aluminium T-slot profiles of the CTA demo 113656 (test parts for the interface).

    .venv-app\\Scripts\\python.exe app\\exemples\\make_profiles.py
-> app/exemples/profile_court_40x40x80.stl, profile_long_40x40x147.stl (millimetres)
Not the CTA's CAD (we do not have it): a 40 x 40 mm section with one T-slot per side (8 mm opening, wider chamber)
and a central hole, extruded to the lengths measured on the video (80 mm, 147 mm; docs/JOURNAL.md, P2.3).
The volume, hence the mass computed by the interface, is of the right order for such profiles.
"""
from pathlib import Path

import trimesh
from shapely.affinity import rotate
from shapely.geometry import Point, box
from shapely.ops import unary_union

S = 40.0          # section, mm
OPEN = 8.0        # slot opening
NECK = 4.0        # depth of the opening (the slot's lips)
CHAMBER_W, CHAMBER_D = 16.0, 7.0   # chamber behind the lips
HOLE = 8.4        # central hole diameter


def section():
    outer = box(-S / 2, -S / 2, S / 2, S / 2)
    # one slot on the +y side, then rotated to the 4 sides
    neck = box(-OPEN / 2, S / 2 - NECK, OPEN / 2, S / 2 + 1)
    chamber = box(-CHAMBER_W / 2, S / 2 - NECK - CHAMBER_D, CHAMBER_W / 2, S / 2 - NECK)
    slot = unary_union([neck, chamber])
    cuts = unary_union([rotate(slot, a, origin=(0, 0)) for a in (0, 90, 180, 270)] + [Point(0, 0).buffer(HOLE / 2, 32)])
    return outer.difference(cuts)


def main() -> None:
    out = Path(__file__).parent
    sec = section()
    for name, length in (("profile_court_40x40x80", 80.0), ("profile_long_40x40x147", 147.0)):
        m = trimesh.creation.extrude_polygon(sec, length)
        assert m.is_watertight
        m.export(out / f"{name}.stl")
        print(f"{name}.stl: {m.extents.round(1).tolist()} mm, volume {m.volume / 1000:.1f} cm3, "
              f"aluminium {m.volume / 1e9 * 2700 * 1000:.0f} g")


if __name__ == "__main__":
    main()
