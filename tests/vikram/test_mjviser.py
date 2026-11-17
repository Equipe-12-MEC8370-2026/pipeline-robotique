"""
Test Mujoco Camera with mjviser
"""
import mujoco
from mjviser import Viewer


def render_fn():
    pass


def main():
    # Load the model and create a data instance.
    model = mujoco.MjModel.from_xml_path("so101/scene.xml")
    data = mujoco.MjData(model)

    viewer = Viewer(
        model, data, render_fn=render_fn
    )

    Viewer(model, data).run()


if __name__ == "__main__":
    main()