"""
Testing Mujoco inbuilt viewer
"""

import sys
sys.path.append("../")
import time
import numpy as np

import mujoco
import mujoco.viewer
from PIL import Image


class SimState:
    """Small container to hold mutable state shared with the key callback."""
    def __init__(self):
        self.paused = False
        self.should_exit = False


def make_key_callback(state: SimState):
    def key_callback(keycode):
        # keycode is delivered as a GLFW key code (int).
        # Space = 32, Escape = 256
        if keycode == 32:  # SPACE
            state.paused = not state.paused
            print("Paused" if state.paused else "Resumed")
        elif keycode == 256:  # ESC
            print("Escape pressed, exiting...")
            state.should_exit = True
    return key_callback


def test_viewer():
    # Load the model and create a data instance.
    model = mujoco.MjModel.from_xml_path("../so101/scene.xml")
    data = mujoco.MjData(model)

    state = SimState()
    key_callback = make_key_callback(state)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not state.should_exit:
            step_start = time.time()

            if not state.paused:
                mujoco.mj_step(model, data)

            # Sync the viewer with the latest physics/visualization state.
            viewer.sync()

            # Try to roughly match the model's timestep for real-time playback.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    sys.exit(0)


def test_cam_output():
    """
    export camera output
    """
    model = mujoco.MjModel.from_xml_path("../so101/scene.xml")
    data = mujoco.MjData(model)

    state = SimState()
    key_callback = make_key_callback(state)

    ii = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

        renderer = mujoco.Renderer(model, height=224, width=224)
        data.ctrl[0] = np.pi

        while viewer.is_running() and not state.should_exit and ii < 10:
            step_start = time.time()

            if not state.paused:
                mujoco.mj_step(model, data)

            # Sync the viewer with the latest physics/visualization state.
            viewer.sync()

            if ii % 10 == 0:
                renderer.update_scene(data, camera="wrist_cam")
                Image.fromarray(renderer.render()).save(f"temp/camera_{ii//10}.png")
            ii += 1

            # Try to roughly match the model's timestep for real-time playback.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    sys.exit(0)


if __name__ == "__main__":
    test_viewer()
    #test_cam_output()