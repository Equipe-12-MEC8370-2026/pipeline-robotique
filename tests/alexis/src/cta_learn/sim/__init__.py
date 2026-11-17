"""Isaac Lab simulation side of cta_learn (modules 5 and 7).

Everything under ``cta_learn.sim`` imports Isaac Lab, so it must only be imported
AFTER ``isaaclab.app.AppLauncher`` has started the simulator, and only from the
``isaacsim-env`` runtime (with ``PYTHONPATH=<repo>/src``). ``cta_learn.core`` never
imports this package.

Importing ``cta_learn.sim.tasks`` registers the gym environments.
"""
