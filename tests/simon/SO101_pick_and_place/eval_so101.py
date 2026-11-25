from stable_baselines3 import SAC
from so101_env import SO101PickPlaceEnv

env = SO101PickPlaceEnv(render_mode="human")

# Vous pouvez aussi charger un point de sauvegarde intermédiaire (ex: "./checkpoints/sac_so101_100000_steps")
model = SAC.load("./sac_so101_final", env=env)

obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()