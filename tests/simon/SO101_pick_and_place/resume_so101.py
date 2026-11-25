import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from so101_env import SO101PickPlaceEnv

torch.set_num_threads(2)

env = SO101PickPlaceEnv(render_mode=None)

# Indiquez ici le chemin vers votre dernier fichier .zip (sans l'extension .zip)
checkpoint_path = "checkpoints/sac_so101_500000_steps"

# Chargement du cerveau et rattachement à l'environnement
model = SAC.load(checkpoint_path, env=env)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="sac_so101",
)

print(f"Reprise de l'entraînement depuis {checkpoint_path}...")

try:
    # reset_num_timesteps=False permet de continuer le compteur d'étapes au lieu de repartir à 0
    model.learn(
        total_timesteps=200000,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
    )
    model.save("sac_so101_final")
except KeyboardInterrupt:
    print("\nSauvegarde avant arrêt...")
    model.save("checkpoints/sac_so101_interrupted")

env.close()