from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from so101_env import SO101PickPlaceEnv

# 1. Création de l'environnement sans affichage (vitesse maximale sur CPU)
env = SO101PickPlaceEnv(render_mode=None)

# Sauvegarde automatique toutes les 50 000 étapes dans le dossier ./checkpoints/
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="sac_so101",
)

# 2. Initialisation de l'algorithme SAC
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=200000,
    batch_size=256,
    verbose=1,
)

print("Début de l'entraînement du bras SO-101...")

# 3. Apprentissage (300 000 étapes pour commencer à voir la saisie et le déplacement)
model.learn(total_timesteps=300000, callback=checkpoint_callback)

# 4. Sauvegarde finale
model.save("sac_so101_final")
print("Entraînement terminé ! Modèle sauvegardé sous 'sac_so101_final.zip'.")

env.close()