# Importieren der notwendigen Bibliotheken
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Importieren der benutzerdefinierten Umgebung für FROG
from pulse_rl_env_frog import PulseShaperFROGEnv
from callbacks_rl import RewardComponentsPerEpisodeCallback, EvalAndLogCallback

# ---- FROG-Config laden ----
# Hier werden alle wichtigen Parameter aus der config_frog.py-Datei geladen.
from config_frog import (
    ENV_FROG_KW, MAX_EP_LEN, TOTAL_STEPS,
    EVAL_EVERY, SAVE_EVERY, LOGDIR_FROG, TRAIN_FROG_KW,
)
    
# Die Hyperparameter für das Training werden hier gesetzt.
HP = TRAIN_FROG_KW

# ======= Umgebungen =======
# Erstellen der Trainings- und Evaluierungsumgebungen für das FROG-Setup.
# Monitor wickelt die Umgebung ein, um Statistiken wie Belohnung und Episodenlänge zu protokollieren.
# TimeLimit begrenzt die maximale Anzahl an Schritten pro Episode.
train_env = Monitor(TimeLimit(PulseShaperFROGEnv(**ENV_FROG_KW), max_episode_steps=MAX_EP_LEN))
eval_env  = Monitor(TimeLimit(PulseShaperFROGEnv(**ENV_FROG_KW), max_episode_steps=MAX_EP_LEN))

# ======= Callbacks =======
# Callbacks sind Funktionen, die während des Trainings aufgerufen werden.
# CheckpointCallback: Speichert das Modell in regelmäßigen Abständen.
checkpoint_cb = CheckpointCallback(save_freq=SAVE_EVERY, save_path="./checkpoints_frog", name_prefix="sac_pulse_frog")
# ProgressBarCallback: Zeigt eine Fortschrittsanzeige für das Training an.
progress_cb   = ProgressBarCallback()
# RewardComponentsPerEpisodeCallback: Loggt einzelne Belohnungskomponenten.
ep_metrics_cb = RewardComponentsPerEpisodeCallback(csv_dir="logs_frog", verbose=1)
# EvalAndLogCallback: Evaluiert das Modell regelmäßig auf der eval_env.
eval_log_cb   = EvalAndLogCallback(eval_env=eval_env, eval_freq=EVAL_EVERY, csv_dir="logs_frog", n_episodes=5, verbose=1)

# ======= Modell =======
# Hier wird das Soft Actor-Critic (SAC) Modell erstellt.
# "CnnPolicy" wird verwendet, da die Beobachtung ein Bild (die FROG-Spur) enthält.
# Die ganzen Hyperparameter (HP) werden dem Modell übergeben.
model = SAC(
    "CnnPolicy",
    train_env,
    learning_rate=HP["learning_rate"],
    gamma=HP["gamma"],
    buffer_size=HP["buffer_size"],
    batch_size=HP["batch_size"],
    ent_coef=HP["ent_coef"],
    train_freq=HP["train_freq"],
    gradient_steps=HP["gradient_steps"],
    learning_starts=HP.get("learning_starts", 0),
    tau=HP["tau"],
    policy_kwargs=HP["policy_kwargs"],
    device=HP["device"],
    tensorboard_log=LOGDIR_FROG,
    verbose=HP["verbose"],
    
)

# Ausgabe von nützlichen Informationen vor dem Trainingsstart
print(">>> Policy (Actor):")
try:
    # Zeigt die Architektur des Actor-Netzwerks an (inkl. CNN-Extraktor).
    print(model.policy.actor)
except Exception:
    pass
print(">>> Policy device:", model.policy.device) # Auf welchem Gerät (CPU/GPU) das Training läuft.
print(">>> Hyperparams (FROG):")
for k, v in HP.items():
    # Listet alle verwendeten Hyperparameter auf.
    print(f"      {k}: {v}")

# Start des eigentlichen Trainingsprozesses
print(">>> Starte FROG-Training ...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    # Die Callbacks werden hier übergeben.
    callback=[checkpoint_cb, progress_cb, ep_metrics_cb, eval_log_cb]
)

# Speichern des finalen Modells nach Abschluss des Trainings
model.save("sac_pulse_shaper_frog")
print(">>> Fertig. Modell gespeichert: sac_pulse_shaper_frog.zip")