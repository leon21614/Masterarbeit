# Importieren der notwendigen Bibliotheken
import numpy as np
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Importieren der benutzerdefinierten Umgebung und Callbacks
from pulse_rl_env import PulseShaperEnv
from callbacks_rl import RewardComponentsPerEpisodeCallback, EvalAndLogCallback

# ---- Laden der Konfiguration ----
# Hier werden alle wichtigen Parameter aus der config.py-Datei geladen.
from config import (
    ENV_KW, MAX_EP_LEN, TOTAL_STEPS,
    EVAL_EVERY, SAVE_EVERY, LOGDIR,
    PULSE_SEED, TRAIN_KW,
)

# ======= Hyperparameter (Standardwerte, können durch config.TRAIN_KW überschrieben werden) =======
# Die Hyperparameter für das Training werden hier gesetzt.
# Man kann sie direkt in der config.py anpassen, ohne den Code hier zu ändern.
HP = TRAIN_KW

# ======= Erstellen der Umgebungen (ohne Vektorisierung oder Normalisierung) =======
# Die Trainings- und Evaluierungsumgebungen werden erstellt.
# Monitor wickelt die Umgebung ein, um Statistiken wie Belohnung und Episodenlänge zu protokollieren.
# TimeLimit begrenzt die maximale Anzahl an Schritten pro Episode.
train_env = Monitor(TimeLimit(PulseShaperEnv(**ENV_KW), max_episode_steps=MAX_EP_LEN))
eval_env  = Monitor(TimeLimit(PulseShaperEnv(**ENV_KW), max_episode_steps=MAX_EP_LEN))

# Setzen des Seeds für die Zufallszahlengeneratoren, um reproduzierbare Ergebnisse zu erhalten.
train_env.reset(seed=PULSE_SEED)
eval_env.reset(seed=PULSE_SEED)

# ======= Callbacks für das Training =======
# Callbacks sind Funktionen, die an bestimmten Stellen während des Trainings aufgerufen werden.
# CheckpointCallback: Speichert das Modell in regelmäßigen Abständen.
checkpoint_cb = CheckpointCallback(save_freq=SAVE_EVERY, save_path="./checkpoints", name_prefix="sac_pulse")
# ProgressBarCallback: Zeigt eine Fortschrittsanzeige für das Training an.
progress_cb   = ProgressBarCallback()
# RewardComponentsPerEpisodeCallback: Ein benutzerdefinierter Callback, um die einzelnen Belohnungskomponenten zu loggen.
ep_metrics_cb = RewardComponentsPerEpisodeCallback(csv_dir="logs", verbose=1)
# EvalAndLogCallback: Ein weiterer benutzerdefinierter Callback, der das Modell regelmäßig evaluiert und die Ergebnisse loggt.
eval_log_cb   = EvalAndLogCallback(eval_env=eval_env, eval_freq=EVAL_EVERY, csv_dir="logs", n_episodes=5, verbose=1)

# ======= Initialisierung des SAC-Modells =======
# Hier wird das Soft Actor-Critic (SAC) Modell erstellt.
# "MlpPolicy" bedeutet, dass wir ein Multi-Layer Perceptron (also ein einfaches neuronales Netz) als Policy verwenden.
# Die ganzen Hyperparameter (HP) werden dem Modell übergeben.
model = SAC(
    "MlpPolicy",
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
    tensorboard_log=LOGDIR,
    verbose=HP["verbose"],
)

# Ausgabe von nützlichen Informationen vor dem Trainingsstart
print(">>> Policy (Actor):")
try:
    # Zeigt die Architektur des Actor-Netzwerks an.
    print(model.policy.actor)
except Exception:
    pass
print(">>> Policy device:", model.policy.device) # Auf welchem Gerät (CPU/GPU) das Training läuft.
print(">>> Hyperparams:")
for k, v in HP.items():
    # Listet alle verwendeten Hyperparameter auf.
    print(f"    {k}: {v}")

# Start des eigentlichen Trainingsprozesses
print(">>> Starte Training ...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    # Die Callbacks werden hier übergeben.
    callback=[checkpoint_cb, progress_cb, ep_metrics_cb, eval_log_cb]
)

# Speichern des finalen Modells nach Abschluss des Trainings
model.save("sac_pulse_shaper")
print(">>> Fertig. Modell gespeichert: sac_pulse_shaper.zip")