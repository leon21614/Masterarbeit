# Importieren der notwendigen Bibliotheken
import numpy as np
from torch import nn

# =========================
# Grundlegende Trainings-Settings
# =========================
MAX_EP_LEN   = 20          # Maximale Schritte pro Episode.
TOTAL_STEPS  = 200_000       # Gesamte Trainingsschritte über alle Episoden.
PULSE_SEED = 1337            # Seed für den Zufallsgenerator, um reproduzierbare Pulse zu erzeugen.

# Logging / Checkpoints
LOGDIR     = "./tb"          # Verzeichnis für TensorBoard-Logs.
SAVE_EVERY = 25_000          # Speicher-Intervall für Modell-Checkpoints.
EVAL_EVERY = 5_000          # Evaluations-Intervall des Modells.

# =========================
# Environment-Parameter (Defaults)
# =========================
# Standard-Parameter für die Simulationsumgebung (PulseShaperEnv).
ENV_KW = dict(
    # Beobachtungsraum
    include_spec_amp=True,   # Spektrale Amplitude als Teil der Beobachtung nutzen?
    spec_size=256,           # Größe des Spektralamplituden-Vektors.
    obs_size=256,            # Größe des Intensitäts-Vektors.
   
    # Shaper
    S=25,                    # Anzahl der Phasen-Stützstellen (Aktionsdimension).
    phi_max=np.pi,           # Maximaler Phasenwert (z.B. ±π).
 
    # Aktionen
    incremental_actions=True,      # True: Aktionen sind relative Änderungen. False: Aktionen sind absolute Werte.
    action_scale=(np.pi/MAX_EP_LEN), # Skalierungsfaktor für Aktionen (Schrittweite).

    # Randomisierung der Eingabe
    randomize_on_reset=True,   # Bei Reset einen neuen, zufälligen Eingangspuls erzeugen?
    hold_pulse_episodes= 1,    # Anzahl der Episoden, für die der gleiche Puls wiederverwendet wird.
)

# =========================
# Trainings-Hyperparameter für den SAC-Algorithmus
# =========================
TRAIN_KW = {
    "learning_rate": 3e-4,      # Lernrate des Optimierers.
    "gamma": 0.7,               # Discount-Faktor für zukünftige Belohnungen.
    "buffer_size": 50_000,      # Größe des Replay-Buffers.
    "batch_size": 128,          # Batch-Größe für das Training.
    "ent_coef": "auto",         # Entropie-Koeffizient für die Exploration.
    "train_freq": 1,            # Trainings-Frequenz (in Schritten).
    "gradient_steps": 1,        # Anzahl der Gradienten-Schritte pro Update.
    "learning_starts": 50_000,  # Anzahl der Schritte vor dem ersten Training.
    "tau": 0.005,               # Soft-Update-Faktor für Target-Netzwerke.
    "policy_kwargs": dict(      # Architektur der neuronalen Netze.
        net_arch=dict(pi=[512,1024,512], qf=[512,1024,512]), # Neuronen pro Schicht für Policy (pi) und Q-Funktion (qf).
        activation_fn= nn.ReLU, # Aktivierungsfunktion.
    ),
    "device": "cuda",           # Trainingsgerät ("cuda" oder "cpu").
    "verbose": 1,               # Log-Level (1 für Infos).
}

