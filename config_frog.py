import numpy as np
from torch import nn

# =========================
# Grundlegende Trainings-Settings (FROG-Setup)
# =========================
MAX_EP_LEN   = 20           # Maximale Schritte pro Episode.
TOTAL_STEPS  = 400_000      # Gesamte Trainingsschritte über alle Episoden.

# Logging / Checkpoints
LOGDIR_FROG = "./tb_frog"   # Verzeichnis für TensorBoard-Logs (FROG-Setup).
SAVE_EVERY  = 25_000        # Speicher-Intervall für Modell-Checkpoints.
EVAL_EVERY  = 5_000         # Evaluations-Intervall des Modells.

# =========================
# Environment-Parameter (FROG)
# =========================
ENV_FROG_KW = dict(
    
    # Shaper
    S=50,                   # Anzahl der Phasen-Stützstellen (Aktionsdimension).
    phi_max=np.pi,          # Maximaler Phasenwert (z.B. ±π).
    incremental_actions=True, # True: Aktionen sind relative Änderungen. False: Aktionen sind absolute Werte.
    action_scale=(np.pi/MAX_EP_LEN), # Skalierungsfaktor für Aktionen (Schrittweite).

    # Eingabe als 256x256 FROG-Bild + Vektor-Features
    frog_bins=256,          # Anzahl der Bins für die FROG-Spur (ergibt 256x256 Bild).
    randomize_on_reset=True, # Bei Reset einen neuen, zufälligen Eingangspuls erzeugen?
    hold_pulse_episodes=1,   # Anzahl der Episoden, für die der gleiche Puls wiederverwendet wird.
)

# =========================
# Trainings-Hyperparameter (empfohlene Defaults, anpassbar)
# =========================
TRAIN_FROG_KW = {
    "learning_rate": 5e-5,    # Lernrate des Optimierers.
    "gamma": 0.7,             # Discount-Faktor für zukünftige Belohnungen.
    "buffer_size": 50_000,    # Größe des Replay-Buffers (Achtung: Bilder sind speicherintensiv).
    "batch_size": 128,        # Batch-Größe für das Training.
    "ent_coef": "auto",       # Entropie-Koeffizient für die Exploration.
    "train_freq": 1,          # Trainings-Frequenz (in Schritten).
    "gradient_steps": 1,      # Anzahl der Gradienten-Schritte pro Update.
    "learning_starts": 50_000,  # Anzahl der Schritte vor dem ersten Training.
    "tau": 0.005,             # Soft-Update-Faktor für Target-Netzwerke.
    "policy_kwargs": dict(
        # Nach dem CNN-Feature-Extractor folgt dieses MLP für Actor/Critic
        net_arch=dict(pi=[512,1024,512], qf=[512,1024,512]), # Neuronen pro Schicht für Policy (pi) und Q-Funktion (qf).
        # Optional: CNN-Ausgangsdimension (Default 512). Bei Bedarf freischalten:
        features_extractor_kwargs=dict(features_dim=256), # Dimension der Features nach dem CNN-Extraktor.
        activation_fn= nn.ReLU, # Aktivierungsfunktion.
    ),
    "device": "cuda",         # Trainingsgerät ("cuda" oder "cpu").
    "verbose": 1,             # Log-Level (1 für Infos).
}