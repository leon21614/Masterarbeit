"""
compare_rl.py

Dieses Skript dient zur Evaluierung und zum Vergleich von trainierten 
Reinforcement Learning-Modellen (gespeichert als .zip-Dateien von Stable Baselines3).

Es führt zwei Hauptaufgaben für jedes Modell in der 'MODELS'-Liste aus:

1.  **Einzel-Evaluierung & Plot:**
    Ein einzelner, fester Startpuls (definiert durch RUN_SEED) wird verwendet, 
    um die Leistung des Modells zu visualisieren. Es plottet die Intensität (Zeit)
    und die Phase (Frequenz) des initialen Pulses im Vergleich zum vom RL-Modell 
    optimierten Puls.

2.  **Batch-Evaluierung & Statistik:**
    Das Modell wird auf einer größeren Anzahl (N_PULSES) von unterschiedlichen,
    zufälligen Startpulsen getestet. Es berechnet und druckt statistische 
    Kennzahlen, wie die durchschnittliche RMS-Pulsbreitenreduktion.
"""

# ======= Import der notwendigen Bibliotheken =======
import os
import gc  # Garbage Collector zur Speicherfreigabe
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces

# ======= Import aus den Projekt-Dateien =======
# Import der Simulationsumgebung und Basiskonfiguration
from pulse_rl_env import PulseShaperEnv
from config import ENV_KW, MAX_EP_LEN, PULSE_SEED

# Import der Kernphysik-Funktionen (Puls-Erzeugung, Shaper, RMS-Berechnung)
from pulseOpt import ttc, femto, pulseShaper, calcRMSWidth, Ts, angFregTHz

# ======= Konfiguration der Evaluierung =======

# --- Modelle hier eintragen ---
# Trage hier die Pfade zu den Modellen ein, die du evaluieren möchtest.
MODELS = [
    "./Intensitaet_25_Stuetzstellen.zip",
    #"./Intensitaet_50_Stuetzstellen.zip",

]

# --- Evaluierungs-Parameter ---
# Seed für den EINEN Puls, der für die Plots verwendet wird.
# (Standardmäßig aus config.py importiert)
RUN_SEED = PULSE_SEED 

# Anzahl der verschiedenen Pulse für die statistische Auswertung (Batch-Evaluierung)
N_PULSES_STATS = 100

# Ob die Aktionen bei der Evaluierung deterministisch sein sollen (empfohlen)
DETERMINISTIC = True

# Unterdrückt die Warnung von SB3, dass der Replay Buffer nicht in den Speicher passt.
# Bei der Inferenz (Evaluierung) wird kein Replay Buffer benötigt, daher ist die Warnung irrelevant.
warnings.filterwarnings(
    "ignore",
    message="This system does not have apparently enough memory to store the complete replay buffer"
)

# ======= Gym-Wrapper für Kompatibilität =======

class PadOrCropObs(gym.ObservationWrapper):
    """
    Gymnasium-Wrapper, um die Observations-Dimension der Umgebung an die 
    vom geladenen Modell erwartete Dimension anzupassen.

    Dies ist nützlich, wenn du Modelle evaluierst, die mit unterschiedlichen
    Observations-Größen (z.B. durch include_spec_amp=True/False) trainiert wurden.

    - Wenn die Env-Observation kleiner ist als vom Modell erwartet -> Padding mit Nullen.
    - Wenn die Env-Observation größer ist als vom Modell erwartet -> Cropping (Abschneiden).
    """
    def __init__(self, env, target_dim: int):
        super().__init__(env)
        self.target_dim = int(target_dim)
        
        # Passe den Observation Space des Wrappers an die Zieldimension an
        # Die Wertebereiche (low/high) müssen ggf. angepasst werden, falls
        # die Observation nicht auf [-1, 10] normiert ist.
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(self.target_dim,), dtype=np.float32
        )

    def observation(self, obs):
        """Passt die Observation 'obs' an die 'target_dim' an."""
        obs = np.asarray(obs, dtype=np.float32)
        cur = obs.shape[0]
        
        if cur < self.target_dim:
            # Padding mit Nullen
            pad = np.zeros(self.target_dim - cur, dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=0)
        elif cur > self.target_dim:
            # Cropping (Abschneiden)
            obs = obs[:self.target_dim]
        return obs

# ======= Hilfsfunktionen =======

def time_fs():
    """Gibt die globale Zeitachse 'ttc' in Femtosekunden zurück."""
    return np.asarray(ttc) / femto  # s -> fs

def make_env_matching_model(model, base_env_kw):
    """
    Erstellt eine PulseShaperEnv-Instanz, die zu den Dimensionen des 
    geladenen Modells passt (Aktions- und Observationsraum).

    Args:
        model: Das geladene Stable Baselines3 Modell.
        base_env_kw: Die Basis-Konfiguration für die Umgebung (aus config.py).

    Returns:
        Eine (ggf. gewrappte) Gymnasium-Umgebung.
    """
    try:
        # Lese die Aktionsdimension (Anzahl Stützstellen S) aus dem Modell
        model_S = int(model.action_space.shape[0])
    except Exception:
        # Fallback, falls das Lesen fehlschlägt
        model_S = base_env_kw.get("S", 25)

    env_kw = dict(base_env_kw)
    # Setze die Anzahl der Stützstellen (Aktionsdim) in der Env auf die des Modells
    env_kw["S"] = model_S  
    env = PulseShaperEnv(**env_kw)

    # Lese die Observationsdimensionen
    env_dim = int(env.observation_space.shape[0])
    model_dim = int(model.observation_space.shape[0])

    # Wenn die Dimensionen nicht übereinstimmen, nutze den Wrapper
    if env_dim == model_dim:
        return env
    
    print(f"  [Info] Passe Obs-Dim an: Env={env_dim} -> Modell={model_dim}")
    return PadOrCropObs(env, target_dim=model_dim)

# ======= Kernfunktion: Einzel-Evaluierung (für Plot) =======

def run_model_once(model_path, env_kw, max_steps=100, deterministic=True, seed=0):
    """
    Lädt ein SAC-Modell, führt eine einzelne Episode auf der Umgebung 
    (mit spezifischem Seed) aus und gibt die finalen Ergebnisse zurück.

    Args:
        model_path (str): Pfad zur .zip-Datei des Modells.
        env_kw (dict): Konfiguration für die Umgebung.
        max_steps (int): Maximale Anzahl an Schritten in der Episode.
        deterministic (bool): Ob die Aktionen deterministisch gewählt werden sollen.
        seed (int): Der Seed für den Startpuls der Umgebung.

    Returns:
        dict: Ein Dictionary mit den finalen Puls-Eigenschaften (Intensität, Phase, etc.).
    """
    print(f"--- Lade Modell für Einzel-Plot: {model_path} ---")
    model = SAC.load(
        model_path,
        # buffer_size=1 ist ein Trick, um Speicher bei der Inferenz zu sparen
        custom_objects={"buffer_size": 1}, 
        device="cpu",
        print_system_info=False
    )
    
    # Erstelle eine Umgebung, die zum geladenen Modell passt
    env = make_env_matching_model(model, env_kw)
    obs, _info = env.reset(seed=seed)

    # Führe eine Episode durch
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Extrahiere die finalen Pulsdaten aus der Umgebung
    # (Unterscheidung, ob die Umgebung gewrapped ist oder nicht)
    if isinstance(env, PadOrCropObs):
        # Zugriff auf die "innere" Umgebung via env.env
        _, inten_final = env.env.get_current_intensity()
        w_st, phase_final = env.env.get_current_phase()
    else:
        _, inten_final = env.get_current_intensity()
        w_st, phase_final = env.get_current_phase()

    # Extrahiere einen Label-Namen aus dem Dateipfad
    label = os.path.splitext(os.path.basename(model_path))[0]

    # Ressourcen freigeben
    try:
        env.close()
    except Exception:
        pass
    del model
    gc.collect()

    return dict(
        inten=inten_final,
        phase=np.asarray(phase_final).copy(),
        w_st=np.asarray(w_st).copy(),
        tt=time_fs(),
        label=label
    )

def initial_from_env(env_kw, seed=0):
    """
    Erstellt den initialen Referenzpuls (Startpuls mit Null-Phase) 
    für einen gegebenen Seed.

    Args:
        env_kw (dict): Konfiguration für die Umgebung.
        seed (int): Der Seed für den Startpuls.

    Returns:
        dict: Ein Dictionary mit den initialen Puls-Eigenschaften.
    """
    # Nutzt die Basis-Env (S=25), da die Aktionsdim hier irrelevant ist
    env = PulseShaperEnv(**env_kw)
    env.reset(seed=seed)

    # Setze die Phase manuell auf Null
    # -> Der Output entspricht dem (ungeformten) Eingangspuls
    env._phase[:] = 0.0

    # Hole Intensität und Frequenz-Stützstellen
    _, inten0 = env.get_current_intensity()
    w_st, _ = env.get_current_phase() # Phase ist hier eh Null

    try:
        env.close()
    except Exception:
        pass

    return dict(
        inten=inten0,
        phase=np.zeros_like(w_st),
        w_st=np.asarray(w_st).copy(),
        tt=time_fs()
    )

# ======= Kernfunktion: Plotting =======

def plot_single_with_phase(initial, rl, out_png):
    """
    Plottet einen 2-Panel-Vergleich (Intensität + Phase) zwischen dem 
    initialen Puls und dem vom RL-Modell optimierten Puls.

    Args:
        initial (dict): Rückgabe von initial_from_env().
        rl (dict): Rückgabe von run_model_once().
        out_png (str): Dateiname für den Speicherort des Plots.
    """
    tt = initial["tt"]

    # Normiere Intensitäten (jede Kurve separat auf max=1)
    I_init = initial["inten"] / (np.max(initial["inten"]) + 1e-20)
    I_rl   = rl["inten"]      / (np.max(rl["inten"])      + 1e-20)

    # Phasen & Frequenzachse (umgerechnet in THz für den Plot)
    w_init_thz = initial["w_st"] / angFregTHz
    w_rl_thz   = rl["w_st"]      / angFregTHz

    # Phasen auf [-pi, pi) wrappen für eine saubere Darstellung
    wrap = lambda p: (np.asarray(p) + np.pi) % (2*np.pi) - np.pi
    phi_init = wrap(initial["phase"])
    phi_rl   = wrap(rl["phase"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # --- Oben: Intensität (Zeitbereich) ---
    ax = axes[0]
    ax.plot(tt, I_init, lw=2, label="Initial")
    ax.plot(tt, I_rl,   lw=2, label=f"RL: {rl['label']}")
    ax.set_xlabel("Zeit (fs)")
    ax.set_ylabel("Intensität (norm.)")
    ax.set_title("Intensität im Zeitbereich")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(-2500, 2500) # Optional: Zoom auf den relevanten Zeitbereich

    # --- Unten: Shaper-Phase (Frequenzbereich) ---
    ax = axes[1]
    ax.plot(w_init_thz, phi_init, lw=2, label="Initial (Phase=0)")
    ax.plot(w_rl_thz,   phi_rl,   lw=2, label=f"RL: {rl['label']}")
    ax.set_xlabel("Frequenz (THz)")
    ax.set_ylabel("Phase φ(ω) [rad]")
    ax.set_title("Shaper-Phase (erlernt vom RL-Agent)")
    ax.set_ylim(-np.pi * 1.1, np.pi * 1.1) # Leicht größerer Bereich
    ax.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"  [OK] Plot gespeichert: {out_png}")

# ======= Kernfunktion: Batch-Evaluierung (für Statistik) =======

def evaluate_model_stats(model_path, env_kw, n_pulses=100, max_steps=100, deterministic=True, base_seed=0):
    """
    Evaluiert ein Modell über 'n_pulses' verschiedene Startpulse (Seeds).

    Misst für jeden Seed:
    1. Die RMS-Breite des Eingangspulses (mit Null-Phase).
    2. Die RMS-Breite des Ausgangspulses (nach Anwendung des RL-Modells).

    Gibt ein Dict mit aggregierten Statistiken zurück.

    Args:
        model_path (str): Pfad zur .zip-Datei des Modells.
        env_kw (dict): Konfiguration für die Umgebung.
        n_pulses (int): Anzahl der zu testenden Zufallspulse.
        max_steps (int): Maximale Anzahl an Schritten pro Episode.
        deterministic (bool): Ob die Aktionen deterministisch gewählt werden sollen.
        base_seed (int): Ein Basis-Seed, um die 'n_pulses' Seeds reproduzierbar zu machen.

    Returns:
        dict: Ein Dictionary mit statistischen Kennzahlen (Listen und Mittelwerte).
    """
    print(f"--- Starte Batch-Evaluierung ({n_pulses} Pulse) für: {model_path} ---")
    
    # Modell einmalig laden
    model = SAC.load(
        model_path,
        custom_objects={"buffer_size": 1},
        device="cpu",
        print_system_info=False
    )
    
    # Umgebung für RL-Durchläufe (wird innen wiederverwendet)
    env_rl = make_env_matching_model(model, env_kw)
    # Umgebung für Initial-Messung (wird innen wiederverwendet)
    env_in = PulseShaperEnv(**env_kw)

    rms_in_list = []
    rms_out_list = []

    for i in range(n_pulses):
        # Für jeden Puls einen neuen, reproduzierbaren Seed verwenden
        seed_i = int(base_seed + i + 1) # +1, um den Plot-Seed nicht zu wiederholen

        # --- 1. Eingangs-RMS messen (Nullphase) ---
        env_in.reset(seed=seed_i)
        env_in._phase[:] = 0.0  # Nullphase = kein Shaping
        _, inten_in = env_in.get_current_intensity()
        rms_in = calcRMSWidth(inten_in, ttc) / femto  # in fs
        
        # --- 2. RL-Run für denselben Startpuls ---
        obs, _ = env_rl.reset(seed=seed_i)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env_rl.step(action)
            if terminated or truncated:
                break
        
        # Finale Intensität nach RL holen
        if isinstance(env_rl, PadOrCropObs):
            _, inten_out = env_rl.env.get_current_intensity()
        else:
            _, inten_out = env_rl.get_current_intensity()
        
        rms_out = calcRMSWidth(inten_out, ttc) / femto  # in fs

        rms_in_list.append(float(rms_in))
        rms_out_list.append(float(rms_out))

    # Umgebungen schließen
    try:
        env_in.close()
        env_rl.close()
    except Exception:
        pass
    
    # Modell freigeben
    del model
    gc.collect()

    # --- Statistiken berechnen ---
    rms_in_arr = np.asarray(rms_in_list, dtype=float)
    rms_out_arr = np.asarray(rms_out_list, dtype=float)

    # Mittelwerte
    mean_rms_in = float(np.mean(rms_in_arr))
    mean_rms_out = float(np.mean(rms_out_arr))

    # Verhältnis der Mittelwerte (wichtigste Metrik)
    ratio_mean = mean_rms_out / (mean_rms_in + 1e-20)
    
    # Absolute Reduktion
    mean_abs_reduction = float(np.mean(rms_in_arr - rms_out_arr))
    
    # Prozentuale Reduktion (basierend auf dem Verhältnis der Mittelwerte)
    percent_reduction = 100.0 * (1.0 - ratio_mean)

    return dict(
        rms_in_list=rms_in_list,
        rms_out_list=rms_out_list,
        mean_rms_in=mean_rms_in,
        mean_rms_out=mean_rms_out,
        ratio_mean_out_over_in=ratio_mean,
        mean_abs_reduction_fs=mean_abs_reduction,
        percent_reduction=percent_reduction,
    )

# ======= Main-Funktion =======

def main():
    """
    Haupt-Ausführungsfunktion.
    Iteriert durch alle Modelle in der 'MODELS'-Liste und führt 
    sowohl die Einzel-Plot-Evaluierung als auch die Batch-Statistik-Evaluierung durch.
    """
    print(f"[INFO] Evaluierung startet.")
    print(f"[INFO] Basis-Seed für Plot-Puls: {RUN_SEED}")
    print(f"[INFO] Basis-Umgebungs-KW (aus config.py): {ENV_KW}")
    print(f"[INFO] Maximale Episodenlänge: {MAX_EP_LEN}\n")

    # Erzeuge den initialen Referenzpuls (nur einmal für alle Plots)
    # WICHTIG: Nutzt die Basis-ENV_KW, nicht die S-Dimension eines Modells!
    initial = initial_from_env(ENV_KW, seed=RUN_SEED)

    # Iteriere über alle zu testenden Modelle
    for model_path in MODELS:
        if not os.path.isfile(model_path):
            print(f"[WARN] Modell nicht gefunden: {model_path} – überspringe.")
            continue

        print(f"\n{'='*60}")
        print(f"VERARBEITE MODELL: {os.path.basename(model_path)}")
        print(f"{'='*60}")

        # --- 1. Einzeldurchlauf für Plot ---
        rl_run = run_model_once(
            model_path, 
            ENV_KW, 
            max_steps=MAX_EP_LEN, 
            deterministic=DETERMINISTIC, 
            seed=RUN_SEED
        )

        # Plot erstellen und speichern
        out_png = f"compare_{rl_run['label']}_S{rl_run['w_st'].size}_plot.png"
        plot_single_with_phase(initial, rl_run, out_png)

        # RMS-Werte für diesen einen Plot-Durchlauf berechnen
        rms_i = calcRMSWidth(initial["inten"], ttc) / femto
        rms_r = calcRMSWidth(rl_run["inten"],   ttc) / femto
        print(f"  [Info] Plot-Run: RMS_init={rms_i:.2f} fs | RMS_RL={rms_r:.2f} fs")


        # --- 2. Batch-Evaluierung für Statistiken ---
        stats = evaluate_model_stats(
            model_path=model_path,
            env_kw=ENV_KW,
            n_pulses=N_PULSES_STATS,
            max_steps=MAX_EP_LEN,
            deterministic=DETERMINISTIC,
            base_seed=RUN_SEED  # Nutzt Seeds RUN_SEED+1 ... RUN_SEED+N_PULSES_STATS
        )

        # Statistiken ausgeben
        print(f"\n--- [Statistik über {N_PULSES_STATS} Pulse] {rl_run['label']} ---")
        print(f"  Ø RMS Eingang:       {stats['mean_rms_in']:.2f} fs")
        print(f"  Ø RMS RL (Ausgang):  {stats['mean_rms_out']:.2f} fs")
        print(f"  Ø Abs. Reduktion:  {stats['mean_abs_reduction_fs']:.2f} fs")
        print(f"  Verhältnis (Øout/Øin): {stats['ratio_mean_out_over_in']:.4f}")
        print(f"  -> Mittlere Reduktion: {stats['percent_reduction']:.2f} %")
        print(f"--------------------------------------------------\n")

# ======= Skript-Ausführung =======

if __name__ == "__main__":
    main()