"""
compare_rl_frog.py

Dieses Skript dient zur Evaluierung und zum Vergleich von trainierten 
RL-Modellen (gespeichert als .zip), die auf FROG-Bildern (Image-Observations)
trainiert wurden.

Es führt zwei Hauptaufgaben für jedes Modell in der 'MODELS'-Liste aus:

1.  **Einzel-Evaluierung & Plots (3 Plots):**
    Ein einzelner, fester Startpuls (definiert durch RUN_SEED) wird verwendet, 
    um die Leistung des Modells zu visualisieren. Es plottet:
    a) Intensität (Zeit) vs. Phase (Frequenz) – Initial vs. RL.
    b) FROG-Bild (Initial) vs. FROG-Bild (RL).

2.  **Batch-Evaluierung & Statistik:**
    Das Modell wird auf einer größeren Anzahl (N_PULSES_STATS) von unterschiedlichen,
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
from scipy.interpolate import CubicSpline
from stable_baselines3 import SAC
import gymnasium as gym
from gymnasium import spaces

# ======= Import aus den Projekt-Dateien =======
# Import der Umgebungen (Basis-Env für Referenz, FROG-Env für Inferenz)
from pulse_rl_env import PulseShaperEnv
from pulse_rl_env_frog import PulseShaperFROGEnv
# Import der Konfigurationen
from config import ENV_KW, MAX_EP_LEN, PULSE_SEED
from config_frog import ENV_FROG_KW

# Import der Kernphysik-Funktionen
from pulseOpt import N, ttc, femto, pulseShaper, calcRMSWidth, Ts, angFregTHz
# Import der FROG-spezifischen Funktionen
from frogOpt import frog_from_pulse

# ======= Konfiguration der Evaluierung =======

# --- Modelle hier eintragen ---
MODELS = [
    "./FROG_25_Stuetzstellen.zip",
    #"./FROG_50_Stuetzstellen.zip", 
]

# --- Evaluierungs-Parameter ---
# Seed für den EINEN Puls, der für die Plots verwendet wird.
RUN_SEED = PULSE_SEED 

# Anzahl der verschiedenen Pulse für die statistische Auswertung (Batch-Evaluierung)
N_PULSES_STATS = 100

# Ob die Aktionen bei der Evaluierung deterministisch sein sollen (empfohlen)
DETERMINISTIC = True

# Unterdrückt die Warnung von SB3, dass der Replay Buffer nicht in den Speicher passt.
warnings.filterwarnings(
    "ignore",
    message="This system does not have apparently enough memory to store the complete replay buffer"
)


# ======= Gym-Wrapper & Helfer für Bild-Observaions =======

class EnsureImageLayout(gym.ObservationWrapper):
    """
    Gymnasium-Wrapper: Erzwingt ein bestimmtes Kanal-Layout für Bild-Beobachtungen.
    Stable Baselines3 erwartet standardmäßig "CHW" (Channels, Height, Width), 
    während die Umgebung vielleicht "HWC" liefert (oder umgekehrt).
    
    Dieser Wrapper transponiert die Bild-Arrays bei Bedarf.

    - target='chw': Erzwingt (C, H, W)
    - target='hwc': Erzwingt (H, W, C)
    """
    def __init__(self, env, target: str):
        assert target in ("chw", "hwc"), "Ziel-Layout muss 'chw' oder 'hwc' sein."
        super().__init__(env)
        self.target = target
        shp = env.observation_space.shape
        assert len(shp) == 3, "EnsureImageLayout ist nur für 3D-Bild-Beobachtungen."

        # Heuristik: Prüfen, ob eine Transponierung nötig ist
        HWC_like = (shp[-1] in (1, 3)) # Kanal-Dimension ist HINTEN
        CHW_like = (shp[0]  in (1, 3)) # Kanal-Dimension ist VORNE

        if target == "chw" and HWC_like:
            # Env liefert HWC, Modell will CHW
            C, H, W = shp[-1], shp[0], shp[1]
            self.observation_space = spaces.Box(low=0, high=255, shape=(C, H, W), dtype=env.observation_space.dtype)
        elif target == "hwc" and CHW_like:
            # Env liefert CHW, Modell will HWC
            C, H, W = shp[0], shp[1], shp[2]
            self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, C), dtype=env.observation_space.dtype)
        else:
            # Layout passt bereits
            self.observation_space = env.observation_space

    def observation(self, obs):
        """Führt die Transponierung der Beobachtung durch."""
        arr = np.asarray(obs)
        if arr.ndim != 3:
            return arr # Nicht-Bild-Daten (sollte hier nicht vorkommen)
        
        # HWC -> CHW
        if self.target == "chw" and arr.shape[-1] in (1, 3):
            return np.transpose(arr, (2, 0, 1))
        # CHW -> HWC
        if self.target == "hwc" and arr.shape[0] in (1, 3):
            return np.transpose(arr, (1, 2, 0))
        # Layout passt
        return arr

def model_expects_image(model) -> bool:
    """Prüft, ob der observation_space des Modells 3D (ein Bild) ist."""
    shp = getattr(model.observation_space, "shape", None)
    return isinstance(shp, tuple) and len(shp) == 3

def image_layout(space: spaces.Box) -> str:
    """Heuristik, um das Bild-Layout (chw/hwc) eines Spaces zu erraten."""
    shp = space.shape
    # Annahme: Kanal-Dimension ist 1 (Graustufen) oder 3 (RGB)
    if shp[-1] in (1, 3):
        return "hwc"
    if shp[0] in (1, 3):
        return "chw"
    # Fallback (Standard bei SB3-CNNs)
    return "chw"

def make_env_matching_model(model, base_env_kw_frog, base_env_kw_vec):
    """
    Erstellt die passende Umgebung (FROG oder Vektor) für das geladene Modell.
    Passt Aktionsdimension (S) und Observations-Layout (CHW/HWC) an.
    """
    try:
        # Lese die Aktionsdimension (Anzahl Stützstellen S) aus dem Modell
        model_S = int(model.action_space.shape[0])
    except Exception:
        model_S = base_env_kw_frog.get("S", 50) # Fallback auf FROG-Config

    if model_expects_image(model):
        # --- FROG-Modell (Bild-Input) ---
        env_kw = dict(base_env_kw_frog)
        env_kw["S"] = model_S # Aktionsdim anpassen
        env = PulseShaperFROGEnv(**env_kw)

        # Prüfe, ob das Bild-Layout (CHW/HWC) von Env und Modell passt
        model_layout = image_layout(model.observation_space)
        env_layout   = image_layout(env.observation_space)
        
        if model_layout != env_layout:
            print(f"  [Info] Passe Bild-Layout an: Env={env_layout} -> Modell={model_layout}")
            env = EnsureImageLayout(env, target=model_layout)
        
        return env

    # --- Vektor-Modell (z.B. klassische Intensitäts-Env) ---
    # (Dieser Teil ist für Kompatibilität, falls man Vektor-Modelle testet)
    env_kw = dict(base_env_kw_vec)
    env_kw["S"] = model_S
    env = PulseShaperEnv(**env_kw)
    # Hier könnte man analog den PadOrCropObs-Wrapper nutzen, falls nötig
    return env


# ======= Hilfsfunktionen (Physik & Plotting) =======

def time_fs():
    """Gibt die globale Zeitachse 'ttc' in Femtosekunden zurück."""
    return np.asarray(ttc) / femto  # s -> fs

def pulseShaper_phys_local(ytin, Ts, phStuetz, wStuetz):
    """
    Lokale Kopie der physikalischen Shaper-Funktion (ohne Normalisierung).
    Wird benötigt, um die FROG-Spuren des initialen und finalen Pulses 
    physikalisch korrekt (d.h. mit Energieerhaltung) zu berechnen.
    """
    Nloc = len(ytin)
    ws_loc = 2 * np.pi / Ts
    w0_loc = ws_loc / Nloc
    ww_loc = np.arange(0, Nloc) * w0_loc
    
    Yw = np.fft.fft(np.fft.fftshift(ytin))
    
    phIntp_loc = CubicSpline(wStuetz, phStuetz, extrapolate=False)
    ph = np.zeros(Nloc)
    mask = (ww_loc >= wStuetz[0]) & (ww_loc <= wStuetz[-1])
    ph[mask] = phIntp_loc(ww_loc[mask])
    
    Yw *= np.exp(1j * ph)
    ytout_raw = np.fft.fftshift(np.fft.ifft(Yw))
    return ytout_raw

# ======= Kernfunktion: Einzel-Evaluierung (für Plot) =======

def initial_from_env(env_kw_vec, S_ref, seed=0):
    """
    Erstellt den initialen Referenzpuls (Startpuls mit Null-Phase) 
    für einen gegebenen Seed.
    
    Nutzt die Vektor-Umgebung, da dies einfacher ist als die FROG-Env.
    
    Args:
        env_kw_vec (dict): Konfiguration für die (Vektor-)Umgebung.
        S_ref (int): Anzahl der Stützstellen (S) für die Referenzphase.
        seed (int): Der Seed für den Startpuls.

    Returns:
        dict: Ein Dictionary mit den initialen Puls-Eigenschaften.
    """
    env_kw = dict(env_kw_vec)
    env_kw['S'] = S_ref # Setze S auf die gewünschte Referenz-Dimension
    env = PulseShaperEnv(**env_kw)
    env.reset(seed=seed)

    # Hole den initialen (komplexen) Eingangspuls
    y_in = env.y_in.copy()
    
    # Setze die Phase manuell auf Null
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
        y_in=np.asarray(y_in).copy(), # WICHTIG für FROG-Plot
        tt=time_fs()
    )


def run_model_once(model_path, max_steps=100, deterministic=True, seed=0):
    """
    Lädt ein (FROG)-Modell, führt eine einzelne Episode auf der Umgebung 
    (mit spezifischem Seed) aus und gibt die finalen Ergebnisse zurück.
    """
    print(f"--- Lade Modell für Einzel-Plot: {model_path} ---")
    model = SAC.load(
        model_path,
        custom_objects={"buffer_size": 1}, 
        device="cpu",
        print_system_info=False
    )
    
    # Erstelle eine Umgebung, die zum geladenen Modell passt
    # (Übergibt beide Configs, die Funktion wählt die richtige aus)
    env = make_env_matching_model(model, ENV_FROG_KW, ENV_KW)
    obs, _info = env.reset(seed=seed)

    # Führe eine Episode durch
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Extrahiere die finalen Pulsdaten aus der Umgebung
    # (Unterscheidung, ob die Umgebung gewrapped ist oder nicht)
    if isinstance(env, (EnsureImageLayout, gym.Wrapper)):
        # Zugriff auf die "innere" Umgebung via env.env (oder env.unwrapped)
        base_env = env.unwrapped
    else:
        base_env = env
        
    _, inten_final = base_env.get_current_intensity()
    w_st, phase_final = base_env.get_current_phase()
    # Hole den Eingangspuls für den FROG-Plot
    y_in = base_env.y_in.copy() 

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
        y_in=np.asarray(y_in).copy(), # WICHTIG
        tt=time_fs(),
        label=label
    )

# ======= Kernfunktion: Plotting (Intensität/Phase) =======

def plot_single_with_phase(initial, rl, out_png):
    """
    Plottet einen 2-Panel-Vergleich (Intensität + Phase) zwischen dem 
    initialen Puls und dem vom RL-Modell optimierten Puls.
    
    (Diese Funktion ist identisch zu der in compare_rl.py,
     nur ohne die Nelder-Mead-Optionen)
    """
    tt_fs = initial["tt"]
    
    # Intensitäten für den Plot normieren
    I_init = initial["inten"] / (np.max(initial["inten"]) + 1e-20)
    I_rl   = rl["inten"]      / (np.max(rl["inten"])      + 1e-20)

    # Frequenzachsen (in THz) und Phasen (gewrapped)
    w_init_thz = initial["w_st"] / angFregTHz
    w_rl_thz   = rl["w_st"]      / angFregTHz
    
    wrap = lambda p: (np.asarray(p) + np.pi) % (2*np.pi) - np.pi
    phi_init = wrap(initial["phase"])
    phi_rl   = wrap(rl["phase"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # --- Oben: Intensität (Zeitbereich) ---
    ax = axes[0]
    ax.plot(tt_fs, I_init, lw=2, label="Initial")
    ax.plot(tt_fs, I_rl,   lw=2, label=f"RL: {rl['label']}")
    ax.set_xlabel("Zeit (fs)")
    ax.set_ylabel("Intensität (norm.)")
    ax.set_title("Intensität im Zeitbereich")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(-2500, 2500) # Optional: Zoom

    # --- Unten: Shaper-Phase (Frequenzbereich) ---
    ax = axes[1]
    ax.plot(w_init_thz, phi_init, lw=2, label="Initial")
    ax.plot(w_rl_thz,   phi_rl,   lw=2, label=f"RL: {rl['label']}")
    ax.set_xlabel("Frequenz (THz)")
    ax.set_ylabel("Phase φ(ω) [rad]")
    ax.set_title("Shaper-Phase")
    ax.set_ylim(-np.pi * 1.1, np.pi * 1.1)
    ax.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"  [OK] Plot (Intensität + Phase) gespeichert: {out_png}")


# ======= Kernfunktion: Plotting (FROG-Spuren) =======

def frog_trace_from_field(y_complex):
    """Wrapper: Berechnet die normalisierte FROG-Spur (Bild) aus einem Zeitfeld."""
    Ishg, tau_axis = frog_from_pulse(y_complex, Ts, normalize=True)
    return Ishg, tau_axis

def plot_frog_comparison(y_in_phys, y_out_phys, out_png_frog, title_left="Input FROG", title_right="Output FROG"):
    """
    Plottet einen 2-Panel-Vergleich der FROG-Bilder (Initial vs. RL).
    
    Args:
        y_in_phys: Komplexes Zeitfeld des Startpulses (physikalisch, unnormiert).
        y_out_phys: Komplexes Zeitfeld des RL-Pulses (physikalisch, unnormiert).
        out_png_frog: Dateipfad für den Plot.
    """
    # Berechne beide FROG-Spuren
    Ishg_in,  tau_axis = frog_trace_from_field(y_in_phys)
    Ishg_out, _        = frog_trace_from_field(y_out_phys)

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.4))
    
    # Linker Plot: Input FROG
    im0 = axs[0].imshow(Ishg_in, aspect='auto', origin='lower', extent=[tau_axis[0]/femto, tau_axis[-1]/femto, 0, N-1])
    axs[0].set_title(title_left)
    axs[0].set_xlabel("Delay τ (fs)")
    axs[0].set_ylabel("Frequenz-Bin (Index)")
    cbar0 = plt.colorbar(im0, ax=axs[0], pad=0.02)
    cbar0.set_label("Norm. Intensität")

    # Rechter Plot: Output FROG
    im1 = axs[1].imshow(Ishg_out, aspect='auto', origin='lower', extent=[tau_axis[0]/femto, tau_axis[-1]/femto, 0, N-1])
    axs[1].set_title(title_right)
    axs[1].set_xlabel("Delay τ (fs)")
    axs[1].set_ylabel("Frequenz-Bin (Index)")
    cbar1 = plt.colorbar(im1, ax=axs[1], pad=0.02)
    cbar1.set_label("Norm. Intensität")

    plt.tight_layout()
    plt.savefig(out_png_frog, dpi=160)
    plt.close()
    print(f"  [OK] Plot (FROG-Vergleich) gespeichert: {out_png_frog}")


# ======= Kernfunktion: Batch-Evaluierung (für Statistik) =======

def evaluate_model_stats(model_path, n_pulses=100, max_steps=100, deterministic=True, base_seed=0):
    """
    Evaluiert ein Modell über 'n_pulses' verschiedene Startpulse (Seeds).
    Misst für jeden Seed die RMS-Breite (Initial vs. RL-Output).
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
    env_rl = make_env_matching_model(model, ENV_FROG_KW, ENV_KW)
    
    # Umgebung für Initial-Messung (Vektor-Env ist hierfür ausreichend)
    env_in = PulseShaperEnv(**ENV_KW) 

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
        
        # Finale Intensität nach RL holen (Wrapper-sicher)
        base_env = env_rl.unwrapped
        _, inten_out = base_env.get_current_intensity()
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

    # --- Statistiken berechnen (analog zu compare_rl.py) ---
    rms_in_arr = np.asarray(rms_in_list, dtype=float)
    rms_out_arr = np.asarray(rms_out_list, dtype=float)

    mean_rms_in = float(np.mean(rms_in_arr))
    mean_rms_out = float(np.mean(rms_out_arr))
    ratio_mean = mean_rms_out / (mean_rms_in + 1e-20)
    mean_abs_reduction = float(np.mean(rms_in_arr - rms_out_arr))
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
    die Einzel-Evaluierung (Plots) und die Batch-Statistik durch.
    """
    print(f"[INFO] Evaluierung (FROG) startet.")
    print(f"[INFO] Basis-Seed für Plot-Puls: {RUN_SEED}")
    print(f"[INFO] Basis-Umgebungs-KW (FROG): {ENV_FROG_KW}")
    print(f"[INFO] Maximale Episodenlänge: {MAX_EP_LEN}\n")

    # Iteriere über alle zu testenden Modelle
    for model_path in MODELS:
        if not os.path.isfile(model_path):
            print(f"[WARN] Modell nicht gefunden: {model_path} – überspringe.")
            continue

        print(f"\n{'='*60}")
        print(f"VERARBEITE MODELL: {os.path.basename(model_path)}")
        print(f"{'='*60}")

        # --- 1. Einzeldurchlauf für Plots ---
        
        # Führe das RL-Modell aus
        rl_run = run_model_once(
            model_path, 
            max_steps=MAX_EP_LEN, 
            deterministic=DETERMINISTIC, 
            seed=RUN_SEED
        )
        
        # Erzeuge den initialen Referenzpuls (mit passender S-Dimension)
        initial = initial_from_env(
            ENV_KW, 
            S_ref=rl_run['w_st'].size, # Nutze S vom geladenen Modell
            seed=RUN_SEED
        )

        # --- Plot 1: Intensität vs. Phase ---
        out_png_phase = f"compare_FROG_{rl_run['label']}_S{rl_run['w_st'].size}_plot_Phase.png"
        plot_single_with_phase(initial, rl_run, out_png_phase)

        # RMS-Werte für diesen einen Plot-Durchlauf
        rms_i = calcRMSWidth(initial["inten"], ttc) / femto
        rms_r = calcRMSWidth(rl_run["inten"],   ttc) / femto
        print(f"  [Info] Plot-Run: RMS_init={rms_i:.2f} fs | RMS_RL={rms_r:.2f} fs")

        # --- Plot 2: FROG-Bilder (Initial vs. RL) ---
        
        # Berechne die physikalischen (unnormierten) Zeitfelder
        y_init_phys = pulseShaper_phys_local(initial['y_in'], Ts, initial['phase'], initial['w_st'])
        y_rl_phys   = pulseShaper_phys_local(rl_run['y_in'],   Ts, rl_run['phase'],   rl_run['w_st'])
        
        out_png_frog = f"compare_FROG_{rl_run['label']}_S{rl_run['w_st'].size}_plot_FROG.png"
        plot_frog_comparison(
            y_in_phys=y_init_phys, 
            y_out_phys=y_rl_phys, 
            out_png_frog=out_png_frog,
            title_left="Input FROG", 
            title_right=f"Output FROG (RL: {rl_run['label']})"
        )

        # --- 2. Batch-Evaluierung für Statistiken ---
        stats = evaluate_model_stats(
            model_path=model_path,
            n_pulses=N_PULSES_STATS,
            max_steps=MAX_EP_LEN,
            deterministic=DETERMINISTIC,
            base_seed=RUN_SEED
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