# pulse_rl_env_frog.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Importiert die physikalischen Kernfunktionen
from pulseOpt import (
    N, Ts, ttc, femto,
    pulseShaper, calcRMSWidth,
    make_random_input_pulse,
    shaperLims,
)
# Importiert die FROG-spezifische Funktion
from frogOpt import frog_from_pulse


class PulseShaperFROGEnv(gym.Env):
    """
    RL-Umgebung für Pulsformung (FROG-Setup).

    Diese Umgebung nutzt ein FROG-Bild (Spektrogramm) als einzige 
    Beobachtung (Observation) für den Agenten.
    Dies ist für Faltungsneuronale Netze (CNNs) ausgelegt.

    - Beobachtung: FROG-Bild (1, K, K) als spaces.Box, K=frog_bins
    - Aktion:       Box([-1,1], shape=(S,)) - Der Phasenvektor
    - Reward:       RMS-Verbesserung gegenüber Start & Bestwert
    - Episodenende: via TimeLimit-Wrapper (extern in train_sac_frog.py)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        S=40,
        # Aktionslogik
        incremental_actions=True,
        action_scale=0.03,
        phi_max=np.pi,
        # Reset-Logik:
        randomize_on_reset=True,
        hold_pulse_episodes=1,
        # FROG:
        frog_bins=256,
    ):
        """
        Initialisiert die FROG-Umgebung.
        Alle Parameter werden typischerweise aus 'config_frog.py' (via ENV_FROG_KW) übergeben.
        """
        super().__init__()

        # --- Hyperparameter ---
        self.S = int(S)                     # Anzahl Phasen-Stützstellen (Aktionsdimension)
        self.incremental_actions = bool(incremental_actions) # Inkrementelle Aktionen?
        self.action_scale = float(action_scale) # Skalierung der Aktionen
        self.phi_max = float(phi_max)       # Max. Phasenwert (für Skalierung/Wrap)

        self.randomize_on_reset = bool(randomize_on_reset) # Neue Pulse beim Reset?
        self.hold_pulse_episodes = int(hold_pulse_episodes) # Puls-Wiederholungen

        self.frog_bins = int(frog_bins)     # Auflösung des FROG-Bildes (K x K)
        # Stellt sicher, dass die FROG-Auflösung gültig ist
        assert 8 <= self.frog_bins <= N, "frog_bins muss im Bereich [8, N] liegen."

        # --- Interne Zustände ---
        self._phase = np.zeros(self.S, dtype=np.float64) # Aktuelle Phasenmaske
        self._prev_action = np.zeros(self.S, dtype=np.float64) # Vorherige Aktion (für Info)
        self.w_st = np.linspace(shaperLims[0], shaperLims[1], self.S) # Frequenz-Stützstellen
        self.episode_counter = 0            # Zähler für Episoden
        self._rng = np.random.default_rng() # Zufallszahlengenerator
        self._t = 0                         # Schrittzähler pro Episode
        
        # RMS-Speicher für Belohnungsberechnung
        self._prev_rms_s = None
        self._best_rms_s = None
        self._rms_in_s = None

        # --- Observation- und Action-Spaces ---
        
        # Beobachtungsraum: Ein Bild (K x K) mit einem Kanal (Graustufen)
        # Wertebereich [0, 255] (uint8), passend für CNNs in SB3
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.frog_bins, self.frog_bins), # (Kanäle, Höhe, Breite)
            dtype=np.uint8,
        )
        # Aktionsraum: Ein Vektor der Länge S mit Werten von -1.0 bis 1.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.S,), dtype=np.float32
        )

    # ---------- Hilfsfunktionen ----------
    @staticmethod
    def _wrap_to_pi(phi):
        """Mappt Phasenwerte mathematisch korrekt auf den Bereich (-π, π]."""
        return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_phase(self, phase):
        """
        Wendet die aktuelle Phasenmaske auf den Eingangspuls 'self.y_in' an
        und gibt das komplexe Zeitfeld sowie die Intensität zurück.
        """
        # Nutzt die Funktion aus pulseOpt
        y_out = pulseShaper(self.y_in, Ts, phase, self.w_st)
        inten = np.abs(y_out) ** 2
        return y_out, inten

    def _downsample(self, arr, K):
        """Hilfsfunktion: Indexbasiertes Downsampling eines Arrays auf Länge K."""
        if K <= 0:
            return np.empty((0,), dtype=np.float32)
        idx = np.linspace(0, len(arr) - 1, K).astype(int)
        return np.asarray(arr, dtype=np.float64)[idx]

    def _frog_image_from_field(self, y_complex):
        """
        Erzeugt das finale FROG-Bild (Beobachtung) aus einem komplexen Zeitfeld.
        """
        # 1. Berechne den SHG-FROG-Trace (N x N) und normiere auf [0,1]
        Ishg, _ = frog_from_pulse(y_complex, Ts, normalize=True)
        
        # 2. Downsample auf die gewünschte Auflösung (frog_bins x frog_bins)
        idx = np.linspace(0, N - 1, self.frog_bins).astype(int)
        X = Ishg[np.ix_(idx, idx)] # Indexbasiertes 2D-Downsampling
        
        # 3. Normiere erneut (falls durch Downsampling Skalierung verloren ging)
        X = X / (np.max(X) + 1e-20)
        
        # 4. Skaliere auf [0, 255] (uint8) und füge eine Kanal-Dimension hinzu
        img = (X * 255.0).astype(np.uint8)[None, :, :]  # Ergibt Shape (1, K, K)
        return img

    def _obs_from_pulse(self, y_complex):
        """
        Erzeugt die Beobachtung (hier nur das Bild) aus dem Zeitfeld.
        """
        img = self._frog_image_from_field(y_complex)  # uint8 (1,H,W)
        # Gibt ein Diktat zurück. 'reset' und 'step' greifen dann
        # mit '["image"]' darauf zu.
        return {"image": img}
    
    # ---------- Reset-Funktion (Start einer neuen Episode) ----------
    def reset(self, *, seed=None, options=None):
        """
        Setzt die Umgebung auf einen neuen Startzustand zurück.
        Wird am Anfang jeder Episode aufgerufen.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Interne Zähler zurücksetzen
        self._t = 0
        self.episode_counter += 1
        self._prev_action[:] = 0.0

        # --- Logik zur Pulserzeugung ---
        # Entscheidet, ob ein neuer Puls generiert werden muss
        need_new_pulse = False
        if self.randomize_on_reset:
            if ((self.episode_counter - 1) % max(1, self.hold_pulse_episodes)) == 0:
                need_new_pulse = True

        # Lade neuen Puls, wenn nötig oder wenn noch kein Puls existiert
        if need_new_pulse or not hasattr(self, "y_in"):
            seed_val = int(self._rng.integers(1_000_000_000))
            self.y_in = make_random_input_pulse(seed=seed_val)

        # --- Startzustand setzen ---
        self._phase[:] = 0.0 # Phase auf Null zurücksetzen
        y0, inten0 = self._apply_phase(self._phase) # Start-Puls berechnen
        
        # Berechne die Start-RMS. Dies ist die Referenz (Baseline) für die Belohnung.
        self._rms_in_s = float(calcRMSWidth(inten0, ttc))
        self._prev_rms_s = float(self._rms_in_s)
        self._best_rms_s = float(self._rms_in_s)

        # Erzeuge die erste Beobachtung für den Agenten
        obs_dict = self._obs_from_pulse(y0) # 'inten0' wird hier nicht mehr benötigt
        
        # Info-Dict (nützlich für Logging)
        info = {"rms_in_fs": self._rms_in_s / femto}
        
        # Gibt NUR das Bild und das Info-Diktat zurück
        return obs_dict["image"], info

    # ---------- Step-Funktion (Ein Schritt in der Umgebung) ----------
    def step(self, action):
        """
        Führt einen Schritt in der Umgebung aus, basierend auf der 'action' des Agenten.
        """
        self._t += 1

        action_arr = np.asarray(action, dtype=np.float64)

        # --- Aktion anwenden ---
        # Phase-Update (inkrementell oder absolut)
        if self.incremental_actions:
            self._phase += self.action_scale * action_arr
        else:
            self._phase = self.action_scale * action_arr
        
        # Wende Phasen-Wrapping an, um den Bereich (-pi, pi] einzuhalten
        self._phase = self._wrap_to_pi(self._phase)

        # --- Ergebnis berechnen ---
        # Wende die NEUE Phase an und berechne die resultierende Intensität
        y_out, inten = self._apply_phase(self._phase)
        rms_s = float(calcRMSWidth(inten, ttc)) # Berechne die NEUE RMS-Breite
        fwhm_s = 2.355 * rms_s

        # ---------- Belohnungsberechnung (Reward) ----------
        # (Identisch zur Logik in pulse_rl_env.py, "Belohnungsfunktion 2")
        eps = 1e-20
        phi = femto * 1000 / (self._rms_in_s + eps)
        r_ratio = 1.0 - float(rms_s / (self._rms_in_s + eps))        # Verbesserung ggü. Start
        r_best = (self._best_rms_s - rms_s) / (self._rms_in_s + eps) # Verbesserung ggü. Bestwert
        
        # Gesamt-Reward
        reward = r_ratio * phi + 0.5 * r_best * phi

        # --- Zustands-Update (für nächsten Schritt) ---
        if rms_s < self._best_rms_s:
            self._best_rms_s = rms_s # Neuen Bestwert speichern
        self._prev_rms_s = rms_s

        # Erzeuge die Beobachtung für den nächsten Zustand
        obs_dict = self._obs_from_pulse(y_out) # 'inten' wird hier nicht mehr benötigt

        # Info-Dict für Logging
        info = {
            "rms_fs": rms_s / femto,
            "fwhm_fs": fwhm_s / femto,
            "rms_in_fs": self._rms_in_s / femto,
            "ratio": rms_s / (self._rms_in_s + 1e-20),
            "r_best": float(r_best),
            "reward": float(reward),
        }
        self._prev_action = action_arr.copy()

        # Episodenende wird extern durch den 'TimeLimit'-Wrapper gehandhabt
        # (terminated=False, truncated=False)
        
        # Standard-Rückgabe: (Beobachtung, Belohnung, Terminated, Truncated, Info)
        # Gibt NUR das Bild zurück
        return obs_dict["image"], reward, False, False, info

    # ---------- Getter-Funktionen (für Evaluierung) ----------
    def get_current_intensity(self):
        """
        Gibt die aktuelle Intensität und das Zeitfeld zurück.
        Wird von 'compare_rl.py' genutzt (falls dieses angepasst wird).
        """
        return self._apply_phase(self._phase)

    def get_current_phase(self):
        """
        Gibt die aktuellen Frequenz-Stützstellen und die Phasenwerte zurück.
        Wird von 'compare_rl.py' genutzt (falls dieses angepasst wird).
        """
        return self.w_st, self._phase