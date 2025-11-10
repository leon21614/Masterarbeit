# pulse_rl_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Importiert die physikalischen Kernfunktionen aus pulseOpt
from pulseOpt import (
    N, Ts, ttc, femto,
    pulseShaper, calcRMSWidth,
    make_random_input_pulse,
    shaperLims,
)

class PulseShaperEnv(gym.Env):
    """
    RL-Umgebung (Environment) für die Pulsformung.

    Diese Klasse definiert die Simulationsumgebung nach dem Gymnasium-Standard.
    Sie legt fest, wie der Agent die Umgebung beobachtet (Observation),
    welche Aktionen er ausführen kann (Action) und wie er dafür
    belohnt wird (Reward).

    - Aktion: Phasenvektor (S Stützstellen), inkrementell oder absolut
    - Beobachtung:
            [downsampled Intensität(t)]
          + [downsampled Spektralamplitude |Y(ω)| (nur wenn include_spec_amp=True)]
          + [aktuelle Phase φ(ω) (S) / phi_max]
          + [RMS-Feature: rms_ratio = RMS_out / RMS_in (auf [0,1] geclippt)]
    - Reward: Belohnung basierend auf der RMS-Verbesserung.
    - WICHTIG: Phase wird nicht geclippt, sondern modulo 2π auf (-π, π] gewrappt.
    """
    # Metadaten für Gymnasium
    metadata = {"render.modes": []}

    def __init__(self,
                 S=25,
                 obs_size=256,
                 # Spektral-Features
                 include_spec_amp=False,
                 spec_size=128,
                 # Aktionen
                 incremental_actions=True,
                 action_scale=0.03,
                 phi_max=np.pi,
                 # Randomisierung
                 randomize_on_reset=True,
                 hold_pulse_episodes=5,
                 # Frequenzband (optional)
                 band_thz=None):
        """
        Initialisiert die Umgebung.
        Alle Parameter werden typischerweise aus der 'config.py' (via ENV_KW) übergeben.
        """
        super().__init__() # Standard-Initialisierung der Gym-Umgebung

        # --- Konfigurationsparameter speichern ---
        self.S = int(S)                     # Anzahl der Phasen-Stützstellen (Aktionsdimension)
        self.obs_size = int(obs_size)       # Länge des Intensitäts-Vektors in der Beobachtung
        self.include_spec_amp = bool(include_spec_amp) # Ob die Spektralamplitude Teil der Beobachtung ist
        self.spec_size = int(spec_size)     # Länge des Spektral-Vektors in der Beobachtung
        self.incremental_actions = bool(incremental_actions) # Ob Aktionen relativ (additiv) oder absolut sind
        self.action_scale = float(action_scale) # Skalierungsfaktor für Aktionen
        self.phi_max = float(phi_max)       # Maximaler Phasenwert (für Normierung der Beobachtung)
        self.randomize_on_reset = bool(randomize_on_reset) # Ob bei jedem Reset ein neuer Puls erzeugt wird
        self.hold_pulse_episodes = int(hold_pulse_episodes) # Wie viele Episoden der gleiche Puls genutzt wird

        # --- Interne Zustände der Umgebung ---
        # Der Vektor, der die aktuelle Phaseneinstellung des Shapers speichert.
        self._phase = np.zeros(self.S, dtype=np.float64) 
        
        # Definiert die Frequenz-Stützstellen (w_st) für die Phasenmaske
        if band_thz is not None:
            # Falls ein spezifisches Frequenzband (in THz) übergeben wurde
            w_lo = 2*np.pi*float(band_thz[0]) * 1e12  # THz -> rad/s
            w_hi = 2*np.pi*float(band_thz[1]) * 1e12
            self.w_st = np.linspace(w_lo, w_hi, self.S)
        else:
            # Standard: Nutzt die globalen shaperLims aus pulseOpt
            self.w_st = np.linspace(shaperLims[0], shaperLims[1], self.S)

        self.episode_counter = 0            # Zählt die Anzahl der Episoden
        self._rng = np.random.default_rng() # Eigener Zufallszahlengenerator
        self._t = 0                         # Zählt die Schritte innerhalb einer Episode
        
        # Speichert RMS-Breiten für die Belohnungsberechnung
        self._prev_rms_s = None             # RMS der vorherigen Aktion
        self._best_rms_s = None             # Beste RMS in dieser Episode
        self._rms_in_s = None               # RMS des Startpulses (Referenz)
        
        # Speicher für den (optionalen) Spektral-Teil der Beobachtung
        self._spec_obs = np.zeros(self.spec_size, dtype=np.float32) if self.include_spec_amp else None

        # --- Definition der Aktions- und Beobachtungsräume ---
        
        # Berechne die Gesamtlänge des Beobachtungsvektors
        obs_len = self.obs_size # Intensität
        if self.include_spec_amp:
            obs_len += self.spec_size # Optional: Spektrum
        obs_len += self.S # Aktuelle Phase
        obs_len += 1 # RMS-Ratio-Feature
        
        # Definition des Beobachtungsraums (was der Agent sieht)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )
        
        # Definition des Aktionsraums (was der Agent tun kann)
        # Ein Vektor der Länge S (Anzahl Stützstellen) mit Werten von -1.0 bis 1.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.S,), dtype=np.float32
        )

    # ---------- Hilfsfunktionen ----------

    @staticmethod
    def _wrap_to_pi(phi):
        """
        Mappt Phasenwerte mathematisch korrekt auf den Bereich (-π, π].
        Dies ist besser als einfaches Clippen (Abschneiden), da 3π und π 
        physikalisch dieselbe Phase sind.
        """
        return (np.asarray(phi) + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_phase(self, phase):
        """
        Wendet die aktuelle Phasenmaske auf den Eingangspuls an.
        Nutzt die pulseShaper-Funktion aus pulseOpt.
        """
        y_out = pulseShaper(self.y_in, Ts, phase, self.w_st)
        inten = np.abs(y_out) ** 2
        return y_out, inten

    def _downsample_vector(self, vec, out_len):
        """
        Skaliert einen Vektor 'vec' auf eine feste Länge 'out_len' herunter,
        indem gleichmäßig verteilte Punkte ausgewählt werden.
        """
        idx = np.linspace(0, len(vec) - 1, int(out_len)).astype(int)
        return vec[idx]

    def _compute_spec_obs(self, y_time):
        """
        Erzeugt den Spektralamplituden-Teil der Beobachtung aus dem Zeitfeld.
        - Führt eine FFT durch.
        - Nimmt nur die positiven Frequenzen (0 bis Nyquist).
        - Normiert das Spektrum auf [0, 1].
        - Skaliert es auf die 'spec_size' herunter.
        """
        if not self.include_spec_amp:
            return None
        Yw = np.fft.fft(np.fft.fftshift(y_time))
        Yamp = np.abs(Yw)
        # Positive Frequenzen (Indizes 0 bis N//2)
        pos = Yamp[: Yamp.size // 2]
        pos = pos / (np.max(pos) + 1e-20) # Normierung
        spec_ds = self._downsample_vector(pos, self.spec_size).astype(np.float32)
        return spec_ds

    def _obs_from_intensity(self, inten):
        """
        Baut den vollständigen Beobachtungsvektor (Array) zusammen, 
        den der Agent erhält.
        """
        # 1. Intensitäts-Teil (normiert und heruntergesampelt)
        obs_int = self._downsample_vector(inten, self.obs_size)
        obs_int = obs_int / (np.max(obs_int) + 1e-20)
        parts = [obs_int.astype(np.float32)]

        # 2. Spektral-Teil (optional)
        if self.include_spec_amp and self._spec_obs is not None:
            parts.append(self._spec_obs.astype(np.float32))

        # 3. Phasen-Teil (gewrappt und auf [-1, 1] skaliert)
        phi_wrapped = self._wrap_to_pi(self._phase)
        phase_scaled = (phi_wrapped / (self.phi_max + 1e-20)).astype(np.float32)
        phase_scaled = np.clip(phase_scaled, -1.0, 1.0)
        parts.append(phase_scaled)

        # 4. RMS-Ratio-Feature (Feature Engineering)
        # Gibt dem Agenten direktes Feedback, wie gut er im Verhältnis zum Start ist.
        rms_s = float(calcRMSWidth(inten, ttc))
        rms_ratio = rms_s / (self._rms_in_s + 1e-20)
        rms_ratio = float(np.clip(rms_ratio, 0.0, 1.0)) # Clip auf [0, 1]
        parts.append(np.array([rms_ratio], dtype=np.float32))

        # Fügt alle Teile zu einem einzigen Vektor zusammen
        return np.concatenate(parts, dtype=np.float32)

    # ---------- Reset-Funktion (Start einer neuen Episode) ----------
    
    def reset(self, *, seed=None, options=None):
        """
        Setzt die Umgebung auf einen neuen Startzustand zurück.
        Wird am Anfang jeder Episode aufgerufen.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0 # Schrittzähler zurücksetzen
        self.episode_counter += 1

        # --- Logik zur Pulserzeugung ---
        # Entscheidet, ob ein neuer Puls generiert werden muss,
        # basierend auf 'hold_pulse_episodes'.
        need_new_pulse = False
        if self.randomize_on_reset:
            if ((self.episode_counter - 1) % max(1, self.hold_pulse_episodes)) == 0:
                need_new_pulse = True

        # Lade neuen Puls, wenn nötig oder wenn noch kein Puls existiert
        if need_new_pulse or not hasattr(self, "y_in"):
            seed_val = int(self._rng.integers(1_000_000_000))
            # Ruft die Funktion aus pulseOpt auf, um einen neuen,
            # zufälligen Eingangspuls zu erstellen.
            self.y_in = make_random_input_pulse(seed=seed_val)

        # Berechne den Spektral-Teil der Beobachtung (bleibt für die Episode konstant)
        if self.include_spec_amp:
            self._spec_obs = self._compute_spec_obs(self.y_in)

        # --- Startzustand setzen ---
        self._phase[:] = 0.0 # Phase auf Null zurücksetzen
        _, inten0 = self._apply_phase(self._phase) # Intensität des Startpulses
        
        # Berechne die Start-RMS. Dies ist die Referenz (Baseline) für die Belohnung.
        self._rms_in_s = float(calcRMSWidth(inten0, ttc))
        self._prev_rms_s = float(self._rms_in_s)
        self._best_rms_s = float(self._rms_in_s)

        # Erzeuge die erste Beobachtung für den Agenten
        obs = self._obs_from_intensity(inten0)
        
        # Info-Dict (nützlich für Logging)
        info = {
            "rms_in_fs": self._rms_in_s / femto,
        }
        return obs, info

    # ---------- Step-Funktion (Ein Schritt in der Umgebung) ----------
    
    def step(self, action):
        """
        Führt einen Schritt in der Umgebung aus, basierend auf der 'action' des Agenten.
        """
        self._t += 1 # Schrittzähler erhöhen

        action_arr = np.asarray(action, dtype=np.float64)

        # --- Aktion anwenden ---
        # Aktualisiere die Phasenmaske basierend auf der Aktion
        if self.incremental_actions:
            # Addiere die Aktion (skaliert) zur aktuellen Phase
            self._phase += self.action_scale * action_arr
        else:
            # Setze die Phase absolut auf die (skalierte) Aktion
            self._phase = self.action_scale * action_arr

        # Wende Phasen-Wrapping an, um den Bereich (-pi, pi] einzuhalten
        self._phase = self._wrap_to_pi(self._phase)

        # --- Ergebnis berechnen ---
        # Wende die Phase an und berechne die resultierende Intensität
        _, inten = self._apply_phase(self._phase)
        # Berechne die RMS-Breite
        rms_s = float(calcRMSWidth(inten, ttc))

        # ---------- Belohnungsberechnung (Reward) ----------
        # Diese Implementierung entspricht "Belohnungsfunktion 2" aus der Arbeit.
        
        eps = 1e-20 # Kleiner Wert, um Division durch Null zu verhindern
        
        # Skalierungsfaktor 'phi'
        phi = femto * 1000 / (self._rms_in_s + eps) 

        # Belohnung 1: Relative Verbesserung im Vergleich zum START-Puls
        r_ratio = 1.0 - float(rms_s / (self._rms_in_s + eps))
        
        # Belohnung 2: Bonus, wenn ein neuer BESTWERT in dieser Episode erreicht wurde
        r_best = (self._best_rms_s - rms_s) / (self._rms_in_s + eps)

        # Gesamt-Belohnung: Kombination aus Verbesserung (r_ratio) und Bonus (r_best)
        reward = r_ratio * phi + 0.5 * r_best * phi

        # --- Zustands-Update (für nächsten Schritt) ---
        if rms_s < self._best_rms_s:
            self._best_rms_s = rms_s # Neuen Bestwert speichern
        self._prev_rms_s = rms_s

        # Episodenende wird extern durch den 'TimeLimit'-Wrapper (in train_sac.py) gehandhabt
        truncated = False 
        
        # Erzeuge die Beobachtung für den nächsten Zustand
        obs = self._obs_from_intensity(inten)
        
        # Info-Dict für Logging (wird von Callbacks genutzt)
        info = {
            "rms_fs": rms_s / femto,
            "rms_in_fs": self._rms_in_s / femto,
            "ratio": rms_s / (self._rms_in_s + 1e-20),
            "r_best": float(r_best),
            "reward": reward,
        }
        
        # Standard-Rückgabe: (Beobachtung, Belohnung, Terminated, Truncated, Info)
        return obs, reward, False, truncated, info

    # ---------- Getter-Funktionen (für Evaluierung) ----------
    
    def get_current_intensity(self):
        """
        Gibt die aktuelle Intensität und das Zeitfeld zurück.
        Wird von 'compare_rl.py' genutzt.
        """
        return self._apply_phase(self._phase)

    def get_current_phase(self):
        """
        Gibt die aktuellen Frequenz-Stützstellen und die Phasenwerte zurück.
        Wird von 'compare_rl.py' genutzt.
        """
        # _phase ist bereits auf (-π, π] gewrappt
        return self.w_st, self._phase