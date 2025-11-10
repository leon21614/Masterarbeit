"""
callbacks_rl.py

Diese Datei definiert benutzerdefinierte Callbacks für Stable Baselines3.
Callbacks sind spezielle Funktionen, die an bestimmten Stellen während
des Trainingsprozesses aufgerufen werden (z.B. am Ende jedes Schritts
oder jeder Episode), um Aktionen wie Logging oder Evaluierung durchzuführen.
"""

import os
import csv
from typing import Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

def _ensure_dir(d: Optional[str]):
    """Stellt sicher, dass ein Verzeichnis 'd' existiert. Wenn nicht, wird es erstellt."""
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# =============================================================================
# Trainings-Callback: Loggt Metriken am Ende jeder Episode
# (Speichert in CSV, TensorBoard und optional Konsole)
# =============================================================================
class RewardComponentsPerEpisodeCallback(BaseCallback):
    """
    Ein Callback, der am Ende jeder *Trainings*-Episode ausgelöst wird.
    Er sammelt die finalen Metriken aus dem 'info'-Dictionary der Umgebung
    und speichert sie in einer CSV-Datei und im TensorBoard.
    """
    def __init__(self, csv_dir: str = "logs", verbose: int = 0):
        """
        Initialisiert den Callback.
        :param csv_dir: Das Verzeichnis, in dem die CSV-Logdatei gespeichert wird.
        :param verbose: Wenn 1, werden die Episodenergebnisse auch in der Konsole ausgegeben.
        """
        super().__init__(verbose=verbose)
        self.csv_dir = csv_dir
        self.csv_path = os.path.join(csv_dir, "train_metrics.csv")
        self._fh: Optional[object] = None  # Dateihandle für die CSV-Datei
        self._writer: Optional[csv.DictWriter] = None # CSV-Schreibobjekt
        self._episode_idx: int = 0 # Zähler für die Episoden

    def _on_training_start(self) -> None:
        """
        Wird einmal zu Beginn des Trainings aufgerufen.
        Bereitet die CSV-Datei vor und schreibt den Header (Spaltenüberschriften).
        """
        _ensure_dir(self.csv_dir)
        # Prüft, ob die Datei neu erstellt werden muss (um den Header zu schreiben)
        new_file = not os.path.isfile(self.csv_path)
        self._fh = open(self.csv_path, "a", newline="")
        header = [
            "timestep", "episode", "ep_len", "ep_reward",
            "rms_in_fs", "rms_out_fs", "ratio",
        ]
        self._writer = csv.DictWriter(self._fh, fieldnames=header)
        if new_file:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        """
        Wird nach jedem einzelnen Schritt (step) in der Umgebung aufgerufen.
        """
        # Stable Baselines 3 (SB3) stellt die 'infos' und 'dones' (Episode-Ende-Flags)
        # für (vektorisierte) Umgebungen im 'self.locals'-Dictionary bereit.
        infos: List[dict] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # Gehe durch alle Umgebungen (falls vektorisiert, sonst nur eine)
        for i, done in enumerate(dones):
            # Prüfe, ob die Episode in dieser Umgebung gerade beendet wurde
            if not done:
                continue

            # Episode ist zu Ende, jetzt loggen wir die Metriken
            info = infos[i] if i < len(infos) else {}

            # 1. Standard-Metriken vom Monitor-Wrapper (von SB3 bereitgestellt)
            ep_len = int(info.get("episode", {}).get("l", np.nan)) # Episodenlänge
            ep_reward = float(info.get("episode", {}).get("r", np.nan)) # Gesamtbelohnung

            # 2. Benutzerdefinierte Metriken aus unserem 'info'-Dict (von pulse_rl_env.py)
            rms_out = float(info.get("rms_fs", np.nan))
            rms_in  = float(info.get("rms_in_fs", np.nan))

            # Berechne die Verkürzungs-Ratio (Eingang / Ausgang)
            ratio = np.nan
            if np.isfinite(rms_in) and np.isfinite(rms_out) and rms_out > 0.0:
                # Ratio > 1.0 bedeutet eine Verkürzung
                ratio = rms_in / rms_out

            # Datenzeile für die CSV-Datei vorbereiten
            row = {
                "timestep": int(self.num_timesteps),
                "episode": int(self._episode_idx),
                "ep_len": ep_len,
                "ep_reward": ep_reward,
                "rms_in_fs": rms_in,
                "rms_out_fs": rms_out,
                "ratio": ratio,
            }
            
            # 3. In CSV-Datei schreiben
            if self._writer is not None:
                self._writer.writerow(row)
                self._fh.flush() # Stellt sicher, dass die Daten sofort geschrieben werden

            # 4. In TensorBoard loggen (für Live-Grafiken)
            # Metriken werden im "train/"-Namensraum gespeichert
            self.logger.record("train/ep_len", ep_len)
            self.logger.record("train/ep_reward", ep_reward)
            self.logger.record("train/rms_in_fs", rms_in)
            self.logger.record("train/rms_out_fs", rms_out)
            self.logger.record("train/ratio", ratio)

            # 5. Optional: In Konsole ausgeben
            if self.verbose:
                try:
                    # Formatierte Ausgabe für die Konsole
                    print(
                        f"[TRAIN t={self.num_timesteps} ep={self._episode_idx}] "
                        f"RMS_in={rms_in:.2f} fs → RMS_out={rms_out:.2f} fs | "
                        f"Ratio={ratio:.3f} | Return={ep_reward:.3f} | len={ep_len}"
                    )
                except Exception:
                    # Fallback, falls Formatierung fehlschlägt (z.B. bei NaN-Werten)
                    print(
                        f"[TRAIN t={self.num_timesteps} ep={self._episode_idx}] "
                        f"RMS_in={rms_in} fs → RMS_out={rms_out} fs | "
                        f"Ratio={ratio} | Return={ep_reward} | len={ep_len}"
                    )

            self._episode_idx += 1 # Nächste Episode zählen

        return True # Muss True zurückgeben, um das Training fortzusetzen

    def _on_training_end(self) -> None:
        """
        Wird einmal am Ende des gesamten Trainings aufgerufen.
        Schließt die CSV-Datei ordnungsgemäß.
        """
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None


# =============================================================================
# Evaluations-Callback: Führt periodisch eine separate Evaluierung durch
# (Speichert in CSV, TensorBoard und Konsole)
# =============================================================================
class EvalAndLogCallback(BaseCallback):
    """
    Ein Callback, der das Training periodisch (alle 'eval_freq' Schritte)
    anhält, um eine saubere Evaluierung auf einer separaten
    Evaluierungs-Umgebung ('eval_env') durchzuführen.

    Dies ist wichtig, da die Trainings-Performance (mit Exploration)
    von der deterministischen Evaluations-Performance abweichen kann.
    """
    def __init__(
        self,
        eval_env,                 # Die separate Evaluierungs-Umgebung
        n_episodes: int = 5,      # Wie viele Episoden pro Evaluierungslauf?
        eval_freq: int = 10_000,  # Alle wie viele Trainings-Schritte evaluieren?
        csv_dir: str = "logs",    # Speicherort für die Evaluierungs-CSV
        deterministic: bool = True, # Soll die Evaluierung deterministisch (ohne Exploration) sein?
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.csv_dir = csv_dir
        self.csv_path = os.path.join(csv_dir, "eval_metrics.csv")
        self.deterministic = bool(deterministic)

        self._fh: Optional[object] = None
        self._writer: Optional[csv.DictWriter] = None

    def _on_training_start(self) -> None:
        """
        Bereitet die 'eval_metrics.csv'-Datei zu Beginn vor.
        """
        _ensure_dir(self.csv_dir)
        new_file = not os.path.isfile(self.csv_path)
        self._fh = open(self.csv_path, "a", newline="")
        header = [
            "timestep", "n_episodes",
            "rms_in_fs_mean", "rms_out_fs_mean", "ratio_mean",
            "ep_reward_mean", "ep_len_mean",
        ]
        self._writer = csv.DictWriter(self._fh, fieldnames=header)
        if new_file:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        """
        Wird nach jedem Trainings-Schritt aufgerufen.
        Prüft, ob es Zeit für die nächste Evaluierung ist.
        """
        if self.eval_freq > 0 and (self.num_timesteps % self.eval_freq == 0):
            # Es ist Zeit -> Führe die Evaluierung durch
            self._do_eval()
        return True

    def _do_eval(self):
        """
        Führt den eigentlichen Evaluierungslauf durch:
        Spielt 'n_episodes' auf der 'eval_env' durch und sammelt die Metriken.
        """
        # Listen zum Sammeln der Metriken über alle n_episodes
        rms_in_list, rms_out_list = [], []
        ep_rewards, ep_lens = [], []

        for _ in range(self.n_episodes):
            # Nutze einen zufälligen Seed für jede Episode,
            # damit wir über verschiedene Startpulse mitteln.
            seed = int(np.random.default_rng().integers(1_000_000_000))
            
            # Umgebung zurücksetzen (API: reset -> (obs, info))
            obs, info = self.eval_env.reset(seed=seed)
            # RMS des Startpulses aus dem info-Dict holen
            rms_in = float(info.get("rms_in_fs", np.nan))

            done = False
            ep_rew, ep_len = 0.0, 0

            # Schleife für eine einzelne Evaluierungs-Episode
            while not done:
                # Nutze das trainierte Modell, um die Aktion vorherzusagen
                # (deterministisch = ohne Rauschen/Exploration)
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # Führe den Schritt in der Eval-Umgebung aus
                step_out = self.eval_env.step(action)
                
                # Prüfe, ob die Gymnasium-API (5 Rückgabewerte) oder Gym-API (4) verwendet wird
                if len(step_out) == 5:
                    # Gymnasium: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    # Fallback für klassische Gym-API: (obs, reward, done, info)
                    obs, reward, done, info = step_out
                
                ep_rew += float(reward)
                ep_len += 1

            # Am Ende der Episode: Speichere die finalen Metriken
            rms_out = float(info.get("rms_fs", np.nan)) # Finale RMS
            rms_in_list.append(rms_in)
            rms_out_list.append(rms_out)
            ep_rewards.append(ep_rew)
            ep_lens.append(ep_len)

        # --- Aggregation (Mittelwerte) ---
        # Berechne die Mittelwerte über die 'n_episodes'
        rms_in_arr  = np.array(rms_in_list, dtype=float)
        rms_out_arr = np.array(rms_out_list, dtype=float)
        # Berechne die Ratio (Eingang / Ausgang) für jeden Lauf
        ratio_arr   = rms_in_arr / np.clip(rms_out_arr, 1e-20, None) # Division durch Null vermeiden

        # Datenzeile für CSV vorbereiten
        row = {
            "timestep": int(self.num_timesteps), # Aktueller Trainings-Timestep
            "n_episodes": self.n_episodes,
            "rms_in_fs_mean": float(np.nanmean(rms_in_arr)),
            "rms_out_fs_mean": float(np.nanmean(rms_out_arr)),
            "ratio_mean": float(np.nanmean(ratio_arr)),
            "ep_reward_mean": float(np.mean(ep_rewards)),
            "ep_len_mean": float(np.mean(ep_lens)),
        }

        # 1. In CSV-Datei schreiben
        if self._writer is not None:
            self._writer.writerow(row)
            self._fh.flush()

        # 2. In TensorBoard loggen
        # Metriken werden im "eval/"-Namensraum gespeichert
        self.logger.record("eval/rms_in_fs_mean", row["rms_in_fs_mean"])
        self.logger.record("eval/rms_out_fs_mean", row["rms_out_fs_mean"])
        self.logger.record("eval/ratio_mean", row["ratio_mean"])
        self.logger.record("eval/ep_reward_mean", row["ep_reward_mean"])
        self.logger.record("eval/ep_len_mean", row["ep_len_mean"])

        # 3. Optional: In Konsole ausgeben
        if self.verbose:
            try:
                print(
                    f"[EVAL t={row['timestep']}] "
                    f"RMS_in={row['rms_in_fs_mean']:.2f} fs | "
                    f"RMS_out={row['rms_out_fs_mean']:.2f} fs | "
                    f"Ratio={row['ratio_mean']:.3f} | "
                    f"Reward={row['ep_reward_mean']:.3f} | "
                    f"len={row['ep_len_mean']:.1f}"
                )
            except Exception:
                 print(
                    f"[EVAL t={row['timestep']}] "
                    f"RMS_in={row['rms_in_fs_mean']} fs | "
                    f"RMS_out={row['rms_out_fs_mean']} fs | "
                    f"Ratio={row['ratio_mean']} | "
                    f"Reward={row['ep_reward_mean']} | "
                    f"len={row['ep_len_mean']}"
                )

    def _on_training_end(self) -> None:
        """
        Wird am Ende des gesamten Trainings aufgerufen.
        Führt eine letzte Evaluierung durch und schließt die CSV-Datei.
        """
        if self.verbose:
            print("--- Training beendet. Führe finale Evaluierung durch... ---")
        self._do_eval()
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None