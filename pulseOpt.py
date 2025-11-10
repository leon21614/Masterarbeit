# pulseOpt.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp
import scipy.optimize as opt   # für Nelder-Mead

# ---------- Grundkonstanten ----------
femto = 1e-15
angFregTHz = 2 * np.pi * 1e12  # Winkelkreisfrequenz in rad/s pro THz

# ---------- Gitter / Achsen ----------
N = 512
Ts = 10 * femto
T0 = N * Ts
idxVec = np.arange(0, N)
tt = idxVec * Ts
ttc = tt - T0 / 2
ws = 2 * np.pi / Ts           # Nyquist-Kreisfrequenz
w0 = ws / N
ww = idxVec * w0

# ---------- Shaper-Grenzen ----------
shaperLims = (0, ws / 3)      # in diesem Bereich darf die Phase verändert werden

# ---------- „Beispiel“-Spektrum (wie im Referenzcode) ----------
wp = ws / 8
alpha = 15 * angFregTHz ** 2   # formt die beiden Gauß-Peaks

# ======================================================================
#                            Nützliche Funktionen
# ======================================================================

# in pulseOpt.py (oben sicherstellen):
# from scipy import interpolate as intp
# import numpy as np

def pulseShaper(ytin, Ts, phStuetz, wStuetz, normalize=True, return_scale=False):
    """
    Wendet eine Phasenmaske im Frequenzbereich auf das Zeitfeld ytin an.
    """
    Nloc  = len(ytin)
    ws    = 2 * np.pi / Ts
    w0    = ws / Nloc
    ww    = np.arange(0, Nloc) * w0

    # FFT (Zeit -> Frequenz)
    Yw = np.fft.fft(np.fft.fftshift(ytin))
    
    # pulseOpt.py -> pulseShaper(...)
    phIntp = intp.CubicSpline(wStuetz, phStuetz, extrapolate=False)
    ph = np.zeros(Nloc)
    # Maske, die exakt den Bereich der Stützstellen abdeckt
    mask = (ww >= wStuetz[0]) & (ww <= wStuetz[-1])
    ph[mask] = phIntp(ww[mask])


    # numerisch stabil auf (-π, π] begrenzen
    ph = (ph + np.pi) % (2.0 * np.pi) - np.pi
    
    # nur Phase ändern (Amplitude bleibt erhalten)
    Yw *= np.exp(1j * ph)

    # zurück in die Zeitdomäne
    ytout_raw = np.fft.fftshift(np.fft.ifft(Yw))

    # Peak-Normierung optional
    peak = np.max(np.abs(ytout_raw)) + 1e-20
    if normalize:
        y = ytout_raw / peak
    else:
        y = ytout_raw

    if return_scale:
        return y, float(peak)
    return y


def pulseShaper_phys(ytin, Ts, phStuetz, wStuetz):
    """
    Physikalische Variante ohne Peak-Normierung.
    Semantischer Wrapper für pulseShaper(..., normalize=False).
    """
    return pulseShaper(ytin, Ts, phStuetz, wStuetz, normalize=False, return_scale=False)

def calcRMSWidth(yd, tt):

    Nloc = len(yd)
    if len(tt) != Nloc:
        raise ValueError("yd and tt must have the same length")

    inten = np.asarray(yd, dtype=float)
    inten = np.maximum(inten, 0.0)

    norm_fac = np.sum(inten) + 1e-20
    mom1 = float(np.sum(tt * inten) / norm_fac)
    mom2 = float(np.sum((tt ** 2) * inten) / norm_fac)
    md = mom2 - mom1 ** 2
    if md < 0:
        return -1.0
    return float(np.sqrt(md))

def _calc_tbp_rms(I_t, t_axis, y_t, Ts):
    """
    RMS-basiertes Zeit-Bandbreite-Produkt (TBP) mit deinen FFT-Konventionen.
    Eingaben
      I_t    : Intensität |y(t)|^2 (gleiche Länge wie y_t)
      t_axis : Zeitachse (Sekunden), z.B. ttc
      y_t    : komplexes Zeitfeld y(t) (nicht nur Intensität!)
      Ts     : Abtastabstand in Sekunden

    Rückgabe
      sigma_t [s], sigma_w [rad/s], tbp = sigma_t * sigma_w
    """
    I = np.asarray(I_t, dtype=float)
    t = np.asarray(t_axis, dtype=float)
    y = np.asarray(y_t, dtype=complex)

    # --- σ_t aus der Zeitintensität ---
    I = np.maximum(I, 0.0)
    m0 = np.sum(I) + 1e-20
    mu_t = float(np.sum(t * I) / m0)
    var_t = float(np.sum(((t - mu_t) ** 2) * I) / m0)
    var_t = max(var_t, 0.0)
    sigma_t = float(np.sqrt(var_t))

    # --- σ_ω aus dem Spektrum ---
    N = int(y.size)
    ws = 2.0 * np.pi / Ts         # Nyquist-Kreisfrequenz
    w0 = ws / N
    omega = np.arange(0, N) * w0  # 0 … ws

    Yw = np.fft.fft(np.fft.fftshift(y))
    Sw = np.abs(Yw) ** 2

    m0w = np.sum(Sw) + 1e-20
    mu_w = float(np.sum(omega * Sw) / m0w)
    var_w = float(np.sum(((omega - mu_w) ** 2) * Sw) / m0w)
    var_w = max(var_w, 0.0)
    sigma_w = float(np.sqrt(var_w))

    tbp = float(sigma_t * sigma_w)
    return sigma_t, sigma_w, tbp


# ======================================================================
#        Pulserzeugung (Amplitude fix + GDD/TOD + Jitter)
# ======================================================================

def make_random_input_pulse(
    seed=None,
    Sref_local=25,
    #gdd=0.0 / angFregTHz**2,     # s^2
    gdd=0.3 / angFregTHz**2,     # s^2
    #tod=0.0 / angFregTHz**3,     # s^3
    tod=0.1 / angFregTHz**3,     # s^3
    jitter_scale=1.0             # skaliert die Jitter-Stärke relativ zum Beispiel
):
    """
    Erzeugt einen Eingangspuls:
      - Spektrum: Summe aus 2 Gauß-Peaks fix um wp
      - Phase: GDD + TOD + gauss-gewichteter Zufallsjitter im Shaper-Fenster
    Rückgabe: zeitlicher Puls yt (komplex), normiert auf max|yt| = 1.
    """
    rng = np.random.default_rng(seed)

    # --- Spektrale Amplitude  ---
    YwrefAmpl = np.exp(-1 / alpha * (ww - wp) ** 2) \
              + 0.6 * np.exp(-2 / (1 * alpha) * (ww - 1.5 * wp) ** 2)
    YwrefAmpl = YwrefAmpl / (np.max(YwrefAmpl) + 1e-20)

    # --- Phase: GDD + TOD + gedämpfter Zufallsjitter an Stützstellen ---
    Sref_loc = int(Sref_local)
    #wStuetz_loc = np.linspace(shaperLims[0], shaperLims[1], Sref_loc)
    wStuetz_loc = np.linspace(0.0, (2*np.pi/Ts)/3.0, Sref_loc)  # 0 … ws/3

    phStuetz = 0.5 * gdd * (wStuetz_loc - wp) ** 2 + (1.0 / 6.0) * tod * (wStuetz_loc - wp) ** 3

    phStuetzRd = np.pi * (2.0 * rng.random(Sref_loc) - 1.0) \
                 * np.exp(-0.5 / alpha * (wStuetz_loc - wp) ** 2) * float(jitter_scale)
    phStuetz = phStuetz + phStuetzRd

    # Auf volle Frequenzachse fortschreiben (außerhalb Shaper-Fenster = 0)
    phIntp_loc = intp.CubicSpline(wStuetz_loc, phStuetz)
    phase_loc = np.zeros_like(ww)
    mask = ww < wStuetz_loc[-1]
    phase_loc[mask] = phIntp_loc(ww[mask])

    # --- Zeitpuls erzeugen ---
    Yw = YwrefAmpl * np.exp(1j * phase_loc)
    yt = np.fft.fftshift(np.fft.ifft(Yw))
    yt /= (np.max(np.abs(yt)) + 1e-20)
    return yt


# ======================================================================
#                      Demo (nur bei Direktrun)
# ======================================================================

if __name__ == "__main__":
    # Demo: Beispiel-Puls erzeugen und anzeigen (ohne Optimierung)
    print("[pulseOpt] Demo läuft ... (Beispiel-Puls, ohne Optimierung)")

    # Puls erzeugen (Seed optional für Reproduzierbarkeit)
    yt0 = make_random_input_pulse(seed=None, Sref_local=25, jitter_scale=0.0)

    I0 = np.abs(yt0) ** 2
    I0n = I0 / (np.max(I0) + 1e-20)
    rms0_fs = calcRMSWidth(I0, ttc) / femto
    fwhm0_fs = 2.355 * rms0_fs
    E0 = float(np.trapz(I0, x=ttc))
    print(f"Start:   RMS={rms0_fs:.2f} fs | FWHM={fwhm0_fs:.2f} fs | E={E0:.3g}")

    # Plot: Intensität und Zeitphase des Startpulses
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))

    axs[0].plot(ttc / femto, I0n, label=f"Start (RMS {rms0_fs:.1f} fs)", lw=2)
    axs[0].set_xlabel("Zeit (fs)")
    axs[0].set_ylabel("Intensität (norm.)")
    axs[0].set_title("Beispiel-Puls")
    axs[0].legend(loc="upper right")
    axs[0].grid(alpha=0.3)

    # Zeitphase des Startpulses
    axs[1].plot(ttc / femto, np.unwrap(np.angle(yt0)))
    axs[1].set_xlabel("Zeit (fs)")
    axs[1].set_ylabel("Phase (rad)")
    axs[1].set_title("Zeitphase des Startpulses")
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("[pulseOpt] Demo fertig.")
