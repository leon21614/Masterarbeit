import numpy as np
import matplotlib.pyplot as plt

# --- Import aus pulseOpt.py ---
from pulseOpt import (
    N, Ts, ttc, femto,
    make_random_input_pulse,
    calcRMSWidth,
)

# =============================================================================
#                      Hilfsfunktionen & Metriken
# =============================================================================

def pulse_intensity(yt, normalize=True):
    """|E(t)|^2 und Zeitachse ttc zurückgeben."""
    I = np.abs(yt) ** 2
    if normalize:
        I = I / (np.max(I) + 1e-20)
    return I.astype(np.float32), ttc

def pulse_metrics(yt):
    """RMS (fs), FWHM (fs), Energie (arb.) der Zeit-Intensität |E(t)|^2."""
    I = np.abs(yt) ** 2
    rms_s = calcRMSWidth(I, ttc)
    rms_fs = rms_s / femto
    fwhm_fs = 2.355 * rms_fs
    energy = float(np.trapz(I, x=ttc))
    return dict(rms_fs=rms_fs, fwhm_fs=fwhm_fs, energy=energy)

def _circshift(x, shift):
    """Zirkularer Shift."""
    return np.roll(x, int(shift) % len(x))

# =============================================================================
#                      SHG-FROG (Amplitude & Intensität)
# =============================================================================

def create_shg_ampl(E_t, Ts):
    """
    Komplexe SHG-FROG-Amplitude A(ω, τ) (Form: N x N).
    Spalte = fester Delay τ, Zeile = Frequenz-Bin (fft-Bin).
    """
    E = np.asarray(E_t, dtype=np.complex128)
    Nloc = E.size
    assert Nloc == N, "E_t muss Länge N haben (gleiches Grid wie pulseOpt)."

    # Delays: -N/2 .. N/2-1
    delay_idx = np.arange(-Nloc // 2, Nloc // 2, dtype=int)
    tau_axis = delay_idx * Ts

    # Phasenkorrektur für SHG (Frequenzverdopplung)
    shift_factor = np.exp(-1j * 2.0 * tau_axis)

    A = np.zeros((Nloc, Nloc), dtype=np.complex128)
    for col, d_idx in enumerate(delay_idx):
        E_shifted = _circshift(E, d_idx)
        argument = E * E_shifted * shift_factor[col]
        A[:, col] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(argument)))

    return A

def frog_from_pulse(yt, Ts, normalize=True):
    """
    Aus einem Zeitpuls yt das SHG-FROG-Intensity-Trace berechnen.
    Rückgabe:
      Ishg: (N x N) Intensität (optional auf max=1 normiert)
      tau_axis: (N,) Verzögerungsachse in s
    """
    A = create_shg_ampl(yt, Ts)
    Ishg = np.abs(A) ** 2
    if normalize:
        Ishg = Ishg / (np.max(Ishg) + 1e-20)

    delay_idx = np.arange(-len(yt)//2, len(yt)//2)
    tau_axis = delay_idx * Ts
    return Ishg.astype(np.float32), tau_axis

def frog_to_obs(Ishg, scale_to=(0.0, 10.0), flatten=True, eps=1e-20):
    """
    Skaliert ein FROG-Bild auf [a,b] und gibt optional einen flachen Vektor zurück.
    Default: [0,10] – kompatibel zu RL-Observations.
    """
    a, b = scale_to
    X = np.asarray(Ishg, dtype=np.float32)
    X = X / (np.max(X) + eps)
    X = a + (b - a) * X
    return X.flatten() if flatten else X

def frog_and_pulse(yt, Ts, normalize=True):
    """
    Wrapper: FROG + Zeit-Intensität + Metriken.
    Rückgabe: Ishg, tau_axis, I_t, t_axis, metrics
    """
    Ishg, tau_axis = frog_from_pulse(yt, Ts,  normalize=normalize)
    I_t, t_axis = pulse_intensity(yt, normalize=True)
    metrics = pulse_metrics(yt)
    return Ishg, tau_axis, I_t, t_axis, metrics

# =============================================================================
#                              Demo (Direktrun)
# =============================================================================

if __name__ == "__main__":
    print("[frogOpt] Demo: Random-Puls -> FROG + |E(t)|^2 + RMS/FWHM/Energie")

    # === Random-Puls erzeugen (jeder Lauf neu) ===
    yt = make_random_input_pulse(seed=None)

    # === FROG + Zeit-Intensität + Metriken berechnen ===
    Ishg, tau_axis, I_t, t_axis, met = frog_and_pulse(yt, Ts)

    print(f"RMS  = {met['rms_fs']:.2f} fs")
    print(f"FWHM = {met['fwhm_fs']:.2f} fs")
    print(f"Ener. = {met['energy']:.3g} (arb.)")

    # === Plot: links FROG, rechts Zeit-Intensität ===
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))

    im = axs[0].imshow(
        Ishg, aspect='auto', origin='lower',
        extent=[tau_axis[0]/femto, tau_axis[-1]/femto, 0, N-1]
    )
    axs[0].set_xlabel("Delay τ (fs)")
    axs[0].set_ylabel("freq-bin (index)")
    axs[0].set_title("SHG-FROG (Random)")
    cbar = plt.colorbar(im, ax=axs[0], pad=0.02)
    cbar.set_label("norm. Intensity")

    axs[1].plot(t_axis / femto, I_t, lw=2)
    axs[1].set_xlabel("Zeit (fs)")
    axs[1].set_ylabel("Intensität (norm.)")
    axs[1].set_title(f"|E(t)|^2 – RMS={met['rms_fs']:.1f} fs")
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("[frogOpt] Demo fertig.")
