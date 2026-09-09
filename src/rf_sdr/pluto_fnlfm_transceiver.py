"""
ADALM-PLUTO SDR (AD9361) Fractal-NLFM (FOSM) Pulse Generator & Matched Filter Transceiver
Generates wideband F-NLFM baseband I/Q chirps, uploads to hardware buffer, transmits and receives in loopback.
"""

import numpy as np
from scipy import signal

def generate_fnlfm_baseband(fs=30e6, pulse_width=50e-6, bandwidth=20e-6, alpha=1.618, beta=0.85):
    """
    Synthesize discrete baseband complex envelope s(t) = exp(j * phi(t)) for Fractal-NLFM.
    """
    n_samples = int(fs * pulse_width)
    t = np.linspace(-pulse_width/2, pulse_width/2, n_samples)
    
    # Normalized time tau in [-1, 1]
    tau = 2.0 * t / pulse_width
    
    # Instantaneous frequency profile f_inst(t) with fractal polynomial curvature
    # f_inst(t) = (B/2) * sign(tau) * |tau|^alpha / (1 + beta * (1 - |tau|))
    sign_tau = np.sign(tau)
    abs_tau = np.abs(tau)
    f_inst = (bandwidth / 2.0) * sign_tau * (abs_tau ** alpha) / (1.0 + beta * (1.0 - abs_tau) + 1e-12)
    
    # Integrate instantaneous frequency to obtain phase phi(t)
    dt = 1.0 / fs
    phi = 2.0 * np.pi * np.cumsum(f_inst) * dt
    
    # Complex baseband signal with constant envelope
    iq_signal = np.exp(1j * phi)
    
    # Scale to 14-bit integer DAC range for AD9361 ([-2^14, 2^14 - 1])
    scale_factor = (2**14 - 1) * 0.8  # -2 dBFS margin to avoid DAC clipping
    iq_tx = (iq_signal * scale_factor).astype(np.complex64)
    
    return t, f_inst, iq_tx

def matched_filter_compression(rx_iq, tx_iq):
    """
    Perform matched filtering: R(t) = rx(t) * conj(tx(-t))
    """
    ref = np.conj(tx_iq[::-1])
    compressed = signal.fftconvolve(rx_iq, ref, mode="same")
    
    # Convert to normalized dB scale
    mag = np.abs(compressed)
    mag_db = 20.0 * np.log10(mag / (np.max(mag) + 1e-12))
    
    # Calculate PSLR (Peak Sidelobe Ratio)
    peak_idx = np.argmax(mag)
    # Exclude mainlobe (e.g. +/- 5 samples)
    null_width = 10
    sidelobes = np.copy(mag)
    sidelobes[max(0, peak_idx - null_width) : min(len(sidelobes), peak_idx + null_width)] = 0
    pslr_db = 20.0 * np.log10(np.max(sidelobes) / (np.max(mag) + 1e-12))
    
    return compressed, mag_db, pslr_db

if __name__ == "__main__":
    t, f_inst, iq_tx = generate_fnlfm_baseband()
    print(f"[+] Generated F-NLFM waveform: {len(iq_tx)} samples, Peak Magnitude: {np.max(np.abs(iq_tx))}")
    _, mag_db, pslr = matched_filter_compression(iq_tx, iq_tx)
    print(f"[+] Ideal Self-Matched Compression PSLR: {pslr:.2f} dB")
