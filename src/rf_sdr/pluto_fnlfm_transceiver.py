"""
ADALM-PLUTO SDR (AD9361) Fractal-NLFM (FOSM) Pulse Generator & Matched Filter Transceiver
Generates wideband F-NLFM baseband I/Q chirps, uploads to hardware buffer, transmits and receives in loopback.
"""

import numpy as np
from scipy import signal

from src.rf_sdr.fost_fnlfm import generate_fnlfm_fost

def generate_fnlfm_baseband(fs=30e6, pulse_width=50e-6, bandwidth=20e6, beta=9.5, epsilon=0.002):
    """
    Synthesize discrete baseband complex envelope s(t) = exp(j * phi(t)) for Fractal-NLFM (FOST Engine).
    """
    t, iq_signal, meta = generate_fnlfm_fost(
        pulse_width=pulse_width,
        bandwidth=bandwidth,
        sample_rate=fs,
        beta=beta,
        fractal_dim=1.01,
        gamma=1.61803398875,
        epsilon=epsilon,
        num_layers=5,
        num_freq_points=4096
    )
    
    f_inst = meta["f_instantaneous"]
    
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
