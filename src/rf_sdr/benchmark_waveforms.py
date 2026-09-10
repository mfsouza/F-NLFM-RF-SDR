"""
Biblioteca de Síntese e Comparação de Formas de Onda de Radar.
Implementa LFM e as principais famílias de NLFM (Tangente, Taylor-POSP, Kaiser-POSP, e Fractal-FOST FOSM).
Pesquisa de Doutorado PPGEE / UFPR - Moises Fernandes de Souza.
"""

import numpy as np
from scipy import signal, special
from typing import Tuple, Dict, Any

from src.rf_sdr.fost_fnlfm import generate_fnlfm_fost, ispa_mapping, kaiser_window_spectrum

def generate_lfm(pulse_width: float, bandwidth: float, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera chirp Linear LFM."""
    N = int(np.round(pulse_width * sample_rate))
    t = np.linspace(-pulse_width / 2.0, pulse_width / 2.0, N, endpoint=False)
    K = bandwidth / pulse_width
    f_inst = K * t
    phi = 2.0 * np.pi * (0.5 * K * t**2)
    s = np.exp(1j * phi)
    return t, s, f_inst

def generate_tangent_nlfm(pulse_width: float, bandwidth: float, sample_rate: float, alpha: float = 1.30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera Tangent NLFM (T-NLFM clássica)."""
    N = int(np.round(pulse_width * sample_rate))
    t = np.linspace(-pulse_width / 2.0, pulse_width / 2.0, N, endpoint=False)
    f_inst = (bandwidth / 2.0) * np.tan(2.0 * alpha * t / pulse_width) / np.tan(alpha)
    dt = 1.0 / sample_rate
    phi = 2.0 * np.pi * np.cumsum(f_inst) * dt
    s = np.exp(1j * phi)
    return t, s, f_inst

def generate_taylor_posp_nlfm(pulse_width: float, bandwidth: float, sample_rate: float, num_freq_points: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera NLFM com perfil espectral de Taylor / Hamming via POSP / ISPA."""
    f_grid = np.linspace(-bandwidth / 2.0, bandwidth / 2.0, num_freq_points)
    f_norm = f_grid / (bandwidth / 2.0)
    # Janela de Taylor-like suave (Hann / Hamming ponderada)
    a0 = 0.54
    a1 = 0.46
    w_psd = a0 + a1 * np.cos(np.pi * f_norm)
    w_psd = np.maximum(w_psd, 1e-6)
    
    t, s, f_inst = ispa_mapping(w_psd, f_grid, pulse_width, sample_rate)
    return t, s, f_inst

def generate_kaiser_posp_nlfm(pulse_width: float, bandwidth: float, sample_rate: float, beta: float = 9.5, num_freq_points: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera NLFM com perfil espectral de Kaiser puro (sem perturbação fractal)."""
    f_grid = np.linspace(-bandwidth / 2.0, bandwidth / 2.0, num_freq_points)
    w_psd = kaiser_window_spectrum(f_grid, bandwidth, beta=beta)
    w_psd = np.maximum(w_psd, 1e-12)
    
    t, s, f_inst = ispa_mapping(w_psd, f_grid, pulse_width, sample_rate)
    return t, s, f_inst

def calculate_radar_metrics(s: np.ndarray, sample_rate: float, mainlobe_mask_samples: int = 15) -> Dict[str, float]:
    """Calcula PSLR, ISLR, largura de feixe a -3dB e PAPR."""
    N = len(s)
    corr = np.correlate(s, s, mode='full')
    corr_mag = np.abs(corr)
    corr_mag_norm = corr_mag / np.max(corr_mag)
    corr_db = 20.0 * np.log10(corr_mag_norm + 1e-15)
    
    pk = np.argmax(corr_db)
    
    # 1. Largura de feixe a -3dB
    half_power = 10.0 ** (-3.0 / 20.0)
    idx_3db = np.where(corr_mag_norm >= half_power)[0]
    if len(idx_3db) > 0:
        width_samples_3db = idx_3db[-1] - idx_3db[0] + 1
        width_ns_3db = (width_samples_3db / sample_rate) * 1e9
    else:
        width_samples_3db = 1
        width_ns_3db = (1.0 / sample_rate) * 1e9
        
    # 2. PSLR
    mask = np.ones(len(corr_db), dtype=bool)
    mask[max(0, pk - mainlobe_mask_samples) : min(len(corr_db), pk + mainlobe_mask_samples + 1)] = False
    sidelobes_db = corr_db[mask]
    pslr = float(np.max(sidelobes_db))
    
    # 3. ISLR
    mainlobe_power = np.sum(corr_mag_norm[~mask] ** 2)
    sidelobe_power = np.sum(corr_mag_norm[mask] ** 2)
    islr = float(10.0 * np.log10(sidelobe_power / mainlobe_power + 1e-15))
    
    # 4. PAPR
    power = np.abs(s) ** 2
    papr = float(10.0 * np.log10(np.max(power) / np.mean(power) + 1e-15))
    
    return {
        "pslr_db": pslr,
        "islr_db": islr,
        "width_3db_ns": width_ns_3db,
        "papr_db": papr,
        "corr_db": corr_db,
        "pk_idx": pk
    }
