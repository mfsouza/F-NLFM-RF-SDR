"""
Síntese Analítica e Numérica da Modulação F-NLFM Baseada na Teoria FOST (Fractal Oscillation Source Theory).
Pesquisa de Doutorado PPGEE / UFPR - Moises Fernandes de Souza.
Orientador: Prof. Ph.D. Horacio Tertuliano Filho.

Formulações Implementadas:
- Semente Fractal de Weierstrass-Mandelbrot multi-escala com Proporção Áurea (Eq. 5.5).
- Função Atenuadora Central Notch (Eq. 5.6) para isolamento da portadora e integridade do lóbulo principal.
- Hibridização Espectral Kaiser-FOST (Eq. 5.7).
- Mapeamento Tempo-Frequência via Algoritmo de Fase Estacionária Inversa (ISPA / Inverse POSP).
"""

import numpy as np
from scipy import special
from typing import Tuple, Dict, Any, Optional


def kaiser_window_spectrum(f: np.ndarray, bandwidth: float, beta: float = 9.5) -> np.ndarray:
    """
    Calcula a função esquelética de base no domínio da frequência da janela de Kaiser-Bessel.

    Parâmetros:
        f (np.ndarray): Vetor de frequências em Hz [-B/2, B/2].
        bandwidth (float): Largura de banda B em Hz.
        beta (float): Parâmetro de forma de Kaiser (padrão beta = 9.5).

    Retorna:
        np.ndarray: Densidade espectral de base W_K(f).
    """
    f_norm = f / (bandwidth / 2.0)
    f_norm = np.clip(f_norm, -1.0, 1.0)
    arg = beta * np.sqrt(np.maximum(1.0 - f_norm**2, 0.0))
    w_k = special.i0(arg) / special.i0(beta)
    return w_k


def weierstrass_mandelbrot_seed(
    f: np.ndarray,
    bandwidth: float,
    fractal_dim: float = 1.01,
    gamma: float = 1.61803398875,  # Proporção Áurea
    num_layers: int = 5
) -> np.ndarray:
    """
    Gera a semente fractal M_f(f) baseada na série de Weierstrass-Mandelbrot (Equação 5.5).

    Parâmetros:
        f (np.ndarray): Vetor de frequências em Hz.
        bandwidth (float): Largura de banda B em Hz.
        fractal_dim (float): Dimensão fractal D (D = 1.01 para fase ultra-suave).
        gamma (float): Taxa de crescimento geométrica indexada pela Proporção Áurea (1.618).
        num_layers (int): Número de camadas/escalas fractais (padrão N = 5).

    Retorna:
        np.ndarray: Semente fractal M_f(f).
    """
    m_f = np.zeros_like(f, dtype=np.float64)
    for n in range(1, num_layers + 1):
        amplitude_n = gamma ** ((fractal_dim - 2.0) * n)
        harmonic_phase = 2.0 * np.pi * (gamma ** n) * (f / bandwidth)
        m_f += amplitude_n * np.cos(harmonic_phase)
    return m_f


def central_notch_filter(f: np.ndarray, bandwidth: float, exponent: int = 4) -> np.ndarray:
    """
    Calcula a função de atenuação cônica Central Notch Lambda(f) (Equação 5.6).
    Garante atenuação na componente central (f=0) e atuação unitária nas bordas extremas da banda.

    Parâmetros:
        f (np.ndarray): Vetor de frequências em Hz [-B/2, B/2].
        bandwidth (float): Largura de banda B em Hz.
        exponent (int): Expoente de decaimento (padrão = 4).

    Retorna:
        np.ndarray: Perfil atenuador Lambda(f) [0 no centro, 1 nas bordas].
    """
    # Nas bordas (|f| = B/2) -> Lambda = 1. No centro (f = 0) -> Lambda = 0.
    f_norm = f / (bandwidth / 2.0)
    lambda_f = np.abs(f_norm) ** exponent
    return np.clip(lambda_f, 0.0, 1.0)


def synthesize_kaiser_fost_spectrum(
    f: np.ndarray,
    bandwidth: float,
    beta: float = 9.5,
    fractal_dim: float = 1.01,
    gamma: float = 1.61803398875,
    epsilon: float = 0.002,
    num_layers: int = 5
) -> np.ndarray:
    """
    Hibridização Espectral: Injeção da semente fractal na janela de Kaiser-Bessel (Equação 5.7).

    W_f_fractal(f) = W_K(f) * [ 1 + epsilon * (M_f(f) / max|M_f|) * Lambda(f) ]

    Parâmetros:
        f (np.ndarray): Vetor de frequências discretizadas em Hz.
        bandwidth (float): Largura de banda B em Hz.
        beta (float): Parâmetro Kaiser-Bessel.
        fractal_dim (float): Dimensão fractal D.
        gamma (float): Fator de escala áureo.
        epsilon (float): Fator de acoplamento cirúrgico (0.002 = 0.2%).
        num_layers (int): Número de camadas da semente fractal.

    Retorna:
        np.ndarray: Densidade espectral de potência alvo W_f_fractal(f).
    """
    w_k = kaiser_window_spectrum(f, bandwidth, beta=beta)
    m_f = weierstrass_mandelbrot_seed(f, bandwidth, fractal_dim=fractal_dim, gamma=gamma, num_layers=num_layers)
    m_f_norm = m_f / np.max(np.abs(m_f)) if np.max(np.abs(m_f)) > 0 else m_f
    lambda_f = central_notch_filter(f, bandwidth)

    w_f_fractal = w_k * (1.0 + epsilon * m_f_norm * lambda_f)
    return np.maximum(w_f_fractal, 1e-12)


def ispa_mapping(
    w_f: np.ndarray,
    f_grid: np.ndarray,
    pulse_width: float,
    sample_rate: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Algoritmo ISPA (Inverse Stationary Phase Approximation).
    Mapeia a Densidade Espectral de Potência W(f) na lei de frequência instantânea f_i(t)
    e sintetiza o sinal analítico s(t) com envelope de amplitude estritamente constante.

    Parâmetros:
        w_f (np.ndarray): Densidade espectral de potência discretizada.
        f_grid (np.ndarray): Grade de frequências em Hz.
        pulse_width (float): Duração temporal do pulso T em segundos.
        sample_rate (float): Taxa de amostragem no tempo Fs em Hz.

    Retorna:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (t, s(t), f_instantaneous(t))
    """
    # Integral cumulativa normalizada da energia espectral: C_f(f) = int_{-B/2}^{f} W(nu) dnu / E_total
    cdf_f = np.cumsum(w_f)
    cdf_f = (cdf_f - cdf_f[0]) / (cdf_f[-1] - cdf_f[0])  # Mapeia para [0, 1]

    # Vetor de tempo discreto: t in [-T/2, T/2]
    num_samples = int(np.round(pulse_width * sample_rate))
    t = np.linspace(-pulse_width / 2.0, pulse_width / 2.0, num_samples, endpoint=False)
    tau_norm = (t + pulse_width / 2.0) / pulse_width  # Mapeia tempo para [0, 1]

    # Inversão numérica: interpola f_grid a partir de cdf_f = tau_norm
    f_instantaneous = np.interp(tau_norm, cdf_f, f_grid)

    # Integração numérica cumulativa da frequência para obtenção da fase instantânea phi(t)
    dt = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(f_instantaneous) * dt

    # Sinal analítico em banda base com envelope de amplitude unitária estritamente constante (0 dB mismatch)
    s = np.exp(1j * phase)

    return t, s, f_instantaneous


def generate_fnlfm_fost(
    pulse_width: float = 10e-6,
    bandwidth: float = 30e6,
    sample_rate: float = 100e6,
    beta: float = 9.5,
    fractal_dim: float = 1.01,
    gamma: float = 1.61803398875,
    epsilon: float = 0.002,
    num_layers: int = 5,
    num_freq_points: int = 4096
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Gera o sinal F-NLFM de Excelência Base-Kaiser / FOST da tese de doutorado (PSLR ~ -51.44 dB).

    Parâmetros:
        pulse_width (float): Duração do pulso T (ex: 10 us ou 40 us para SST).
        bandwidth (float): Largura de banda B em Hz (ex: 30 MHz para radar Banda C).
        sample_rate (float): Taxa de amostragem Fs em Hz (ex: 100 MSPS).
        beta (float): Parâmetro da janela Kaiser (9.5).
        fractal_dim (float): Dimensão fractal D (1.01).
        gamma (float): Fator de escala áureo (1.618).
        epsilon (float): Acoplamento fractal (0.002 = 0.2%).
        num_layers (int): Camadas harmônicas de Weierstrass (5).
        num_freq_points (int): Resolução da grade de frequências no ISPA.

    Retorna:
        Tuple[np.ndarray, np.ndarray, dict]: (t, s(t), metadados do sinal)
    """
    f_grid = np.linspace(-bandwidth / 2.0, bandwidth / 2.0, num_freq_points)
    w_psd = synthesize_kaiser_fost_spectrum(
        f=f_grid,
        bandwidth=bandwidth,
        beta=beta,
        fractal_dim=fractal_dim,
        gamma=gamma,
        epsilon=epsilon,
        num_layers=num_layers
    )

    t, s, f_inst = ispa_mapping(
        w_f=w_psd,
        f_grid=f_grid,
        pulse_width=pulse_width,
        sample_rate=sample_rate
    )

    metadata = {
        "pulse_width": pulse_width,
        "bandwidth": bandwidth,
        "sample_rate": sample_rate,
        "beta": beta,
        "fractal_dim": fractal_dim,
        "gamma": gamma,
        "epsilon": epsilon,
        "num_layers": num_layers,
        "f_grid": f_grid,
        "w_psd": w_psd,
        "f_instantaneous": f_inst
    }

    return t, s, metadata
