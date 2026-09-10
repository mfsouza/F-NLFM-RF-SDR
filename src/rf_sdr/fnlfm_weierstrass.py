"""
Módulo para síntese e geração de formas de onda F-NLFM (Fractal Non-Linear Frequency Modulation).
Pesquisa de Doutorado - UFPR
"""

import numpy as np
from typing import Tuple, Dict, Any


def generate_lfm(
    pulse_width: float,
    bandwidth: float,
    sample_rate: float,
    f_center: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera um sinal LFM (Linear Frequency Modulation / Chirp clássico) de referência.

    Parâmetros:
        pulse_width (float): Duração do pulso T em segundos.
        bandwidth (float): Largura de banda B em Hz.
        sample_rate (float): Taxa de amostragem Fs em Hz.
        f_center (float): Frequência central em Hz (banda base = 0.0).

    Retorna:
        Tuple[np.ndarray, np.ndarray]: (vetor de tempo t, sinal complexo baseband s(t))
    """
    num_samples = int(np.round(pulse_width * sample_rate))
    t = np.linspace(-pulse_width / 2.0, pulse_width / 2.0, num_samples, endpoint=False)
    chirp_rate = bandwidth / pulse_width
    
    # Fase instantânea phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
    phase = 2.0 * np.pi * (f_center * t + 0.5 * chirp_rate * (t ** 2))
    signal = np.exp(1j * phase)
    return t, signal


def generate_fnlfm(
    pulse_width: float,
    bandwidth: float,
    sample_rate: float,
    fractal_dim: float = 1.3,
    num_iterations: int = 5,
    alpha: float = 0.5,
    f_center: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Gera uma forma de onda com modulação fractal de frequência não-linear (F-NLFM).

    A modulação de frequência combina uma trajetória básica com uma perturbação fractal
    (série multi-harmônica auto-similar / tipo Weierstrass-Mandelbrot) para dispersão
    de energia de lóbulos secundários (PSLR / ISLR otimizados).

    Parâmetros:
        pulse_width (float): Duração do pulso T em segundos.
        bandwidth (float): Largura de banda B em Hz.
        sample_rate (float): Taxa de amostragem Fs em Hz.
        fractal_dim (float): Dimensão fractal efetiva D (1.0 < D < 2.0).
        num_iterations (int): Número de escalas/componentes fractais.
        alpha (float): Fator de ponderação da componente fractal versus LFM.
        f_center (float): Frequência central em Hz.

    Retorna:
        Tuple[np.ndarray, np.ndarray, dict]: (t, s(t), metadados com freq_instantanea e fase)
    """
    num_samples = int(np.round(pulse_width * sample_rate))
    t = np.linspace(-pulse_width / 2.0, pulse_width / 2.0, num_samples, endpoint=False)
    tau = t / (pulse_width / 2.0)  # Tempo normalizado [-1, 1]

    # Componente base linear: f_linear(t) = (B/2) * tau
    f_base = (bandwidth / 2.0) * tau

    # Modulação fractal somatória tipo Weierstrass
    # f_fractal(tau) = sum_n [ gamma^(-n * (2 - D)) * sin(2*pi * gamma^n * tau) ]
    gamma = 2.0  # fator de escala geométrica
    f_fractal = np.zeros_like(tau)
    
    for n in range(1, num_iterations + 1):
        amplitude_n = (gamma ** (-n * (2.0 - fractal_dim)))
        f_fractal += amplitude_n * np.sin(2.0 * np.pi * (gamma ** n) * (tau / 2.0))

    # Normalização da componente fractal
    if np.max(np.abs(f_fractal)) > 0:
        f_fractal = f_fractal / np.max(np.abs(f_fractal))

    # Frequência instantânea combinada f_inst(t) = f_base + alpha * (B/2) * f_fractal
    f_instantaneous = (1.0 - alpha) * f_base + alpha * (bandwidth / 2.0) * f_fractal + f_center

    # Fase instantânea obtida por integração numérica cumulativa da frequência instantânea
    dt = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(f_instantaneous) * dt

    # Sinal analítico em banda base (envelope complexo com amplitude constante)
    signal = np.exp(1j * phase)

    metadata = {
        "pulse_width": pulse_width,
        "bandwidth": bandwidth,
        "sample_rate": sample_rate,
        "fractal_dim": fractal_dim,
        "num_iterations": num_iterations,
        "alpha": alpha,
        "f_instantaneous": f_instantaneous,
        "phase": phase
    }

    return t, signal, metadata
