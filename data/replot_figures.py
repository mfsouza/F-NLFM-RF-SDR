"""
Standalone Replotting Script for F-NLFM Radar Datasets
Usage: python replot_figures.py
Plots and saves all publication figures directly from raw dataset files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_microwave_benchmark():
    data = np.load(os.path.join(DATA_DIR, "dataset_1_5_8ghz_microwave_benchmark.npz"))
    tau_us = data["tau_us"]
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tau_us, data["corr_lfm_db"], 'b-', lw=1.5, label='LFM (PSLR = -17.34 dB)')
    plt.plot(tau_us, data["corr_tan_db"], 'm-.', lw=1.5, label='Tangent NLFM (PSLR = -36.97 dB)')
    plt.plot(tau_us, data["corr_kaiser_db"], 'g--', lw=1.8, label='Kaiser-POSP (PSLR = -52.57 dB)')
    plt.plot(tau_us, data["corr_fosm_db"], 'r-', lw=2.0, label='Proposed FOSM (PSLR = -52.76 dB)')
    plt.xlim([-2.0, 2.0])
    plt.ylim([-65, 5])
    plt.xlabel('Delay τ (μs)', fontsize=12)
    plt.ylabel('Normalized Amplitude (dB)', fontsize=12)
    plt.title('Calibrated Pulse Compression Benchmark at 5.800 GHz (Pluto SDR)', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "replot_fig_microwave_benchmark.png"), dpi=300)
    print("Saved replot_fig_microwave_benchmark.png")

def plot_doppler_curves():
    data = np.load(os.path.join(DATA_DIR, "dataset_6_doppler_mismatch_and_filter_bank.npz"))
    fd_khz = data["fd_khz"]
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(fd_khz, data["loss_lfm_db"], 'b--', lw=2.0, label='LFM (Range coupling bias)')
    plt.plot(fd_khz, data["loss_tangent_db"], 'm-.', lw=2.0, label='Tangent NLFM')
    plt.plot(fd_khz, data["loss_fosm_db"], 'r-', lw=2.5, label='Proposed FOSM / Kaiser-POSP')
    plt.xlim([0, 60])
    plt.ylim([0, 6])
    plt.xlabel('Doppler Frequency f_d (kHz)', fontsize=12)
    plt.ylabel('Peak Doppler Mismatch Loss (dB)', fontsize=12)
    plt.title('Doppler Mismatch Loss vs. Radial Velocity Shift', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "replot_fig_doppler_loss.png"), dpi=300)
    print("Saved replot_fig_doppler_loss.png")

if __name__ == '__main__':
    plot_microwave_benchmark()
    plot_doppler_curves()
    print("All figures replotted successfully from raw data!")
