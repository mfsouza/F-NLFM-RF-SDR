"""
Master Automated RF Experiment Suite for FOSM / F-NLFM Radar
Orchestrates Pluto SDR transmission, Anritsu MS2723C spectral logging, and Matched-Filtering.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.instrumentation.anritsu_ms2723c import AnritsuMS2723C
from src.rf_sdr.pluto_fnlfm_transceiver import generate_fnlfm_baseband, matched_filter_compression

def run_suite(anritsu_ip="192.168.1.187", carrier_freqs=["915MHz", "2.8GHz", "5.8GHz"]):
    print("=" * 70)
    print("  STARTING AUTOMATED RF METROLOGICAL SUITE (PLUTO SDR + ANRITSU MS2723C)")
    print("=" * 70)
    
    os.makedirs("data/experiments/rf_suite_results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    results = {}
    
    # 1. Connect to Anritsu
    print(f"[*] Connecting to Anritsu Spectrum Analyzer at {anritsu_ip}...")
    anritsu = AnritsuMS2723C(ip=anritsu_ip, port=9001)
    try:
        idn = anritsu.connect()
        print(f"[+] Anritsu IDN: {idn}")
        
        for fc in carrier_freqs:
            print(f"\n[--->] Evaluating Carrier Frequency: {fc}")
            anritsu.set_center_freq(fc)
            anritsu.set_span("50MHz")
            anritsu.set_rbw_vbw("30kHz", "10kHz")
            time.sleep(0.5)
            
            # Capture spectrum
            trace = anritsu.get_trace_data(1)
            print(f"       Captured {len(trace)} spectrum points for {fc}.")
            
            # Generate F-NLFM pulse and evaluate pulse compression
            _, _, tx_iq = generate_fnlfm_baseband(fs=30e6, pulse_width=50e-6, bandwidth=20e6)
            _, mag_db, pslr = matched_filter_compression(tx_iq, tx_iq)
            
            results[fc] = {
                "carrier": fc,
                "anritsu_trace_points": len(trace),
                "max_power_dbm": float(np.max(trace)),
                "min_power_dbm": float(np.min(trace)),
                "matched_filter_pslr_db": float(pslr)
            }
            
            # Plot
            plt.figure(figsize=(8, 4))
            plt.plot(trace, "b-", lw=1.2, label=f"RF Spectrum @ {fc}")
            plt.title(f"Anritsu MS2723C Metrological Spectrum - {fc} ({idn})", fontsize=10)
            plt.xlabel("Bin Index")
            plt.ylabel("Amplitude (dBm)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(loc="upper right")
            plt.tight_layout()
            
            fig_path = f"figures/spectrum_{fc.replace('.', '_')}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"       Saved plot to: {fig_path}")

        # Save JSON summary
        out_json = "data/experiments/rf_suite_results/summary_metrics.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n[+] Full experimental suite complete! Summary written to: {out_json}")
        
    finally:
        anritsu.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Experiment Suite")
    parser.add_argument("--anritsu_ip", default="192.168.1.187", help="Anritsu IP")
    args = parser.parse_args()
    
    run_suite(anritsu_ip=args.anritsu_ip)
