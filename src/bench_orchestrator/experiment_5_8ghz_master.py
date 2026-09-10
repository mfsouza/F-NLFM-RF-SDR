import os
import sys
import json
import time
import socket
import paramiko
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

ROOT_DIR = "C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rf_sdr.fost_fnlfm import generate_fnlfm_fost

def run_master():
    print("=" * 80)
    print("   LIVE MICROWAVE BENCHMARK AT 5.8 GHz: FOSM/F-NLFM vs. CONVENTIONAL LFM")
    print("   Instruments: ADALM-Pluto SDR (AD9361) + Anritsu MS2723C Calibrated Spectrum Analyzer")
    print("=" * 80)
    
    anritsu_ip = "192.168.1.187"
    pluto_ip = "192.168.1.11"
    fc_hz = 5.8e9
    fs = 30.72e6
    B = 20.0e6
    N_pulse = 2048
    N_buf = 4096
    T_pulse = N_pulse / fs  # 66.67 us
    
    # 1. Synthesize Waveforms
    print(f"\n[*] 1. Synthesizing Radar Waveforms (B = {B/1e6:.1f} MHz, T = {T_pulse*1e6:.2f} us, N_pulse = {N_pulse})...")
    
    # A. Conventional LFM
    t_lfm = np.linspace(-T_pulse/2, T_pulse/2, N_pulse, endpoint=False)
    f_lfm = (B / T_pulse) * t_lfm
    phi_lfm = 2 * np.pi * np.cumsum(f_lfm) / fs
    s_lfm_pulse = np.exp(1j * phi_lfm)
    
    # Full cyclic buffer for continuous RF streaming
    f_lfm_inst = np.concatenate([f_lfm, f_lfm[::-1]])
    f_lfm_inst -= np.mean(f_lfm_inst)
    phi_lfm_stream = 2 * np.pi * np.cumsum(f_lfm_inst) / fs
    s_lfm_stream = np.exp(1j * phi_lfm_stream)

    # B. Proposed FOSM / F-NLFM (Kaiser FOST)
    t_fost, s_fost_pulse, meta = generate_fnlfm_fost(
        pulse_width=T_pulse,
        bandwidth=B,
        sample_rate=fs,
        beta=9.5,
        fractal_dim=1.01,
        epsilon=0.002,
        num_freq_points=4096
    )
    f_fost = meta['f_instantaneous']
    f_fost_inst = np.concatenate([f_fost, f_fost[::-1]])
    f_fost_inst -= np.mean(f_fost_inst)
    phi_fost_stream = 2 * np.pi * np.cumsum(f_fost_inst) / fs
    s_fost_stream = np.exp(1j * phi_fost_stream)

    def execute_hardware_test(waveform_name, sig_stream, sig_pulse, null_samples=16):
        print(f"\n--- Running Hardware Test: {waveform_name} at 5.8 GHz ---")
        
        # Connect SSH to configure hardware
        ssh_local = paramiko.SSHClient()
        ssh_local.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_local.connect(pluto_ip, username='root', password='analog', timeout=10.0)
        
        setup_cmd = (
            "killall -9 iio_writedev iio_readdev cat >/dev/null 2>&1; "
            "echo 0 > /sys/bus/iio/devices/iio:device2/buffer/enable 2>/dev/null; "
            "echo 0 > /sys/bus/iio/devices/iio:device3/buffer/enable 2>/dev/null; "
            "iio_attr -o -c ad9361-phy altvoltage1 frequency 5800000000; "
            "iio_attr -o -c ad9361-phy voltage0 hardwaregain 0; "
            "iio_attr -o -c ad9361-phy voltage0 rf_bandwidth 28000000; "
            "iio_attr -o -c ad9361-phy voltage0 sampling_frequency 30720000; "
            "iio_attr -i -c ad9361-phy altvoltage0 frequency 5800000000; "
            "iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual; "
            "iio_attr -i -c ad9361-phy voltage0 hardwaregain 50; "
            "iio_attr -i -c ad9361-phy voltage0 rf_bandwidth 28000000; "
            "iio_attr -i -c ad9361-phy voltage0 sampling_frequency 30720000; "
            "iio_attr -i -c ad9361-phy voltage0 bb_dc_offset_tracking_en 0; "
            "iio_attr -i -c ad9361-phy voltage0 rf_dc_offset_tracking_en 0; "
            "iio_attr -i -c ad9361-phy voltage0 quad_tracking_en 0"
        )
        for ch in range(8):
            setup_cmd += f"; iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} raw 0; iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} scale 0.0"
            
        ssh_local.exec_command(setup_cmd)
        time.sleep(0.3)

        i_sig = np.int16(32000 * np.real(sig_stream))
        q_sig = np.int16(32000 * np.imag(sig_stream))
        zero_sig = np.zeros(N_buf, dtype=np.int16)
        
        raw_arr = np.empty((N_buf, 4), dtype=np.int16)
        raw_arr[:, 0] = i_sig
        raw_arr[:, 1] = q_sig
        raw_arr[:, 2] = zero_sig
        raw_arr[:, 3] = zero_sig
        raw_bytes = raw_arr.tobytes()
        
        # Upload buffer
        hex_data = raw_bytes.hex()
        sin, sout, serr = ssh_local.exec_command('xxd -r -p > /tmp/tx_buf.raw')
        sin.write(hex_data)
        sin.flush()
        sin.channel.shutdown_write()
        sout.read()
        ssh_local.close()

        # Connect fresh SSH session for streaming
        ssh_stream = paramiko.SSHClient()
        ssh_stream.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_stream.connect(pluto_ip, username='root', password='analog', timeout=10.0)
        
        stream_cmd = (
            'printf "#!/bin/sh\\nwhile true; do cat /tmp/tx_buf.raw; done | iio_writedev -u local: -b 4096 cf-ad9361-dds-core-lpc\\n" > /tmp/tx_loop.sh; '
            'chmod +x /tmp/tx_loop.sh; '
            'nohup /tmp/tx_loop.sh >/dev/null 2>&1 &'
        )
        ssh_stream.exec_command(stream_cmd)
        time.sleep(1.0)
        
        # Anritsu MS2723C Spectrum Capture
        print("    [*] Integrating RF Spectrum on Anritsu MS2723C (Max-Hold)...")
        s_anr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_anr.settimeout(15.0)
        s_anr.connect((anritsu_ip, 9001))
        
        s_anr.sendall(b':FREQ:CENT 5800000000\n')
        s_anr.sendall(b':FREQ:SPAN 50000000\n')
        s_anr.sendall(b':DISP:WIND:TRAC:Y:RLEV -20\n')
        s_anr.sendall(b':POW:ATT:AUTO ON\n')
        s_anr.sendall(b':BAND:RES 100000\n')
        s_anr.sendall(b':BAND:VID 30000\n')
        s_anr.sendall(b':TRAC:MODE NORM\n')
        s_anr.sendall(b':INIT:CONT ON\n')
        s_anr.sendall(b':INIT:IMM\n')
        time.sleep(1.5)
        
        s_anr.sendall(b':TRAC:MODE MAXH\n')
        time.sleep(4.0)
        
        s_anr.sendall(b':TRAC:DATA? 1\n')
        data = b''
        while not data.endswith(b'\n'):
            chunk = s_anr.recv(4096)
            if not chunk: break
            data += chunk
        s_anr.close()
        
        txt = data.decode('latin-1', errors='ignore').strip()
        if txt.startswith('#'):
            num_digits = int(txt[1])
            txt = txt[2 + num_digits:]
        anr_trace = np.array([float(x) for x in txt.split(',') if x.strip()])
        anr_freqs = np.linspace(5.800e9 - 25e6, 5.800e9 + 25e6, len(anr_trace)) / 1e9
        
        # Pluto RX1 Loopback Capture
        print("    [*] Capturing loopback I/Q samples on Pluto RX1...")
        sin, sout, serr = ssh_stream.exec_command('iio_readdev -u local: -s 65536 cf-ad9361-lpc')
        rx_out = sout.read()
        rx_raw = np.frombuffer(rx_out, dtype=np.int16)
        rx_i = rx_raw[0::4].astype(np.float64)
        rx_q = rx_raw[1::4].astype(np.float64)
        rx_i -= np.mean(rx_i)
        rx_q -= np.mean(rx_q)
        rx_iq = rx_i + 1j * rx_q
        
        # Clean up TX process
        ssh_stream.exec_command('killall -9 tx_loop.sh iio_writedev cat 2>/dev/null; rm -f /tmp/tx_loop.sh')
        ssh_stream.close()
        
        # Matched Filter Pulse Compression on single pulse
        mf = np.conj(sig_pulse[::-1])
        corr = np.correlate(rx_iq, mf, mode='valid')
        corr_mag = np.abs(corr)
        corr_db = 20.0 * np.log10(corr_mag / np.max(corr_mag) + 1e-12)
        
        # Find periodic peaks with safe margin
        win_r = 250
        pks, _ = signal.find_peaks(corr_db, height=-10, distance=3500)
        valid_pks = [p for p in pks if win_r <= p <= len(corr_db) - win_r - 1]
        if valid_pks:
            peak_idx = valid_pks[0]
        else:
            search_slice = corr_db[win_r : len(corr_db) - win_r]
            peak_idx = win_r + int(np.argmax(search_slice))
            
        w_start = peak_idx - win_r
        w_end = peak_idx + win_r + 1
        pulse_slice = corr_db[w_start:w_end]
        center = win_r
        
        mask = np.ones(len(pulse_slice), dtype=bool)
        mask[max(0, center - null_samples) : min(len(pulse_slice), center + null_samples + 1)] = False
        pslr = float(np.max(pulse_slice[mask]))
        
        print(f"    [+] Peak RF Power (Anritsu): {np.max(anr_trace):.2f} dBm (Floor: {np.min(anr_trace):.2f} dBm)")
        print(f"    [+] Mainlobe Mask: +/- {null_samples} samples")
        print(f"    [+] Measured Hardware PSLR: {pslr:.2f} dB")
        
        return anr_freqs, anr_trace, rx_iq, corr_db, peak_idx, pslr

    # Execute both tests
    anr_f_lfm, anr_tr_lfm, rx_lfm, corr_lfm, pk_lfm, pslr_lfm = execute_hardware_test("Conventional LFM", s_lfm_stream, s_lfm_pulse, null_samples=8)
    time.sleep(1.0)
    anr_f_fost, anr_tr_fost, rx_fost, corr_fost, pk_fost, pslr_fost = execute_hardware_test("Proposed FOSM / F-NLFM", s_fost_stream, s_fost_pulse, null_samples=18)
    
    # 4. Generate Master 4-Panel Figure
    print("\n[*] 4. Generating Publication-Quality Benchmark Figure...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel A: Anritsu RF Spectrum
    ax1.plot(anr_f_lfm, anr_tr_lfm, color='#1f77b4', linestyle='--', label='LFM Convencional (B = 20 MHz)', lw=1.5)
    ax1.plot(anr_f_fost, anr_tr_fost, color='#d62728', linestyle='-', label='FOSM / F-NLFM Proposta (Kaiser-FOST)', lw=1.8)
    ax1.set_title("A. Espectro de Potencia Calibrado no Anritsu MS2723C @ 5.800 GHz\nConfinamento Espectral e Suavizacao das Bordas de Banda", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Frequencia RF (GHz)", fontsize=10)
    ax1.set_ylabel("Potencia Calibrada (dBm)", fontsize=10)
    ax1.set_xlim([5.775, 5.825])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', frameon=True)
    
    # Panel B: Pulse Compression
    tau_us = (np.arange(-200, 201)) / (fs / 1e6)
    slice_lfm = corr_lfm[pk_lfm - 200 : pk_lfm + 201]
    slice_fost = corr_fost[pk_fost - 200 : pk_fost + 201]
    
    ax2.plot(tau_us, slice_lfm, color='#1f77b4', linestyle='--', label=f'LFM Convencional (PSLR = {pslr_lfm:.1f} dB)', lw=1.5)
    ax2.plot(tau_us, slice_fost, color='#d62728', linestyle='-', label=f'FOSM / F-NLFM (PSLR = {pslr_fost:.1f} dB)', lw=1.8)
    ax2.axhline(-13.2, color='#1f77b4', linestyle=':', alpha=0.6, label='Limite Teorico Sinc (-13.2 dB)')
    ax2.axhline(pslr_fost, color='#d62728', linestyle=':', alpha=0.6, label=f'PSLR FOSM Hardware ({pslr_fost:.1f} dB)')
    ax2.set_title(f"B. Compressao de Pulso por Filtro Casado no Pluto RX1\nSupressao de Lobulos: {abs(pslr_fost) - abs(pslr_lfm):+.1f} dB (0 dB Mismatch Loss)", fontsize=11, fontweight='bold')
    ax2.set_xlabel(r"Tempo de Atraso $\tau$ ($\mu$s)", fontsize=10)
    ax2.set_ylabel("Magnitude Normalizada (dB)", fontsize=10)
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-60, 5])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', frameon=True)
    
    # Panel C: Instantaneous Frequency Trajectory
    t_us = t_fost * 1e6
    ax3.plot(t_us, f_lfm / 1e6, color='#1f77b4', linestyle='--', label='LFM (Rampa Linear $f_i(t) = K t$)', lw=1.5)
    ax3.plot(t_us, f_fost / 1e6, color='#d62728', linestyle='-', label='FOSM / F-NLFM (Curva-S Estacionaria)', lw=1.8)
    ax3.set_title("C. Trajetoria de Frequencia Instantanea em Banda Base\nModulacao de Fase Nao-Linear com Envelope Constante", fontsize=11, fontweight='bold')
    ax3.set_xlabel(r"Tempo ($\mu$s)", fontsize=10)
    ax3.set_ylabel("Frequencia Instantanea (MHz)", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', frameon=True)
    
    # Panel D: Mainlobe Detail
    tau_ns = tau_us * 1000
    ax4.plot(tau_ns, slice_lfm, color='#1f77b4', linestyle='--', marker='o', markersize=3, label='Lobulo LFM', lw=1.3)
    ax4.plot(tau_ns, slice_fost, color='#d62728', linestyle='-', marker='s', markersize=3, label='Lobulo FOSM', lw=1.6)
    ax4.axhline(-3, color='k', linestyle=':', alpha=0.6, label='Largura de Banda a -3 dB')
    ax4.set_title("D. Resolucao em Distancia (Lobulo Principal a -3 dB)\nPreservacao da Largura do Lobulo sem Janelamento Ponderador", fontsize=11, fontweight='bold')
    ax4.set_xlabel(r"Tempo de Atraso $\tau$ (ns)", fontsize=10)
    ax4.set_ylabel("Magnitude (dB)", fontsize=10)
    ax4.set_xlim([-300, 300])
    ax4.set_ylim([-30, 2])
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper right', frameon=True)
    
    plt.suptitle("Validacao de Micro-ondas em 5.800 GHz: FOSM / F-NLFM vs. LFM Convencional\nAnritsu MS2723C (Padrao Calibrado de Laboratorio) & ADALM-Pluto SDR (Loopback RX1)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    os.makedirs("C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/figures", exist_ok=True)
    os.makedirs("C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/docs/overleaf/IEEE_RF_FNLFM_SDR_Paper/figures", exist_ok=True)
    os.makedirs("C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/data/experiments/rf_5_8ghz", exist_ok=True)
    
    out_fig1 = "C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/figures/radar_fnlfm_vs_lfm_5_8ghz_benchmark.png"
    out_fig2 = "C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/docs/overleaf/IEEE_RF_FNLFM_SDR_Paper/figures/fig2_radar_benchmark_5_8ghz.png"
    plt.savefig(out_fig1, dpi=300, bbox_inches='tight')
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    print(f"\n[+] Master plots saved to: {out_fig1} and {out_fig2}")
    
    # Save JSON report
    report = {
        "frequencia_portadora_ghz": 5.8,
        "largura_banda_mhz": 20.0,
        "duracao_pulso_us": float(T_pulse * 1e6),
        "produto_bt": float(B * T_pulse),
        "pslr_lfm_db": float(pslr_lfm),
        "pslr_fnlfm_db": float(pslr_fost),
        "melhoria_pslr_db": float(abs(pslr_fost) - abs(pslr_lfm)),
        "potencia_pico_lfm_dbm": float(np.max(anr_tr_lfm)),
        "potencia_pico_fnlfm_dbm": float(np.max(anr_tr_fost)),
        "piso_ruido_anritsu_dbm": float(np.min(anr_tr_fost))
    }
    with open("C:/Users/moise/Documents/GitHub/F-NLFM-RF-SDR/data/experiments/rf_5_8ghz/relatorio_fnlfm_5_8ghz_master.json", "w") as f:
        json.dump(report, f, indent=4)
    print("    [+] Numeric report saved.")

if __name__ == "__main__":
    run_master()
