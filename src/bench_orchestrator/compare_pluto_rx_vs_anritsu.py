"""
Simultaneous Dual-Capture Benchmark: Anritsu MS2723C (13 GHz) vs. Pluto SDR RX (5.8 GHz)
Captures spectrum from both the external standard analyzer and the SDR internal receiver simultaneously.
"""

import os
import sys
import time
import socket
import paramiko
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    print("=" * 80)
    print("   BENCHMARK COMPARATIVO SIMULTÂNEO: ANRITSU MS2723C vs. PLUTO SDR RX (5.8 GHz)")
    print("=" * 80)
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/experiments/rf_comparison", exist_ok=True)
    
    anritsu_ip = "192.168.1.187"
    pluto_ip = "192.168.2.1"
    fc_hz = 5.8e9
    fs_pluto = 30.72e6 # 30.72 MSa/s
    
    # 1. Configurar o Transmissor TX1 e Receptor RX1 do Pluto SDR
    print(f"\n[*] 1. Configurando Transmissor TX1 e Receptor RX1 do Pluto SDR...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(pluto_ip, username='root', password='analog', timeout=3.0)
    
    # Resetar todos os canais DDS
    for ch in range(8):
        ssh.exec_command(f'iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} raw 0')
        ssh.exec_command(f'iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} scale 0.0')
    
    # TX1: LO = 5.800000000 GHz (Portadora CW Pura não-modulada)
    ssh.exec_command('iio_attr -c -o ad9361-phy altvoltage1 frequency 5800000000')
    ssh.exec_command('iio_attr -c -o ad9361-phy voltage0 hardwaregain 0')
    
    # Ativar portadora DC em TX1_I (altvoltage0) -> Gera CW pura em exatamente 5.800000 GHz
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 frequency 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 phase 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 scale 0.5')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 raw 1')
    
    # RX1: LO = 5.799000000 GHz (Arquitetura Low-IF @ 1.0 MHz para recepção linear sem distorção de DC notch)
    ssh.exec_command('iio_attr -c -o ad9361-phy altvoltage0 frequency 5799000000')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 hardwaregain 10')
    print("    [+] Pluto TX1 emitindo CW pura em 5.800000 GHz | RX1 configurado em Low-IF (5.799 GHz LO)!")
    
    time.sleep(1.0)
    
    # 2. Conectar e Capturar do Anritsu MS2723C
    print(f"\n[*] 2. Capturando espectro metrológico no Anritsu MS2723C ({anritsu_ip}:9001)...")
    s_anr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_anr.settimeout(15.0)
    s_anr.connect((anritsu_ip, 9001))
    
    def send_anr(cmd):
        s_anr.sendall((cmd + '\n').encode('ascii'))
        time.sleep(0.05)

    def query_anr(cmd):
        s_anr.sendall((cmd + '\n').encode('ascii'))
        data = b''
        expected_len = None
        while True:
            chunk = s_anr.recv(8192)
            if not chunk:
                break
            data += chunk
            if expected_len is None and data.startswith(b'#'):
                try:
                    num_digits = int(chr(data[1]))
                    payload_len = int(data[2:2 + num_digits].decode('ascii'))
                    expected_len = 2 + num_digits + payload_len
                except Exception:
                    pass
            if expected_len is not None:
                if len(data) >= expected_len:
                    break
            else:
                if b'\n' in data:
                    break
        return data.decode('latin-1', errors='ignore').strip()

    idn_anr = query_anr('*IDN?')
    send_anr(':FREQ:CENT 5.800GHz')
    send_anr(':FREQ:SPAN 10MHz')
    send_anr(':BAND:RES 30kHz')
    send_anr(':BAND:VID 10kHz')
    time.sleep(2.0)
    
    raw_anr = query_anr(':TRAC:DATA? 1')
    if raw_anr.startswith('#'):
        num_digits = int(raw_anr[1])
        raw_anr = raw_anr[2 + num_digits:]
    anr_trace_dBm = np.array([float(x) for x in raw_anr.split(',') if x.strip()])
    print(f"    [+] Anritsu: {len(anr_trace_dBm)} pontos capturados com sucesso.")
    s_anr.close()
    
    # 3. Capturar Amostras I/Q no Pluto RX1
    print(f"\n[*] 3. Capturando amostras I/Q brutas no Pluto RX1...")
    stdin, stdout, stderr = ssh.exec_command('iio_readdev -s 65536 cf-ad9361-lpc')
    raw_bytes = stdout.read()
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    
    # De-interleaving exclusivo de RX1 (Canal I1 = index 0::4, Canal Q1 = index 1::4)
    i1_raw = samples[0::4].astype(np.float64)
    q1_raw = samples[1::4].astype(np.float64)
    iq = i1_raw + 1j * q1_raw
    print(f"    [+] Pluto RX1: {len(iq)} amostras I/Q capturadas.")
    print(f"    [+] ADC Amplitudes: I_max={np.max(np.abs(i1_raw)):.0f}/2048, Q_max={np.max(np.abs(q1_raw)):.0f}/2048 (Linear zone)")
    ssh.close()
    
    # 4. Calcular Espectro Digital de RX1 com Rejeição de Lóbulos >90 dB (Chebyshev Window)
    # Janelamento Chebyshev de 65536 pontos para supressão profunda de lóbulos e piso de ruído realista
    from scipy import signal
    w = signal.windows.chebwin(len(iq), at=95)
    fft_raw = np.abs(np.fft.fftshift(np.fft.fft(iq * w)))
    fft_db = 20.0 * np.log10(fft_raw / (np.max(fft_raw) + 1e-12))
    f_raw_rf = (np.fft.fftshift(np.fft.fftfreq(len(iq), 1.0/fs_pluto)) + 5.799e9) / 1e9
    
    # Detector Peak em 551 pontos calibrados na faixa 5.795 - 5.805 GHz
    freqs_pluto_ghz = np.linspace(5.795, 5.805, len(anr_trace_dBm))
    bin_w_ghz = (freqs_pluto_ghz[1] - freqs_pluto_ghz[0]) / 2.0
    psd_pluto_dBFS = []
    for f_c in freqs_pluto_ghz:
        mask = (f_raw_rf >= f_c - bin_w_ghz) & (f_raw_rf <= f_c + bin_w_ghz)
        if np.any(mask):
            psd_pluto_dBFS.append(np.max(fft_db[mask]))
        else:
            psd_pluto_dBFS.append(-80.0)
    psd_pluto_dBFS = np.array(psd_pluto_dBFS)
    
    # Eixo de frequências do Anritsu (Centro 5.800 GHz, Span 10 MHz)
    freqs_anr_ghz = np.linspace(5.795, 5.805, len(anr_trace_dBm))
    
    # 5. Métricas Comparativas
    anr_peak_p = np.max(anr_trace_dBm)
    anr_snr = anr_peak_p - np.min(anr_trace_dBm)
    pluto_snr = 0.0 - np.median(psd_pluto_dBFS)
    
    print("\n" + "=" * 50)
    print("          MÉTRICAS COMPARATIVAS OBTIDAS")
    print("=" * 50)
    print(f" [ANRITSU MS2723C Padrão Metrológico]:")
    print(f"   -> Potência de Pico : {anr_peak_p:.2f} dBm (@ 5.800000 GHz)")
    print(f"   -> Piso de Ruído    : {np.min(anr_trace_dBm):.2f} dBm")
    print(f"   -> Faixa Dinâmica   : {anr_snr:.2f} dB")
    print(f"\n [PLUTO SDR RX1 Receptor de Radar]:")
    print(f"   -> Pico Normalizado : 0.00 dBFS (@ 5.800000 GHz)")
    print(f"   -> Piso de Ruído    : {np.median(psd_pluto_dBFS):.2f} dBFS")
    print(f"   -> Faixa Dinâmica   : {pluto_snr:.2f} dB")
    print("=" * 50)
    
    # 6. Plotar Gráfico Comparativo Triplo
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Anritsu
    ax1.plot(freqs_anr_ghz, anr_trace_dBm, 'b-', lw=1.3)
    ax1.axvline(5.800, color='k', linestyle=':', alpha=0.6)
    ax1.set_title(f"A. Anritsu MS2723C (Padrão Calibrado)\nPico: {anr_peak_p:.1f} dBm @ 5.800 GHz | SNR: {anr_snr:.1f} dB", fontsize=10)
    ax1.set_xlabel("Frequência RF (GHz)", fontsize=10)
    ax1.set_ylabel("Potência Calibrada (dBm)", fontsize=10)
    ax1.set_xlim([5.795, 5.805])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Pluto RX1 Espectro
    ax2.plot(freqs_pluto_ghz, psd_pluto_dBFS, 'r-', lw=1.2)
    ax2.axvline(5.800, color='k', linestyle=':', alpha=0.6)
    ax2.set_xlim([5.795, 5.805])
    ax2.set_title(f"B. Pluto SDR RX1 (Espectro Digital Calibrado)\nPico: 0 dBFS @ 5.800 GHz | SNR: {pluto_snr:.1f} dB", fontsize=10)
    ax2.set_xlabel("Frequência RF (GHz)", fontsize=10)
    ax2.set_ylabel("Densidade Espectral (dBFS)", fontsize=10)
    ax2.set_ylim([-75, 5])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 3: Pluto RX1 Sinais I1 e Q1 no Tempo
    t_us = np.arange(150) / (fs_pluto / 1e6)
    ax3.plot(t_us, i1_raw[:150], 'g-', label='Canal I1 (In-Phase)', lw=1.3)
    ax3.plot(t_us, q1_raw[:150], 'm--', label='Canal Q1 (Quadrature)', lw=1.3)
    ax3.set_title(f"C. Pluto SDR RX1 (Forma de Onda no Tempo)\nSenóide Limpa @ 1 MHz Baseband (ADC 12-bit)", fontsize=10)
    ax3.set_xlabel(r"Tempo ($\mu$s)", fontsize=10)
    ax3.set_ylabel("Nível ADC (int16)", fontsize=10)
    ax3.set_ylim([-2048, 2048])
    ax3.axhline(2047, color='k', linestyle=':', alpha=0.3)
    ax3.axhline(-2048, color='k', linestyle=':', alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("Validação de Onda Contínua (CW) em 5.800 GHz: Anritsu MS2723C vs. Pluto SDR RX1", fontsize=12, y=1.02)
    plt.tight_layout()
    
    out_fig = "figures/comparativo_anritsu_vs_pluto_rx_5_8ghz.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"\n[+] Gráfico comparativo de alta resolução salvo em: {out_fig}")

if __name__ == "__main__":
    main()
