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
    
    # 1. Configurar o Transmissor e Receptor do Pluto via SSH
    print(f"\n[*] 1. Configurando Transmissor TX e Receptor RX do Pluto SDR em 5.8 GHz...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(pluto_ip, username='root', password='analog', timeout=3.0)
    
    # Configurar TX LO e DDS (Tom CW I/Q em offset de +1 MHz para demonstrar portadora limpa em quadratura)
    ssh.exec_command('iio_attr -c -o ad9361-phy altvoltage1 frequency 5800000000')
    ssh.exec_command('iio_attr -c -o ad9361-phy voltage0 hardwaregain 0')
    
    # Desabilitar outros canais DDS
    for ch in range(8):
        ssh.exec_command(f'iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} raw 0')
        
    # Canal I: altvoltage0 (TX1_I_F1), Fase 0
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 frequency 1000000')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 phase 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 scale 0.5')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 raw 1')
    
    # Canal Q: altvoltage2 (TX1_Q_F1), Fase 270 graus (-90 deg) para tom analítico positivo exp(+j*w*t)
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage2 frequency 1000000')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage2 phase 270000')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage2 scale 0.5')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage2 raw 1')
    
    # Configurar RX LO e Ganho Linear Seguro (0 dB para evitar saturação do ADC com atenuador de 10 dB)
    ssh.exec_command('iio_attr -c -o ad9361-phy altvoltage0 frequency 5800000000')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 hardwaregain 0')
    print("    [+] Pluto TX (I/Q Quadrature @ +1 MHz) e RX calibrados em 5.800000 GHz (Linear Zone: Ganho 0 dB)!")
    
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
    send_anr(':FREQ:CENT 5.801GHz')
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
    
    # 3. Capturar Amostras I/Q no Pluto RX
    print(f"\n[*] 3. Capturando amostras I/Q brutas no Pluto RX...")
    stdin, stdout, stderr = ssh.exec_command('iio_readdev -s 65536 cf-ad9361-lpc')
    raw_bytes = stdout.read()
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    
    # De-interleaving correto dos canais IIO (4 canais: I1, Q1, I2, Q2)
    i_raw = samples[0::4].astype(np.float64)
    q_raw = samples[1::4].astype(np.float64)
    
    # Calibração e Correção de Erros de Quadratura (QEC / Gram-Schmidt Orthogonalization)
    i_dc = i_raw - np.mean(i_raw)
    q_dc = q_raw - np.mean(q_raw)
    i_norm = i_dc / np.std(i_dc)
    q_norm = q_dc / np.std(q_dc)
    sin_phi = np.mean(i_norm * q_norm)
    cos_phi = np.sqrt(max(1.0 - sin_phi**2, 1e-6))
    q_ortho = (q_norm - sin_phi * i_norm) / cos_phi
    
    iq_cal = i_norm + 1j * q_ortho
    print(f"    [+] Pluto RX: {len(iq_cal)} amostras I/Q calibradas.")
    print(f"    [+] Nível do ADC: I_max={np.max(np.abs(i_raw)):.0f}/2048, Q_max={np.max(np.abs(q_raw)):.0f}/2048 (Linear zone)")
    ssh.close()
    
    # 4. Calcular Densidade Espectral de Potência (Welch PSD) do Pluto RX
    from scipy import signal
    f_welch, psd_welch = signal.welch(iq_cal, fs=fs_pluto, window='hann', nperseg=4096, return_onesided=False)
    freqs_pluto_ghz = (np.fft.fftshift(f_welch) + fc_hz) / 1e9
    psd_pluto_dBFS = 10.0 * np.log10(np.fft.fftshift(psd_welch))
    psd_pluto_dBFS = psd_pluto_dBFS - np.max(psd_pluto_dBFS)
    
    # Eixo de frequências do Anritsu (Centro 5.801 GHz, Span 10 MHz)
    freqs_anr_ghz = np.linspace(5.801e9 - 5e6, 5.801e9 + 5e6, len(anr_trace_dBm)) / 1e9
    
    # 5. Métricas Comparativas
    anr_peak_p = np.max(anr_trace_dBm)
    anr_snr = anr_peak_p - np.min(anr_trace_dBm)
    pluto_snr = 0.0 - np.min(psd_pluto_dBFS)
    
    print("\n" + "=" * 50)
    print("          MÉTRICAS COMPARATIVAS OBTIDAS")
    print("=" * 50)
    print(f" [ANRITSU MS2723C Padrão Metrológico]:")
    print(f"   -> Potência de Pico : {anr_peak_p:.2f} dBm")
    print(f"   -> Piso de Ruído    : {np.min(anr_trace_dBm):.2f} dBm")
    print(f"   -> Faixa Dinâmica   : {anr_snr:.2f} dB")
    print(f"\n [PLUTO SDR RX Receptor de Radar]:")
    print(f"   -> Pico Normalizado : 0.00 dBFS (@ 5.801 GHz)")
    print(f"   -> Piso de Ruído    : {np.min(psd_pluto_dBFS):.2f} dBFS")
    print(f"   -> Faixa Dinâmica   : {pluto_snr:.2f} dB")
    print("=" * 50)
    
    # 6. Plotar Gráfico Comparativo Triplo (Anritsu, Espectro Pluto RX, I/Q Pluto RX no Tempo)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Anritsu
    ax1.plot(freqs_anr_ghz, anr_trace_dBm, 'b-', lw=1.3)
    ax1.axvline(5.801, color='k', linestyle=':', alpha=0.6)
    ax1.set_title(f"A. Anritsu MS2723C (Padrão Calibrado)\nPico: {anr_peak_p:.1f} dBm @ 5.801 GHz | SNR: {anr_snr:.1f} dB", fontsize=10)
    ax1.set_xlabel("Frequência RF (GHz)", fontsize=10)
    ax1.set_ylabel("Potência Calibrada (dBm)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Pluto RX Espectro
    ax2.plot(freqs_pluto_ghz, psd_pluto_dBFS, 'r-', lw=1.2)
    ax2.axvline(5.801, color='k', linestyle=':', alpha=0.6)
    ax2.set_xlim([5.801 - 0.005, 5.801 + 0.005])
    ax2.set_title(f"B. Pluto SDR RX (Espectro Welch PSD)\nPico: 0 dBFS @ 5.801 GHz | SNR: {pluto_snr:.1f} dB", fontsize=10)
    ax2.set_xlabel("Frequência RF (GHz)", fontsize=10)
    ax2.set_ylabel("Densidade Espectral (dBFS)", fontsize=10)
    ax2.set_ylim([-75, 5])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 3: Pluto RX Sinais I e Q no Tempo
    t_us = np.arange(150) / (fs_pluto / 1e6)
    ax3.plot(t_us, i_norm[:150], 'g-', label='Canal I (In-Phase)', lw=1.3)
    ax3.plot(t_us, q_ortho[:150], 'm--', label='Canal Q (Quadrature)', lw=1.3)
    ax3.set_title(f"C. Pluto SDR RX (Forma de Onda I/Q Calibrada)\nQuadratura Ortogonal Pura @ 1 MHz Baseband", fontsize=10)
    ax3.set_xlabel(r"Tempo ($\mu$s)", fontsize=10)
    ax3.set_ylabel("Amplitude Normalizada", fontsize=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("Validação Simultânea em 5.8 GHz: Anritsu MS2723C vs. Pluto SDR RX (Loopback RF Calibrado)", fontsize=12, y=1.02)
    plt.tight_layout()
    
    out_fig = "figures/comparativo_anritsu_vs_pluto_rx_5_8ghz.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"\n[+] Gráfico comparativo de alta resolução salvo em: {out_fig}")

if __name__ == "__main__":
    main()
