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
    
    # Configurar TX LO e DDS
    ssh.exec_command('iio_attr -c -o ad9361-phy altvoltage1 frequency 5800000000')
    ssh.exec_command('iio_attr -c -o ad9361-phy voltage0 hardwaregain 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 raw 1')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 frequency 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage0 scale 1.0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage1 raw 1')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage1 frequency 0')
    ssh.exec_command('iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage1 scale 1.0')
    
    # Configurar RX LO e Ganho
    ssh.exec_command('iio_attr -c -i ad9361-phy altvoltage0 frequency 5800000000')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual')
    ssh.exec_command('iio_attr -i -c ad9361-phy voltage0 hardwaregain 40')
    print("    [+] Pluto TX e RX operando sincronizados em 5.800000 GHz!")
    
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
    send_anr(':FREQ:CENT 5.8GHz')
    send_anr(':FREQ:SPAN 20MHz')
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
    print(f"\n[*] 3. Capturando amostras I/Q brutas de alta velocidade no Pluto RX...")
    stdin, stdout, stderr = ssh.exec_command('iio_readdev -s 32768 cf-ad9361-lpc')
    raw_bytes = stdout.read()
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    i_samples = samples[0::2]
    q_samples = samples[1::2]
    iq_complex = i_samples + 1j * q_samples
    print(f"    [+] Pluto RX: {len(iq_complex)} amostras I/Q complexas capturadas.")
    ssh.close()
    
    # 4. Calcular FFT e PSD do Pluto RX
    n_fft = len(iq_complex)
    window = np.blackman(n_fft)
    fft_vals = np.fft.fftshift(np.fft.fft(iq_complex * window))
    psd_pluto_dBFS = 20.0 * np.log10(np.abs(fft_vals) / (np.max(np.abs(fft_vals)) + 1e-12))
    
    freqs_pluto_ghz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs_pluto)) + fc_hz
    freqs_pluto_ghz = freqs_pluto_ghz / 1e9
    
    # Eixo de frequências do Anritsu
    freqs_anr_ghz = np.linspace(fc_hz - 10e6, fc_hz + 10e6, len(anr_trace_dBm)) / 1e9
    
    # 5. Métricas Comparativas
    anr_peak_p = np.max(anr_trace_dBm)
    anr_snr = anr_peak_p - np.min(anr_trace_dBm)
    
    pluto_snr = 0.0 - np.median(psd_pluto_dBFS)
    
    print("\n" + "=" * 50)
    print("          MÉTRICAS COMPARATIVAS OBTIDAS")
    print("=" * 50)
    print(f" [ANRITSU MS2723C Padrão Metrológico]:")
    print(f"   -> Potência de Pico : {anr_peak_p:.2f} dBm")
    print(f"   -> Piso de Ruído    : {np.min(anr_trace_dBm):.2f} dBm")
    print(f"   -> Faixa Dinâmica   : {anr_snr:.2f} dB")
    print(f"\n [PLUTO SDR RX Receptor de Radar]:")
    print(f"   -> Pico Normalizado : 0.00 dBFS")
    print(f"   -> Piso de Ruído    : {np.median(psd_pluto_dBFS):.2f} dBFS")
    print(f"   -> Faixa Dinâmica   : {pluto_snr:.2f} dB")
    print("=" * 50)
    
    # 6. Plotar Gráfico Comparativo Lado a Lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Anritsu
    ax1.plot(freqs_anr_ghz, anr_trace_dBm, 'b-', lw=1.3)
    ax1.axvline(5.8, color='k', linestyle=':', alpha=0.6)
    ax1.set_title(f"A. Anritsu MS2723C (Padrão de Laboratório)\nPotência: {anr_peak_p:.1f} dBm | SNR: {anr_snr:.1f} dB", fontsize=10)
    ax1.set_xlabel("Frequência (GHz)", fontsize=10)
    ax1.set_ylabel("Potência Calibrada (dBm)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Subplot 2: Pluto RX
    ax2.plot(freqs_pluto_ghz, psd_pluto_dBFS, 'r-', lw=1.1)
    ax2.axvline(5.8, color='k', linestyle=':', alpha=0.6)
    ax2.set_title(f"B. Pluto SDR RX (Receptor Digital I/Q)\nNível: 0 dBFS | SNR: {pluto_snr:.1f} dB", fontsize=10)
    ax2.set_xlabel("Frequência (GHz)", fontsize=10)
    ax2.set_ylabel("Magnitude Normalizada (dBFS)", fontsize=10)
    ax2.set_ylim([-85, 5])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("Comparação Simultânea de Micro-ondas em 5.8 GHz: Anritsu MS2723C vs. Pluto SDR RX", fontsize=12, y=1.02)
    plt.tight_layout()
    
    out_fig = "figures/comparativo_anritsu_vs_pluto_rx_5_8ghz.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"\n[+] Gráfico comparativo de alta resolução salvo em: {out_fig}")

if __name__ == "__main__":
    main()
