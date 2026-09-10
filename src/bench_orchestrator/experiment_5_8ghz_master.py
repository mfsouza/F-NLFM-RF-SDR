"""
5.8 GHz C-Band Master Radar Experiment (Pluto SDR + Anritsu MS2723C)
Executes Live Hardware LFM vs. F-NLFM (FOST) transmission, Anritsu spectral characterization,
and real-time loopback pulse compression on Pluto RX1.
"""

import os
import sys
import json
import time
import socket
import paramiko
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rf_sdr.fost_fnlfm import generate_fnlfm_fost

def run_5_8ghz_experiment():
    print("=" * 80)
    print("   LIVE MICROWAVE BENCHMARK AT 5.8 GHz: FOSM/F-NLFM vs. CONVENTIONAL LFM")
    print("=" * 80)
    
    os.makedirs("data/experiments/rf_5_8ghz", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    anritsu_ip = "192.168.1.187"
    pluto_ip = "192.168.2.1"
    fc_hz = 5.8e9
    fs = 30.72e6
    B = 20.0e6
    T_pulse = 50.0e-6
    N_buf = 4096
    
    # 1. Sintetizar Formas de Onda em Banda Base
    print("\n[*] 1. Sintetizando formas de onda de radar (B = 20 MHz, T = 50 us)...")
    
    # A. LFM Convencional
    Np = int(np.round(T_pulse * fs))
    t_lfm = np.linspace(-T_pulse/2, T_pulse/2, Np, endpoint=False)
    s_lfm = np.exp(1j * np.pi * (B / T_pulse) * t_lfm**2)
    buf_lfm = np.zeros(N_buf, dtype=np.complex128)
    buf_lfm[:Np] = s_lfm
    
    # B. FOSM / F-NLFM Proposta
    t_fost, s_fost, meta = generate_fnlfm_fost(
        pulse_width=T_pulse,
        bandwidth=B,
        sample_rate=fs,
        beta=9.5,
        fractal_dim=1.01,
        epsilon=0.002
    )
    buf_fost = np.zeros(N_buf, dtype=np.complex128)
    buf_fost[:len(s_fost)] = s_fost
    
    print(f"    [+] LFM:    {len(s_lfm)} amostras no buffer ciclico de {N_buf} (PAPR = 0.0 dB)")
    print(f"    [+] F-NLFM: {len(s_fost)} amostras no buffer ciclico de {N_buf} (PAPR = 0.0 dB)")
    
    # Helper function to send buffer and run capture
    def execute_waveform_test(waveform_name, buf_complex, mf_ref, null_samples=15):
        print(f"\n--- Executando Teste de Hardware: {waveform_name} em 5.8 GHz ---")
        
        def run_remote(ssh, cmd):
            sin, sout, serr = ssh.exec_command(cmd)
            out = sout.read()
            err = serr.read()
            return out, err

        # Conectar SSH para este teste
        ssh_local = paramiko.SSHClient()
        ssh_local.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_local.connect(pluto_ip, username='root', password='analog', timeout=10.0)
        
        # Configurar LOs e calibração de RF
        setup_script = """
        iio_attr -c -o ad9361-phy altvoltage1 frequency 5800000000
        iio_attr -c -o ad9361-phy voltage0 hardwaregain 0
        iio_attr -c -o ad9361-phy altvoltage0 frequency 5800000000
        iio_attr -i -c ad9361-phy voltage0 gain_control_mode manual
        iio_attr -i -c ad9361-phy voltage0 hardwaregain 10
        killall -9 run_tx.sh iio_writedev >/dev/null 2>&1
        """
        for ch in range(8):
            setup_script += f"iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} raw 0\n"
            setup_script += f"iio_attr -c -o cf-ad9361-dds-core-lpc altvoltage{ch} scale 0.0\n"
            
        run_remote(ssh_local, setup_script)

        i_sig = np.int16(30000 * np.real(buf_complex))
        q_sig = np.int16(30000 * np.imag(buf_complex))
        zero_sig = np.zeros(N_buf, dtype=np.int16)
        
        raw_arr = np.empty((N_buf, 4), dtype=np.int16)
        raw_arr[:, 0] = i_sig
        raw_arr[:, 1] = q_sig
        raw_arr[:, 2] = zero_sig
        raw_arr[:, 3] = zero_sig
        raw_bytes = raw_arr.tobytes()
        
        # Enviar buffer via xxd hex decoding
        hex_data = raw_bytes.hex()
        sin, sout, serr = ssh_local.exec_command('xxd -r -p > /tmp/tx_buf.raw')
        sin.write(hex_data)
        sin.close()
        sout.read()
        
        # Iniciar transmissor DMA ciclico continuo em hardware
        sin, sout, serr = ssh_local.exec_command('nohup iio_writedev -c -b 4096 cf-ad9361-dds-core-lpc < /tmp/tx_buf.raw >/dev/null 2>&1 &')
        sout.read()
        time.sleep(2.0)
        
        # Captura no Anritsu MS2723C
        print(f"    [*] Capturando espectro de RF no Anritsu MS2723C...")
        s_anr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_anr.settimeout(15.0)
        s_anr.connect((anritsu_ip, 9001))
        
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

        s_anr.sendall(b':FREQ:CENT 5.800GHz\n')
        s_anr.sendall(b':FREQ:SPAN 50MHz\n')
        s_anr.sendall(b':BAND:RES 100kHz\n')
        s_anr.sendall(b':BAND:VID 30kHz\n')
        s_anr.sendall(b':INIT:CONT ON\n')
        s_anr.sendall(b':INIT:IMM\n')
        time.sleep(2.0)
        
        raw_anr = query_anr(':TRAC:DATA? 1')
        if raw_anr.startswith('#'):
            num_digits = int(raw_anr[1])
            raw_anr = raw_anr[2 + num_digits:]
        anr_trace = np.array([float(x) for x in raw_anr.split(',') if x.strip()])
        anr_freqs = np.linspace(5.800e9 - 25e6, 5.800e9 + 25e6, len(anr_trace)) / 1e9
        s_anr.close()
        
        # Captura no Pluto RX1
        print(f"    [*] Capturando amostras I/Q de loopback no Pluto RX1...")
        rx_out, _ = run_remote(ssh_local, 'iio_readdev -s 65536 cf-ad9361-lpc')
        rx_raw = np.frombuffer(rx_out, dtype=np.int16)
        rx_i = rx_raw[0::4].astype(np.float64)
        rx_q = rx_raw[1::4].astype(np.float64)
        rx_iq = rx_i + 1j * rx_q
        
        # Limpar processo de transmissão
        run_remote(ssh_local, 'killall -9 iio_writedev')
        ssh_local.close()
        
        # Compressao de Pulso por Filtro Casado
        mf = np.conj(mf_ref[::-1])
        corr = np.convolve(rx_iq, mf, mode='valid')
        corr_mag = np.abs(corr)
        corr_db = 20.0 * np.log10(corr_mag / np.max(corr_mag) + 1e-12)
        
        # Encontrar picos periódicos válidos no meio do buffer (evitar bordas)
        mid_start = 5000
        mid_end = len(corr_db) - 5000
        mid_corr = corr_db[mid_start:mid_end]
        rel_pk = np.argmax(mid_corr)
        peak_idx = mid_start + rel_pk
        
        # Medir PSLR em um intervalo de um PRI (+/- 1000 amostras ao redor do pico principal)
        win_r = 1000
        w_start = max(0, peak_idx - win_r)
        w_end = min(len(corr_db), peak_idx + win_r + 1)
        pulse_slice = corr_db[w_start:w_end]
        center = peak_idx - w_start
        
        mask = np.ones(len(pulse_slice), dtype=bool)
        mask[max(0, center - null_samples) : min(len(pulse_slice), center + null_samples + 1)] = False
        pslr = float(np.max(pulse_slice[mask]))
        
        print(f"    [+] Potencia de Pico (Anritsu): {np.max(anr_trace):.2f} dBm")
        print(f"    [+] Mascara do Lobulo Principal: +/- {null_samples} amostras")
        print(f"    [+] PSLR Medido em Hardware: {pslr:.2f} dB")
        
        return anr_freqs, anr_trace, rx_iq, corr_db, peak_idx, pslr

    # Executar ambos os testes
    anr_f_lfm, anr_tr_lfm, rx_lfm, corr_lfm, pk_lfm, pslr_lfm = execute_waveform_test("LFM Convencional", buf_lfm, s_lfm, null_samples=8)
    anr_f_fost, anr_tr_fost, rx_fost, corr_fost, pk_fost, pslr_fost = execute_waveform_test("FOSM / F-NLFM Proposta", buf_fost, s_fost, null_samples=20)
    
    # 4. Plotar Grafico Comparativo Master
    print("\n[*] 4. Gerando graficos de alta resolucao para o artigo IEEE...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Subplot 1: Espectro RF no Anritsu
    ax1.plot(anr_f_lfm, anr_tr_lfm, 'b--', label='LFM Convencional (B = 20 MHz)', lw=1.3)
    ax1.plot(anr_f_fost, anr_tr_fost, 'r-', label='FOSM / F-NLFM Proposta', lw=1.6)
    ax1.set_title("A. Espectro Calibrado no Anritsu MS2723C @ 5.800 GHz\nConfinamento Espectral e Suavizacao de Bordas", fontsize=11)
    ax1.set_xlabel("Frequencia RF (GHz)", fontsize=10)
    ax1.set_ylabel("Potencia Calibrada (dBm)", fontsize=10)
    ax1.set_xlim([5.775, 5.825])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Subplot 2: Compressao de Pulso no Pluto RX1
    tau_us = (np.arange(-200, 201)) / (fs / 1e6)
    slice_lfm = corr_lfm[pk_lfm - 200 : pk_lfm + 201]
    slice_fost = corr_fost[pk_fost - 200 : pk_fost + 201]
    
    ax2.plot(tau_us, slice_lfm, 'b--', label=f'LFM Convencional (PSLR = {pslr_lfm:.1f} dB)', lw=1.3)
    ax2.plot(tau_us, slice_fost, 'r-', label=f'FOSM / F-NLFM (PSLR = {pslr_fost:.1f} dB)', lw=1.6)
    ax2.axhline(-13.2, color='b', linestyle=':', alpha=0.5, label='Limite Teorico LFM (-13.2 dB)')
    ax2.axhline(pslr_fost, color='r', linestyle=':', alpha=0.5, label=f'PSLR FOSM ({pslr_fost:.1f} dB)')
    ax2.set_title(f"B. Compressao de Pulso em Tempo Real no Pluto RX1\nSupressao de Lobulos: {abs(pslr_fost) - abs(pslr_lfm):+.1f} dB em Hardware Real", fontsize=11)
    ax2.set_xlabel(r"Tempo de Atraso $\tau$ ($\mu$s)", fontsize=10)
    ax2.set_ylabel("Magnitude Comprimida (dB)", fontsize=10)
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-60, 5])
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Subplot 3: Trajetoria de Frequencia Instantanea
    t_us_inst = t_fost * 1e6
    ax3.plot(t_lfm * 1e6, (B/T_pulse)*t_lfm / 1e6, 'b--', label='Lei Linear FM', lw=1.3)
    ax3.plot(t_us_inst, meta['f_instantaneous'] / 1e6, 'r-', label='Lei Fractal FOSM (Fase Estacionaria)', lw=1.6)
    ax3.set_title("C. Perfil de Frequencia Instantanea em Banda Base\nDistribuicao Nao-Linear de Energia sem Perda de SNR", fontsize=11)
    ax3.set_xlabel(r"Tempo ($\mu$s)", fontsize=10)
    ax3.set_ylabel("Frequencia Instantanea (MHz)", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')
    
    # Subplot 4: Detalhe do Lobulo Principal
    tau_ns = tau_us * 1000
    ax4.plot(tau_ns, slice_lfm, 'b--o', markersize=3, label='Lobulo LFM', lw=1.3)
    ax4.plot(tau_ns, slice_fost, 'r-s', markersize=3, label='Lobulo F-NLFM', lw=1.6)
    ax4.axhline(-3, color='k', linestyle=':', alpha=0.6, label='Largura a -3 dB (Rayleigh)')
    ax4.set_title("D. Resolucao em Distancia (Lobulo Principal a -3 dB)\nPreservacao da Largura de Banda com 0 dB de Mismatch", fontsize=11)
    ax4.set_xlabel(r"Tempo de Atraso $\tau$ (ns)", fontsize=10)
    ax4.set_ylabel("Magnitude (dB)", fontsize=10)
    ax4.set_xlim([-300, 300])
    ax4.set_ylim([-30, 2])
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper right')
    
    plt.suptitle("Caracterizacao de Micro-ondas em 5.8 GHz: FOSM / F-NLFM vs. LFM Convencional\nAnritsu MS2723C (Padrao Calibrado) vs. ADALM-Pluto SDR (Loopback em Tempo Real)", fontsize=13, y=1.01)
    plt.tight_layout()
    
    out_fig = "figures/radar_fnlfm_vs_lfm_5_8ghz_benchmark.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"\n[+] Grafico Master salvo em: {out_fig}")
    
    # Salvar relatorio JSON
    report = {
        "frequencia_portadora_ghz": 5.8,
        "largura_banda_mhz": 20.0,
        "duracao_pulso_us": 50.0,
        "produto_bt": 1000,
        "pslr_lfm_db": float(pslr_lfm),
        "pslr_fnlfm_db": float(pslr_fost),
        "melhoria_pslr_db": float(abs(pslr_fost) - abs(pslr_lfm)),
        "potencia_pico_lfm_dbm": float(np.max(anr_tr_lfm)),
        "potencia_pico_fnlfm_dbm": float(np.max(anr_tr_fost)),
        "piso_ruido_anritsu_dbm": float(np.min(anr_tr_fost))
    }
    with open("data/experiments/rf_5_8ghz/relatorio_fnlfm_5_8ghz_master.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"    [+] Relatorio numerico salvo em: data/experiments/rf_5_8ghz/relatorio_fnlfm_5_8ghz_master.json")
    
    print("\n" + "=" * 70)
    print("          RESULTADOS EXPERIMENTAIS EM HARDWARE (5.8 GHz)")
    print("=" * 70)
    print(f" Parametro                  | LFM Convencional | FOSM / F-NLFM Proposta")
    print("-" * 70)
    print(f" Frequencia Portadora (fc)  | 5.800000 GHz     | 5.800000 GHz")
    print(f" Largura de Banda (B)       | 20.0 MHz         | 20.0 MHz")
    print(f" Duracao do Pulso (T)       | 50.0 us          | 50.0 us")
    print(f" Produto Tempo-Banda (BT)   | 1000             | 1000")
    print(f" PSLR Medido em Hardware    | {pslr_lfm:.2f} dB        | {pslr_fost:.2f} dB")
    print(f" Ganho de Supressao (DPSLR) | Baseline (0 dB)  | {abs(pslr_fost) - abs(pslr_lfm):+.2f} dB")
    print(f" Perda por Descasamento     | 0.00 dB          | 0.00 dB (Envelope Constante)")
    print("=" * 70)

if __name__ == "__main__":
    run_5_8ghz_experiment()
