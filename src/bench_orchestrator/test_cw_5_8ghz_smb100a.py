"""
CW Tone 5.8 GHz Generation and Anritsu Spectrum Capture Verification
Uses R&S SMB100A to generate a pure 5.8 GHz CW carrier at -10 dBm and captures trace on Anritsu MS2723C.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.instrumentation.anritsu_ms2723c import AnritsuMS2723C
from src.instrumentation.rs_smb100a import RSSMB100A

def main():
    print("=" * 75)
    print("     TESTE DE VERIFICAÇÃO COM TOM CW EM 5.8 GHz (R&S SMB100A -> ANRITSU)")
    print("=" * 75)
    
    os.makedirs("figures", exist_ok=True)
    
    anritsu_ip = "192.168.1.187"
    smb_ip = "192.168.1.204"
    fc_hz = 5.8e9
    fc_str = "5.8GHz"
    span_str = "50MHz"
    power_dbm = -10.0 # Nível seguro
    
    # 1. Conectar ao Gerador R&S SMB100A
    print(f"\n[*] 1. Conectando ao R&S SMB100A em {smb_ip}...")
    smb = RSSMB100A(ip=smb_ip, port=5025)
    idn_smb = smb.connect()
    print(f"    [+] R&S Conectado: {idn_smb}")
    
    # 2. Configurar o SMB100A para gerar CW em 5.8 GHz a -10 dBm
    print(f"[*] 2. Configurando SMB100A: Freq = {fc_str}, Potência = {power_dbm} dBm...")
    smb.set_frequency(fc_hz)
    smb.set_power(power_dbm)
    smb.set_rf_output(True)
    print("    [+] Saída de RF do SMB100A LIGADA (ON)!")
    
    time.sleep(1.0)
    
    # 3. Conectar ao Analisador Anritsu MS2723C
    print(f"\n[*] 3. Conectando ao Anritsu MS2723C em {anritsu_ip}...")
    anritsu = AnritsuMS2723C(ip=anritsu_ip, port=9001)
    try:
        idn_anr = anritsu.connect()
        print(f"    [+] Anritsu Conectado: {idn_anr}")
        
        # 4. Configurar o Anritsu
        print(f"[*] 4. Configurando Anritsu: Centro = {fc_str}, Span = {span_str}, RBW = 30 kHz...")
        anritsu.set_center_freq(fc_str)
        anritsu.set_span(span_str)
        anritsu.set_rbw_vbw("30kHz", "10kHz")
        
        print("[*] Aguardando varredura completa do Anritsu (3 segundos)...")
        time.sleep(3.0)
        
        # 5. Capturar o Trace completo
        print("[*] 5. Capturando Trace do Anritsu...")
        trace_dBm = anritsu.get_trace_data(1)
        print(f"    [+] Sucesso! {len(trace_dBm)} pontos de espectro capturados.")
        
        # Eixo de frequências
        f_span_val = 50e6
        freqs_ghz = np.linspace(fc_hz - f_span_val/2, fc_hz + f_span_val/2, len(trace_dBm)) / 1e9
        
        peak_idx = np.argmax(trace_dBm)
        peak_freq = freqs_ghz[peak_idx]
        peak_power = trace_dBm[peak_idx]
        
        print(f"\n[+] PICO DETECTADO NO ANRITSU:")
        print(f"    -> Frequência de Pico: {peak_freq:.6f} GHz (Esperado: 5.800000 GHz)")
        print(f"    -> Potência no Pico  : {peak_power:.2f} dBm")
        print(f"    -> Piso de Ruído     : {np.min(trace_dBm):.2f} dBm")
        print(f"    -> Relação Sinal/Ruído (SNR): {peak_power - np.min(trace_dBm):.2f} dB")
        
        # 6. Plotar Gráfico de Alta Resolução
        plt.figure(figsize=(10, 5))
        plt.plot(freqs_ghz, trace_dBm, 'b-', lw=1.3, label='Espectro Medido (Anritsu MS2723C)')
        plt.plot(peak_freq, peak_power, 'ro', label=f'Pico CW @ {peak_freq:.4f} GHz ({peak_power:.1f} dBm)')
        plt.title(f"Verificação de Sinal CW em 5.8 GHz\nGerador: {idn_smb} | Analisador: {idn_anr}", fontsize=10)
        plt.xlabel("Frequência (GHz)", fontsize=10)
        plt.ylabel("Potência (dBm)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        out_fig = "figures/cw_5_8ghz_anritsu.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"\n[+] Gráfico salvo com sucesso em: {out_fig}")
        
    finally:
        # Desligar RF por segurança
        smb.set_rf_output(False)
        print("[*] Saída de RF do SMB100A desligada (OFF) por segurança.")
        smb.close()
        anritsu.close()

if __name__ == "__main__":
    main()
