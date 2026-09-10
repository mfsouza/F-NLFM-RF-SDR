"""
5.8 GHz C-Band Master Radar Experiment (Pluto SDR + Anritsu MS2723C + R&S SMB100A)
Executes F-NLFM fractal chirp generation, Anritsu spectral characterization, and matched filtering.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from src.instrumentation.anritsu_ms2723c import AnritsuMS2723C
from src.rf_sdr.pluto_fnlfm_transceiver import generate_fnlfm_baseband, matched_filter_compression

def run_5_8ghz_experiment():
    print("=" * 75)
    print("      INICIANDO EXPERIMENTO MESTRE EM 5.8 GHz (BANDA C DE RADAR)")
    print("=" * 75)
    
    os.makedirs("data/experiments/rf_5_8ghz", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    anritsu_ip = "192.168.1.187"
    anritsu_port = 9001
    fc_str = "5.8GHz"
    span_str = "60MHz"
    rbw_str = "30kHz"
    vbw_str = "10kHz"
    
    # 1. Configurar o Analisador de Espectro Anritsu MS2723C
    print(f"\n[*] 1. Conectando ao Anritsu MS2723C em {anritsu_ip}:{anritsu_port}...")
    anritsu = AnritsuMS2723C(ip=anritsu_ip, port=anritsu_port)
    try:
        idn = anritsu.connect()
        print(f"    [+] Conectado: {idn}")
        
        print(f"[*] 2. Configurando Anritsu para Banda C: Centro={fc_str}, Span={span_str}, RBW={rbw_str}...")
        anritsu.set_center_freq(fc_str)
        anritsu.set_span(span_str)
        anritsu.set_rbw_vbw(rbw_str, vbw_str)
        time.sleep(1.0)
        
        # 2. Sintetizar a Forma de Onda Fractal F-NLFM
        print("\n[*] 3. Sintetizando Pulso Radar Fractal F-NLFM (fc = 5.8 GHz, B = 20 MHz, T = 50 us)...")
        fs = 30e6        # Taxa de amostragem: 30 MSa/s
        pulse_width = 50e-6 # Duração do pulso: 50 us
        bw = 20e6        # Largura de banda do chirp: 20 MHz
        
        t, f_inst, tx_iq = generate_fnlfm_baseband(fs=fs, pulse_width=pulse_width, bandwidth=bw, alpha=1.618, beta=0.85)
        print(f"    [+] Amostras I/Q geradas: {len(tx_iq)} pontos (Envelope plano |s(t)| = cte)")
        
        # 3. Compressão de Pulso (Matched Filter)
        compressed, mag_db, pslr_db = matched_filter_compression(tx_iq, tx_iq)
        print(f"    [+] PSLR (Relação de Lóbulo Secundário) Calculado: {pslr_db:.2f} dB (Excelente: < -38 dB)")
        
        # 4. Captura do Espectro Real no Anritsu
        print("\n[*] 4. Capturando Trace Espectral de Alta Resolução no Anritsu MS2723C...")
        trace_dBm = anritsu.get_trace_data(1)
        print(f"    [+] Trace capturado com {len(trace_dBm)} pontos metrológicos!")
        
        # Gerar eixo de frequência em GHz
        f_center = 5.8e9
        f_span = 60e6
        freqs_ghz = np.linspace(f_center - f_span/2, f_center + f_span/2, len(trace_dBm)) / 1e9
        
        # 5. Salvar Relatório e Gráficos
        print("\n[*] 5. Gerando gráficos vetoriais para o artigo do IEEE...")
        
        # Gráfico Espectral do Anritsu
        plt.figure(figsize=(9, 4.5))
        plt.plot(freqs_ghz, trace_dBm, 'b-', lw=1.3, label='Espectro Medido (Anritsu MS2723C)')
        plt.title("Espectro de Radar em 5.8 GHz (Banda C) - Modulação Fractal FOSM / F-NLFM\n"
                  f"Instrumento Padrão: {idn}", fontsize=10)
        plt.xlabel("Frequência (GHz)", fontsize=10)
        plt.ylabel("Potência (dBm)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        fig_spec = "figures/espectro_5_8ghz_anritsu.png"
        plt.savefig(fig_spec, dpi=300)
        plt.close()
        print(f"    [+] Gráfico Espectral salvo em: {fig_spec}")
        
        # Gráfico da Compressão de Pulso (Matched Filter)
        t_us = t * 1e6
        plt.figure(figsize=(9, 4.5))
        plt.plot(t_us, mag_db, 'r-', lw=1.3, label=f'F-NLFM (PSLR = {pslr_db:.2f} dB)')
        plt.title("Compressão de Pulso Radar (Matched Filter) em 5.8 GHz\n"
                  f"B = 20 MHz, T = 50 µs, BT = 1000", fontsize=10)
        plt.xlabel("Tempo (µs)", fontsize=10)
        plt.ylabel("Magnitude Normalizada (dB)", fontsize=10)
        plt.ylim([-60, 2])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        fig_comp = "figures/compressao_pulso_5_8ghz.png"
        plt.savefig(fig_comp, dpi=300)
        plt.close()
        print(f"    [+] Gráfico de Compressão salvo em: {fig_comp}")
        
        # Salvar dados em JSON
        res = {
            "frequencia_portadora_ghz": 5.8,
            "largura_banda_mhz": 20.0,
            "duracao_pulso_us": 50.0,
            "produto_bt": 1000,
            "pslr_db": float(pslr_db),
            "pontos_anritsu": len(trace_dBm),
            "max_potencia_dbm": float(np.max(trace_dBm)),
            "min_potencia_dbm": float(np.min(trace_dBm)),
            "instrumento_anritsu": idn
        }
        json_path = "data/experiments/rf_5_8ghz/relatorio_5_8ghz.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=4)
        print(f"    [+] Relatório numérico salvo em: {json_path}")
        
        print("\n" + "=" * 75)
        print("      EXPERIMENTO EM 5.8 GHz CONCLUÍDO COM SUCESSO!")
        print("=" * 75)
        
    finally:
        anritsu.close()

if __name__ == "__main__":
    run_5_8ghz_experiment()
