# SDR-Based Implementation and Metrological Microwave Characterization of Fractal NLFM Radar Waveforms (FOSM)

[![IEEE Standard](https://img.shields.io/badge/IEEE-Transactions-blue.svg)](https://www.ieee.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental framework, SCPI automated instrumentation drivers, SDR pulse generation/compression algorithms, and LaTeX manuscripts for the research paper:

> **"SDR-Based Implementation and Metrological Microwave Characterization of Fractal NLFM Radar Waveforms up to 5.8 GHz"**  
> *Target Journals*: IEEE Transactions on Aerospace and Electronic Systems (T-AES) / IEEE Transactions on Microwave Theory and Techniques (T-MTT) / IEEE Transactions on Instrumentation and Measurement (TIM).

---

## 🔬 Experimental Laboratory Infrastructure

The experimental rig combines high-performance Software-Defined Radio (SDR) with a traceable, multi-instrument microwave metrology chain:

```
                      +-------------------------------------------------------------+
                      |                 Master Python Orchestrator                  |
                      |          (SCPI via Ethernet TCP/IP + pyadi-iio USB)         |
                      +------------------------------+------------------------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v (USB / libiio)                                        v (SCPI / Ethernet)
            +---------------------------+                           +---------------------------+
            |  ADALM-PLUTO SDR (AD9361) |                           |  Anritsu MS2723C (13 GHz) |
            |  Zynq-7020 Transceiver    |                           |  Spectrum Analyzer        |
            |  (70 MHz - 6 GHz)         |                           +-------------^-------------+
            +-------------+-------------+                                         |
                          | (TX: 5.8 GHz, 0 dBm)                                  | (-20 dB Coupled)
                          v                                                       |
            +---------------------------+                                         |
            |  Omni Spectra Coupler     +-----------------------------------------+
            |  4.0 - 12.4 GHz (-20 dB)  |
            +-------------+-------------+
                          | (Through: Loss ~0.04 dB)
                          v
            +---------------------------+                           +---------------------------+
            |  MITEQ Directional Coupler+-------------------------->|  Giga-tronics 8541C       |
            |  (-10 dB Coupled Port)    |                           |  + 80351A Sensor (18 GHz) |
            +-------------+-------------+                           +---------------------------+
                          | (Through: Loss ~0.46 dB)
                          v
            +---------------------------+
            |  Calibrated SMA 20 dB Pad |
            +-------------+-------------+
                          | (Safe level: -25 dBm)
                          v
            +---------------------------+
            |  Pluto SDR RX (AD9361)    |
            |  Real-Time Matched Filter |
            +---------------------------+

            [ Calibration Standard ]: Rohde & Schwarz SMB100A (100 kHz - 12.75 GHz)
            [ Envelope & Peak Time ]: Agilent DSO3102A (100 MHz) + RF Crystal Schottky Detector
```

---

## 📊 Key Metrological Parameters & Measurements

1. **Spectral Confinement**: Occupied Bandwidth (OBW 99%) and Adjacent Channel Power Ratio (ACPR) measured by the Anritsu MS2723C.
2. **RF Peak-to-Average Power Ratio (PAPR)**: True RMS vs. Peak Power measured by Giga-tronics 8541C.
3. **Harmonic & Spurious Content**: SFDR, HD2 (11.6 GHz for 5.8 GHz fundamental) across full AD9361 bandwidth.
4. **Matched-Filter Harmonic Defocusing**: Experimental proof of non-linear distortion dispersion across radar pulse compression.

---

## 📁 Repository Structure

```text
├── docs/
│   └── overleaf/
│       └── IEEE_RF_FNLFM_SDR_Paper/    # Full LaTeX IEEEtran manuscript
│           ├── main.tex
│           ├── references.bib
│           └── IEEEtran.cls
├── src/
│   ├── rf_sdr/                         # Pluto SDR AD9361 Tx/Rx & Matched Filter
│   │   └── pluto_fnlfm_transceiver.py
│   ├── instrumentation/                # SCPI Hardware drivers
│   │   ├── anritsu_ms2723c.py
│   │   ├── rs_smb100a.py
│   │   └── gigatronics_8541c.py
│   └── bench_orchestrator/             # Master automated experiment runner
│       └── run_rf_experiment_suite.py
├── data/
│   └── experiments/                    # JSON/CSV raw laboratory datasets
├── figures/                            # High-resolution vector & raster plots
└── README.md
```

---

## 🚀 Quick Start

### 1. Requirements
```powershell
pip install numpy scipy matplotlib pyvisa pyadi-iio
```

### 2. Run Single Anritsu Metrology Capture
```powershell
python src/instrumentation/anritsu_ms2723c.py --ip 192.168.1.187 --freq 5.8GHz --span 100MHz
```

### 3. Run Full Automated RF Radar Suite
```powershell
python src/bench_orchestrator/run_rf_experiment_suite.py
```

---

## 📜 License
MIT License. Copyright (c) 2026.
