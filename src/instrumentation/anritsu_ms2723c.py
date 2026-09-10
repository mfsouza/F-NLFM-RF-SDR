"""
Anritsu MS2723C Spectrum Master (9 kHz - 13 GHz) SCPI Driver
Communicates via TCP Raw Socket (default port 9001) or VXI-11 over Ethernet.
"""

import socket
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

class AnritsuMS2723C:
    def __init__(self, ip="192.168.1.187", port=9001, timeout=15.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        """Establish TCP socket connection to Anritsu MS2723C."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        idn = self.query("*IDN?")
        return idn

    def send_cmd(self, cmd):
        """Send SCPI command."""
        self.sock.sendall((cmd + "\n").encode("ascii"))
        time.sleep(0.1)

    def query(self, cmd):
        """Send SCPI command and receive complete response."""
        self.sock.sendall((cmd + "\n").encode("ascii"))
        data = b""
        self.sock.settimeout(self.timeout)
        expected_len = None
        while True:
            try:
                chunk = self.sock.recv(8192)
                if not chunk:
                    break
                data += chunk
                
                # Check for IEEE 488.2 block header to determine exact length
                if expected_len is None and data.startswith(b"#"):
                    try:
                        num_digits = int(chr(data[1]))
                        payload_len = int(data[2:2 + num_digits].decode("ascii"))
                        expected_len = 2 + num_digits + payload_len
                    except Exception:
                        pass
                        
                if expected_len is not None:
                    if len(data) >= expected_len:
                        break
                else:
                    if b"\n" in data:
                        break
            except socket.timeout:
                if data:
                    break
                raise
        return data.decode("latin-1", errors="ignore").strip()

    def set_center_freq(self, freq_str="5.8GHz"):
        """Set center frequency, e.g., '2.4GHz', '5.8GHz', '915MHz'."""
        self.send_cmd(f":FREQ:CENT {freq_str}")

    def set_span(self, span_str="100MHz"):
        """Set frequency span, e.g., '50MHz', '100MHz'."""
        self.send_cmd(f":FREQ:SPAN {span_str}")

    def set_rbw_vbw(self, rbw_str="30kHz", vbw_str="10kHz"):
        """Set Resolution Bandwidth and Video Bandwidth."""
        self.send_cmd(f":BAND:RES {rbw_str}")
        self.send_cmd(f":BAND:VID {vbw_str}")

    def get_trace_data(self, trace_num=1):
        """
        Capture amplitude trace array (in dBm).
        Handles IEEE 488.2 block header format (#<num_digits><length>...).
        """
        raw_data = self.query(f":TRAC:DATA? {trace_num}")
        if raw_data.startswith("#"):
            num_digits = int(raw_data[1])
            raw_data = raw_data[2 + num_digits:]
        
        vals = [float(x) for x in raw_data.split(",") if x.strip()]
        return np.array(vals)

    def measure_obw(self, percent=99.0):
        """Query Occupied Bandwidth (99%)."""
        self.send_cmd(":CONF:OBW")
        self.send_cmd(f":OBW:PERCENT {percent}")
        self.send_cmd(":INIT")
        time.sleep(0.2)
        try:
            obw_hz = float(self.query(":MEAS:OBW?"))
            return obw_hz
        except Exception:
            return None

    def measure_channel_power(self):
        """Query integrated channel power (dBm)."""
        self.send_cmd(":CONF:CHP")
        self.send_cmd(":INIT")
        time.sleep(0.2)
        try:
            chp_dbm = float(self.query(":MEAS:CHP?"))
            return chp_dbm
        except Exception:
            return None

    def close(self):
        """Close socket connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anritsu MS2723C SCPI Capture")
    parser.add_argument("--ip", default="192.168.1.187", help="IP address")
    parser.add_argument("--port", type=int, default=9001, help="Port")
    parser.add_argument("--freq", default="5.8GHz", help="Center frequency")
    parser.add_argument("--span", default="100MHz", help="Span")
    parser.add_argument("--out", default="figures/anritsu_live_trace.png", help="Output PNG")
    args = parser.parse_args()

    inst = AnritsuMS2723C(ip=args.ip, port=args.port)
    try:
        idn = inst.connect()
        print(f"[+] Connected to: {idn}")
        inst.set_center_freq(args.freq)
        inst.set_span(args.span)
        trace = inst.get_trace_data(1)
        print(f"[+] Captured {len(trace)} trace points.")
        
        # Plot
        plt.figure(figsize=(9, 4.5))
        plt.plot(trace, "b-", lw=1.2, label=f"Center: {args.freq}, Span: {args.span}")
        plt.title(f"Anritsu MS2723C Metrological Spectrum - {idn}", fontsize=10)
        plt.xlabel("Trace Bin Index")
        plt.ylabel("Power (dBm)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        print(f"[+] Plot saved to: {args.out}")
    finally:
        inst.close()
