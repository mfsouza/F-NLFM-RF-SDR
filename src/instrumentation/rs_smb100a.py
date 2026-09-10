"""
Rohde & Schwarz SMB100A Signal Generator (100 kHz - 12.75 GHz) Driver
Communicates via standard SCPI over Raw Socket (Port 5025) or VXI-11.
"""

import socket
import time

class RSSMB100A:
    def __init__(self, ip="192.168.1.204", port=5025, timeout=5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        """Establish TCP socket connection to R&S SMB100A."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        idn = self.query("*IDN?")
        return idn

    def send_cmd(self, cmd):
        """Send SCPI command."""
        self.sock.sendall((cmd + "\n").encode("ascii"))
        time.sleep(0.02)

    def query(self, cmd):
        """Send SCPI query and receive response."""
        self.sock.sendall((cmd + "\n").encode("ascii"))
        return self.sock.recv(4096).decode("latin-1", errors="ignore").strip()

    def set_frequency(self, freq_hz):
        """Set output RF frequency in Hz, e.g., 5.8e9."""
        self.send_cmd(f":FREQ {freq_hz}")

    def set_power(self, power_dbm):
        """Set output RF level in dBm, e.g., -10.0."""
        self.send_cmd(f":POW {power_dbm}")

    def set_rf_output(self, enable=True):
        """Enable or disable RF output state."""
        state = "ON" if enable else "OFF"
        self.send_cmd(f":OUTP {state}")

    def close(self):
        """Close socket connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

if __name__ == "__main__":
    print("[*] R&S SMB100A Driver Module initialized.")
