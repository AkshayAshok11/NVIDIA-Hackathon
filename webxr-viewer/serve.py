#!/usr/bin/env python3
"""
Serves the WebXR viewer over HTTPS on your local network, so a Meta Quest
on the same WiFi can open it in the browser and enter VR.

WHY HTTPS IS REQUIRED (not optional):
WebXR only works in a "secure context" — https:// or localhost. Since the
Quest is a separate device on your network, it can't use "localhost" (that
would mean the Quest itself, not this machine) — it needs your machine's
real local IP address, which means it needs a real HTTPS certificate, even
if that certificate is self-signed and your browser has to click through a
warning once.

This script generates a self-signed cert on first run (if none exists) and
serves the current directory over HTTPS.

Usage:
    python3 serve.py
    (then open the printed URL in the Quest's browser)
"""

import http.server
import ssl
import socket
import subprocess
import sys
from pathlib import Path

PORT = 8443
CERT_FILE = Path("cert.pem")
KEY_FILE = Path("key.pem")


def get_local_ip() -> str:
    """Find this machine's LAN IP — the address the Quest needs to reach it at."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send anything — just asks the OS which local
        # interface/IP would be used to reach an external address.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_self_signed_cert(ip: str) -> None:
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    print("No certificate found — generating a self-signed one (one-time setup)...")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
            "-days", "365", "-nodes",
            "-subj", "/CN=timebox-local",
            "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
    )
    print("Certificate generated.")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Adds the CORS/headers some browsers want for module scripts + large
    binary fetches (the .splat/.ply file), and quiets down default logging
    a bit for readability."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def main():
    local_ip = get_local_ip()
    ensure_self_signed_cert(local_ip)

    httpd = http.server.HTTPServer(("0.0.0.0", PORT), QuietHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    url = f"https://{local_ip}:{PORT}/"
    print(f"\nServing Timebox viewer over HTTPS.")
    print(f"\n  On this machine:   https://localhost:{PORT}/")
    print(f"  On the Quest:      {url}")
    print(f"\nOn the Quest, you'll see a certificate warning (self-signed cert)")
    print(f"— tap 'Advanced' then 'Proceed', it's expected, this is your own")
    print(f"local server, not a real security risk on your own network.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
