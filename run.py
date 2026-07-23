#!/usr/bin/env python3
"""Launch the Nava Vision AI Model Validation UI."""

import os
import socket

import uvicorn


def _lan_ips():
    """Best-effort list of non-loopback IPv4 addresses for sharing on Wi‑Fi."""
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    # Fallback: UDP trick to find primary outbound interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127.") and ip not in ips:
            ips.insert(0, ip)
    except Exception:
        pass
    return ips


if __name__ == "__main__":
    # 0.0.0.0 = listen on all network interfaces (Wi‑Fi / LAN)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    print(f"Listening on http://{host}:{port}")
    print(f"Local:   http://127.0.0.1:{port}")
    for ip in _lan_ips():
        print(f"Network: http://{ip}:{port}   ← share with same-WiFi devs")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
    )
