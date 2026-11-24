#!/usr/bin/env bash
# Menyalakan prototipe DPdp (Streamlit).
#
#   ./run_web.sh            -> app di 0.0.0.0:8501, bisa dibuka di localhost/LAN
#   ./run_web.sh --lan-only -> sama, dengan alamat IP lokal yang lebih jelas
#
# Port bisa diubah: PORT=8600 ./run_web.sh
# Kalau app sudah jalan di port itu, script ini tidak menyalakan instance
# kedua, hanya mencetak alamatnya.
#
# Link publik permanen lewat Tailscale Funnel:
#   https://deteksipenyakitdaunpadi.tail7e934c.ts.net/
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8501}"
LAN_ONLY=0
[ "${1:-}" = "--lan-only" ] && LAN_ONLY=1

# Cari interpreter Python: venv di dalam projek, di atasnya, atau python3 di PATH.
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  for cand in ".venv/bin/python" ".venv/Scripts/python.exe" "../.venv/bin/python" \
              "../../.venv/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then PYTHON="$cand"; break; fi
  done
fi
PYTHON=${PYTHON:-python3}

HEALTH="http://localhost:$PORT/_stcore/health"
sudah_jalan() { curl -sf -o /dev/null "$HEALTH"; }

if sudah_jalan; then
  echo "==> app sudah jalan di port $PORT, tidak menyalakan instance baru."
else
  echo "==> menyalakan Streamlit di 0.0.0.0:$PORT"
  "$PYTHON" -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

  for _ in $(seq 1 30); do
    sleep 1
    sudah_jalan && break
  done

  if ! sudah_jalan; then
    echo "    app belum siap, cek log di atas."
    exit 1
  fi
  echo "    app siap"
fi

echo
echo "  Lokal : http://localhost:$PORT"

if command -v ip >/dev/null 2>&1; then
  LAN_IP=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)
fi
if [ -z "${LAN_IP:-}" ]; then
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
if [ -n "${LAN_IP:-}" ]; then
  echo "  LAN   : http://$LAN_IP:$PORT   (perangkat lain di WiFi yang sama)"
fi

echo
echo "  Publik (Tailscale Funnel, alamat tetap):"
echo "    https://deteksipenyakitdaunpadi.tail7e934c.ts.net/"
