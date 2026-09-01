#!/usr/bin/env bash
# geox-g01 single-node multi-GPU / hostname-resolution diagnostic
# Read-only: does NOT modify /etc/hosts, firewall, DNS, or network settings.
#
# Usage:
#   cd /home/jincai_guo/tianqi/CVPR2027/LaWAM
#   TEST_GPUS=4,5,6,7 DDP_TIMEOUT=45 bash scripts/diagnose_geox_ddp.sh

set -uo pipefail

TEST_GPUS="${TEST_GPUS:-4,5,6,7}"
DDP_TIMEOUT="${DDP_TIMEOUT:-45}"
ACCEL_CONFIG="${ACCEL_CONFIG:-starVLA/config/accelerate/ddp_bf16.yaml}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${PWD}/geox_ddp_diagnosis_${TS}.log"

exec > >(tee "${LOG}") 2>&1

section() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

section "0. Basic information"
date
whoami
pwd
uname -a
echo "TEST_GPUS=${TEST_GPUS}"
echo "DDP_TIMEOUT=${DDP_TIMEOUT}"
echo "ACCEL_CONFIG=${ACCEL_CONFIG}"

section "1. Hostname and local name resolution"
hostname || true
echo
hostname -f || true

HOSTNAME_SHORT="$(hostname 2>/dev/null || true)"

echo
echo "\$ getent hosts ${HOSTNAME_SHORT}"
getent hosts "${HOSTNAME_SHORT}" || true

echo
echo "\$ getent ahosts ${HOSTNAME_SHORT}"
getent ahosts "${HOSTNAME_SHORT}" || true

echo
echo "\$ getent hosts localhost"
getent hosts localhost || true

echo
echo "--- /etc/hosts ---"
cat /etc/hosts || true

echo
echo "--- hosts line in /etc/nsswitch.conf ---"
grep -E '^[[:space:]]*hosts:' /etc/nsswitch.conf || true

echo
echo "--- /etc/resolv.conf ---"
cat /etc/resolv.conf || true

if command -v resolvectl >/dev/null 2>&1; then
  echo
  echo "--- resolvectl status ---"
  resolvectl status || true
fi

section "2. Python socket resolution"
python - <<'PY'
import socket

h = socket.gethostname()
print("socket.gethostname():", h)

for target in [h, "localhost", "127.0.0.1"]:
    print(f"\ngetaddrinfo({target!r}):")
    try:
        for item in socket.getaddrinfo(target, None):
            print(" ", item)
    except Exception as e:
        print(" FAILED:", repr(e))

print("\ngetfqdn():")
try:
    print(socket.getfqdn())
except Exception as e:
    print(" FAILED:", repr(e))
PY

section "3. Network interfaces and routing"
ip -br addr || true
echo
ip route || true

DEFAULT_IF="$(ip route 2>/dev/null | awk '/default/ {print $5; exit}')"
HOST_IP=""
if [[ -n "${DEFAULT_IF}" ]]; then
  HOST_IP="$(ip -4 -o addr show dev "${DEFAULT_IF}" 2>/dev/null | awk '{split($4,a,"/"); print a[1]; exit}')"
fi

echo
echo "Detected DEFAULT_IF=${DEFAULT_IF:-<none>}"
echo "Detected HOST_IP=${HOST_IP:-<none>}"

if [[ -n "${HOST_IP}" ]]; then
  echo
  echo "\$ getent hosts ${HOST_IP}"
  getent hosts "${HOST_IP}" || true

  echo
  echo "\$ ip route get ${HOST_IP}"
  ip route get "${HOST_IP}" || true

  echo
  echo "Python reverse lookup for ${HOST_IP}:"
  python - "${HOST_IP}" <<'PY'
import socket, sys
ip = sys.argv[1]
for flags, label in [(0, "normal"), (socket.NI_NUMERICHOST, "numeric-only")]:
    try:
        print(label, ":", socket.getnameinfo((ip, 0), flags))
    except Exception as e:
        print(label, "FAILED:", repr(e))
PY
fi

section "4. Local TCP self-connect test using numeric IPv4"
if [[ -n "${HOST_IP}" ]]; then
  python - "${HOST_IP}" <<'PY'
import socket, sys, threading

ip = sys.argv[1]
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((ip, 0))
server.listen(1)
port = server.getsockname()[1]

result = {"server": None, "client": None}

def accept_once():
    try:
        conn, addr = server.accept()
        data = conn.recv(16)
        conn.sendall(b"pong")
        conn.close()
        result["server"] = ("OK", addr, data)
    except Exception as e:
        result["server"] = ("FAILED", repr(e))

t = threading.Thread(target=accept_once, daemon=True)
t.start()

try:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(5)
    client.connect((ip, port))
    client.sendall(b"ping")
    data = client.recv(16)
    result["client"] = ("OK", data)
    client.close()
except Exception as e:
    result["client"] = ("FAILED", repr(e))

t.join(timeout=5)
server.close()

print("bind_ip:", ip)
print("port:", port)
print("server:", result["server"])
print("client:", result["client"])

if not (result["server"] and result["server"][0] == "OK" and
        result["client"] and result["client"][0] == "OK"):
    raise SystemExit(2)
PY
  echo "TCP self-connect exit=$?"
else
  echo "[SKIP] Could not determine an IPv4 address."
fi

section "5. Relevant distributed environment variables"
env | grep -E '^(MASTER_|NCCL_|TORCH_NCCL_|GLOO_|CUDA_VISIBLE_DEVICES|WORLD_SIZE|RANK|LOCAL_RANK)=' | sort || true

section "6. PyTorch / Accelerate / CUDA versions"
python - <<'PY'
import sys
print("python:", sys.version.replace("\n", " "))

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    print("torch.version.cuda:", torch.version.cuda)
    try:
        print("torch.cuda.nccl.version:", torch.cuda.nccl.version())
    except Exception as e:
        print("torch.cuda.nccl.version FAILED:", repr(e))
except Exception as e:
    print("torch import FAILED:", repr(e))

try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except Exception as e:
    print("accelerate import FAILED:", repr(e))
PY

echo
nvidia-smi -L || true

section "7. Accelerate configuration"
if [[ -f "${ACCEL_CONFIG}" ]]; then
  cat "${ACCEL_CONFIG}"
else
  echo "[WARN] Accelerate config not found: ${ACCEL_CONFIG}"
fi

section "8. Minimal 4-GPU Accelerate/DDP smoke test"
SMOKE="/tmp/geox_ddp_smoke_${TS}_$$.py"

cat > "${SMOKE}" <<'PY'
import os
import socket
import torch
import torch.distributed as dist
from accelerate import Accelerator

print(
    "[before Accelerator]",
    "pid=", os.getpid(),
    "hostname=", socket.gethostname(),
    "MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
    "MASTER_PORT=", os.environ.get("MASTER_PORT"),
    "RANK=", os.environ.get("RANK"),
    "LOCAL_RANK=", os.environ.get("LOCAL_RANK"),
    flush=True,
)

acc = Accelerator()

print(
    "[after Accelerator]",
    "rank=", acc.process_index,
    "local_rank=", acc.local_process_index,
    "world=", acc.num_processes,
    "device=", acc.device,
    flush=True,
)

x = torch.tensor([float(acc.process_index + 1)], device=acc.device)
dist.all_reduce(x)

print(
    "[all_reduce OK]",
    "rank=", acc.process_index,
    "value=", x.item(),
    flush=True,
)

acc.wait_for_everyone()
if acc.is_main_process:
    print("[SUCCESS] distributed smoke test passed", flush=True)
PY

if [[ -z "${HOST_IP}" ]]; then
  echo "[SKIP] DDP smoke: could not detect HOST_IP."
else
  PORT="$(python - "${HOST_IP}" <<'PY'
import socket, sys
ip = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, 0))
print(s.getsockname()[1])
s.close()
PY
)"

  echo "Using HOST_IP=${HOST_IP}"
  echo "Using PORT=${PORT}"
  echo "Using GPUs=${TEST_GPUS}"
  echo "Launching with a ${DDP_TIMEOUT}s timeout..."

  set +e
  CUDA_VISIBLE_DEVICES="${TEST_GPUS}" \
  GLOO_SOCKET_IFNAME="${DEFAULT_IF}" \
  NCCL_SOCKET_IFNAME="${DEFAULT_IF}" \
  NCCL_SOCKET_FAMILY=AF_INET \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  NCCL_DEBUG=WARN \
  timeout "${DDP_TIMEOUT}s" \
  accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes 4 \
    --main_process_ip "${HOST_IP}" \
    --main_process_port "${PORT}" \
    "${SMOKE}"
  DDP_RC=$?
  set -e

  echo
  echo "DDP_SMOKE_EXIT_CODE=${DDP_RC}"
  if [[ "${DDP_RC}" -eq 0 ]]; then
    echo "Interpretation: DDP smoke passed."
  elif [[ "${DDP_RC}" -eq 124 ]]; then
    echo "Interpretation: timeout expired; distributed initialization/collective did not finish."
  else
    echo "Interpretation: DDP smoke failed with a non-timeout error."
  fi
fi

rm -f "${SMOKE}" || true

section "9. Processes / listening sockets after smoke test"
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd \
  | grep -E 'accelerate|torchrun|train_starvla|geox_ddp_smoke' \
  | grep -v grep || true

echo
echo "Listening TCP sockets:"
ss -ltnp 2>/dev/null | head -100 || true

section "10. Summary for administrator"
echo "Hostname              : ${HOSTNAME_SHORT:-<unknown>}"
echo "Default interface     : ${DEFAULT_IF:-<unknown>}"
echo "Detected IPv4         : ${HOST_IP:-<unknown>}"
echo "Hostname getent result:"
getent hosts "${HOSTNAME_SHORT}" 2>/dev/null || echo "  <no result>"
echo
echo "Diagnostic log:"
echo "  ${LOG}"
echo
echo "Please share this log with the server administrator."
