#!/usr/bin/env bash
# Rebuild Stockfish (libs + binary) and the pybind11 binding in one go.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARCH="${ARCH:-${HYBRID_STOCKFISH_ARCH:-x86-64}}"
JOBS="${JOBS:-${HYBRID_STOCKFISH_JOBS:-$(nproc || sysctl -n hw.ncpu || echo 4)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

pushd "${ROOT_DIR}" >/dev/null
echo "[build_binding] Building Stockfish_hybrid (ARCH=${ARCH}, JOBS=${JOBS})"
ARCH="${ARCH}" JOBS="${JOBS}" ./build_stockfish_hybrid.sh
popd >/dev/null

pushd "${ROOT_DIR}/bindings/pybind11" >/dev/null
echo "[build_binding] Building pybind11 module with ${PYTHON_BIN}"
PYTHONPATH="${ROOT_DIR}/bindings/pybind11:${PYTHONPATH:-}" "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

echo "[build_binding] Done."
