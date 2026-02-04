#!/usr/bin/env bash
# Compile the hybrid Stockfish fork and copy the resulting binary to the expected config path.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SF_DIR="${ROOT_DIR}/Stockfish_hybrid"
LIB_NAME="libstockfish"

if [[ ! -d "${SF_DIR}/src" ]]; then
  echo "Hybrid Stockfish source not found under ${SF_DIR}/src" >&2
  exit 1
fi

ARCH="${ARCH:-${HYBRID_STOCKFISH_ARCH:-x86-64}}"
JOBS="${JOBS:-${HYBRID_STOCKFISH_JOBS:-$(nproc || sysctl -n hw.ncpu || echo 4)}}"

pushd "${SF_DIR}/src" >/dev/null
echo "Building Stockfish_hybrid (ARCH=${ARCH}, JOBS=${JOBS})..."
make clean >/dev/null 2>&1 || true
make BUILD_LIB=yes ARCH="${ARCH}" -j"${JOBS}" build
make BUILD_LIB=yes ARCH="${ARCH}" -j"${JOBS}" lib
make BUILD_LIB=yes ARCH="${ARCH}" -j"${JOBS}" shared-lib
popd >/dev/null

cp "${SF_DIR}/src/stockfish" "${ROOT_DIR}/Stockfish_hybrid/src/stockfish"
echo "Hybrid Stockfish binary available at ${ROOT_DIR}/Stockfish_hybrid/src/stockfish"
echo "Hybrid Stockfish libraries available at:"
echo "  - ${ROOT_DIR}/Stockfish_hybrid/src/${LIB_NAME}.a"
echo "  - ${ROOT_DIR}/Stockfish_hybrid/src/${LIB_NAME}.so"
