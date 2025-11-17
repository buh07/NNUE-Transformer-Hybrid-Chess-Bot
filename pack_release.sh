#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-hybrid_bot_release.zip}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

INCLUDE=(
  README.md
  HybridChessBot.py
  ChessGame.py
  play_vs_stockfish.py
  config.py
  requirements.txt
  src
  chess-transformers
  Stockfish
  checkpoints
)

echo "Creating $OUT ..."
zip -r "$OUT" "${INCLUDE[@]}"
echo "Done -> $OUT"
