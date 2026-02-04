#!/usr/bin/env python
"""
Minimal smoke test for the stockfish_hybrid_binding module.

Use this in CI to ensure the pybind11 extension loads, can set a position,
and returns a numeric evaluation.
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "pybind11"))


def main() -> None:
    from stockfish_hybrid_binding import StockfishHybridEngine

    engine = StockfishHybridEngine(binary_dir=str(REPO_ROOT / "Stockfish_hybrid" / "src"))
    engine.set_fen("startpos", chess960=False, moves=[])
    score = engine.evaluate(white_pov=True)
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"Expected numeric evaluation, got {score!r}")
    print(f"[smoke_test] evaluate(startpos) -> {score} cp")


if __name__ == "__main__":
    main()
