#!/usr/bin/env python3
"""
Measure evaluation throughput with and without the embedded Stockfish NNUE binding.

Usage:
    python tools/benchmark_embedding_speed.py --iterations 200
"""

import argparse
import time

import chess

from HybridChessBot import HybridChessBot


def run_eval(bot: HybridChessBot, iterations: int) -> float:
    """Run `iterations` evaluations on the starting board and return elapsed seconds."""
    board = chess.Board()
    # Warmup (ensures kernels are compiled, caches allocated, etc.)
    bot.hybrid_evaluator.evaluate(board)
    start = time.time()
    for _ in range(iterations):
        bot.hybrid_evaluator.evaluate(board)
    return time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of evaluations per configuration (default: 200)",
    )
    args = parser.parse_args()

    print("== Embedded Stockfish NNUE binding ==")
    embed_bot = HybridChessBot(depth=1, time_limit=0.1, verbose=False)
    embed_seconds = run_eval(embed_bot, args.iterations)
    print(
        f"Embedded binding active: {embed_bot.hybrid_evaluator.nnue._binding_available()} | "
        f"{args.iterations} evals in {embed_seconds:.3f}s "
        f"({args.iterations / embed_seconds:.1f} evals/sec)"
    )

    print("\n== Fallback heuristic path ==")
    fallback_bot = HybridChessBot(depth=1, time_limit=0.1, verbose=False)
    nnue = fallback_bot.hybrid_evaluator.nnue
    # Tear down binding + binary interface to force heuristic policy/value.
    nnue._embedded_stockfish = None
    if getattr(nnue, "_stockfish_interface", None) is not None:
        try:
            nnue._stockfish_interface.close()
        except Exception:
            pass
    nnue._stockfish_interface = None
    nnue.use_stockfish_engine = False

    fallback_seconds = run_eval(fallback_bot, args.iterations)
    print(
        f"Embedded binding active: {fallback_bot.hybrid_evaluator.nnue._binding_available()} | "
        f"{args.iterations} evals in {fallback_seconds:.3f}s "
        f"({args.iterations / fallback_seconds:.1f} evals/sec)"
    )

    speedup = fallback_seconds / embed_seconds if embed_seconds > 0 else float("inf")
    print(f"\nSpeedup (embedded vs heuristic): {speedup:.2f}x")


if __name__ == "__main__":
    main()
