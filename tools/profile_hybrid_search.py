#!/usr/bin/env python3
"""
Profile the hybrid search stack on a fixed gauntlet of tactical-ish positions.

The script runs a series of searches, records per-depth instrumentation from
`AlphaBetaSearch` (nodes, TT hit rate, branching factor, etc.), and captures the
HybridEvaluator timing stats (NNUE backend usage, transformer calls).

Results are written to both JSON (full fidelity) and CSV (top-level summary)
under `logs/profiles/` so regressions can be compared over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
import time
import sys

import chess

# Ensure repository root is importable even when this script runs from tools/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HybridChessBot import HybridChessBot


# Deterministic set of diverse middlegame-ish positions generated with a fixed
# random seed (32 plies, mix of White/Black to move). Replace or extend with
# curated tactical suites as needed.
DEFAULT_FENS: List[str] = [
    "r2qkb1r/pppnpppp/5n2/3p4/4P3/2P4N/PP1P1PP1/RNBQKB1R w KQkq - 1 5",
    "rn1qkb1r/ppp1pp1p/3p2pn/8/6b1/1P5N/PNPPPPPP/R1BQKB1R b KQkq - 3 5",
    "rnbqkb1r/1pppnp1p/4p3/p5p1/4NPP1/7N/PPPPP2P/R1BQKB1R w KQkq - 0 6",
    "rnbqkbnr/1pp1p3/p2p2pB/5p2/8/1P1P3N/P1P1PPPP/RN1QKB1R b KQkq - 0 6",
    "rnbqkb1r/ppp2pp1/3ppn1p/8/P5P1/2N4B/1PPPPP1P/R1BQK1NR w KQkq - 2 5",
    "r1bqkbnr/1ppppppp/8/p7/n1PP4/P3B1P1/1P2PP1P/RN1QKBNR b KQkq - 0 5",
    "rnbq1knr/p1ppp2p/5ppb/1p5Q/3P4/1P2P2N/P1P2PPP/RNB1KB1R w KQ - 0 6",
    "rnbqkb1r/1p1pppp1/p4n2/2p4p/1PP2PP1/P6N/3PP2P/RNBQKB1R b KQkq - 0 6",
    "rnbqkb1r/p1p1pp1p/5np1/1p1p4/2P2P2/1PN5/P2PP1PP/R1BQKBNR w KQkq - 2 5",
    "rnbqkbnr/pp1pp1pp/8/2p2p2/4P3/N5PB/PPPP1P1P/R1BQ1KNR b kq - 0 5",
    "r1bqkb1r/pppp1ppp/4pn2/8/4PN2/1n1P4/PPP2PPP/RNBQKBR1 w Qkq - 2 6",
    "rnbqkbnr/2p1p1pp/pp6/3p4/P3pPP1/N2P4/1PP4P/R1BQKBNR b KQkq - 0 6",
    "r1bqkbnr/p1ppppp1/2n4p/1p2N3/7P/8/PPPPPPP1/R1BQKBNR w KQkq - 0 5",
    "r2qkbnr/pppb1ppp/n2p4/4p3/7P/P4NP1/1PPPPP2/RNBQKB1R b KQkq - 2 5",
    "rnbq1b1r/p2kpppp/2p4n/1p1p4/2P1P2P/N7/PP1P1PP1/1RBQKBNR w K - 2 6",
    "r1bqkb1r/pppp1pp1/n4n2/4p2p/2P1P3/BP6/P2PKPPP/RN1Q1BNR b kq - 1 6",
    "rnb1kbnr/1ppp2pp/p7/4pp2/N3P2q/7P/PPPP1PP1/R1BQKBNR w KQkq - 0 5",
    "rnbqkbr1/pppp1pp1/4pn1p/8/P2P4/3BPP2/1PP3PP/RNBQK1NR b KQq - 0 5",
    "rnbqkb1r/1p1pp2p/5ppn/p1p5/4P3/NP6/P1PP1PPP/R1BQKBNR w Kkq - 0 6",
    "r1bqkbnr/2p1p1pp/ppnp1p2/8/1PPP4/5P1N/P3P1PP/RNBQKB1R b KQkq - 0 6",
    "rn2kbnr/pppbqppp/3p4/4p3/6PP/2N5/PPPPPP2/1RBQKBNR w Kkq - 4 5",
    "r1bqkbnr/ppppppp1/7p/8/n1P3P1/N3P3/PP1PNP1P/R1BQKB1R b KQkq - 2 5",
    "rnbqkbnr/3pp1pp/5p2/1pp5/p3PB2/2NP2P1/PPP2P1P/R2QKBNR w KQkq - 0 6",
    "rnbqkbn1/1ppp1ppr/p6p/4p2P/2N5/4P3/PPPPNPP1/R1BQKB1R b KQq - 2 6",
    "1rbqkbr1/pppppppp/2n4n/8/4P1P1/N7/PPPPKP1P/R1BQ1BNR w - - 3 5",
    "r1bqkbnr/1pp1pp1p/p1n3p1/3p4/3P1P2/2N1P3/PPPQ2PP/R1B1KBNR b KQkq - 1 5",
    "rnb1kbnr/pp1ppp1p/2p1q1p1/4N3/6P1/N2P4/PPP1PP1P/R1BQKB1R w KQkq - 0 6",
    "r1bqk2r/pppp1pp1/2n2n2/4p2p/1b6/P1P2P1N/1PQPP1PP/RNB1KB1R b KQkq - 0 6",
    "r1bqkbnr/pp1pp1pp/2p5/n4p2/8/PPP2N2/3PPPPP/RNBQKB1R w KQkq - 0 5",
    "rnbqkbnr/pp1pp1p1/8/2p2p1p/6P1/1PP2N1B/P2PPP1P/RNBQK2R b KQkq - 0 5",
    "r1bqkbnr/1pppp1p1/p6p/4np2/3Q4/2P1P2P/PP1P1PP1/RNB1KBNR w KQkq - 4 6",
    "rnb1kb1r/1p1ppppp/p2P1n2/8/P7/2N5/2PPPPPP/1RBQKBNR b Kkq - 0 6",
]


@dataclass
class PositionProfile:
    index: int
    fen: str
    best_move: Optional[str]
    score_cp: Optional[float]
    elapsed_s: float
    nodes: int
    nps: float
    tt_hit_rate: float
    depth: int
    quiescence_nodes: int

    @classmethod
    def from_stats(
        cls,
        index: int,
        fen: str,
        best_move: Optional[chess.Move],
        score: Optional[float],
        elapsed: float,
        search_stats: dict,
    ) -> "PositionProfile":
        return cls(
            index=index,
            fen=fen,
            best_move=best_move.uci() if best_move else None,
            score_cp=score,
            elapsed_s=elapsed,
            nodes=search_stats.get("nodes_searched", 0),
            nps=search_stats.get("nps", 0.0),
            tt_hit_rate=search_stats.get("tt_hit_rate", 0.0),
            depth=search_stats.get("depth", 0),
            quiescence_nodes=search_stats.get("quiescence_nodes", 0),
        )


def load_fens(args: argparse.Namespace) -> List[str]:
    if args.fen_file:
        fen_path = Path(args.fen_file)
        with fen_path.open("r", encoding="utf-8") as fh:
            fens = [line.strip() for line in fh if line.strip()]
        if not fens:
            raise ValueError(f"No FENs found in {fen_path}")
        return fens
    return DEFAULT_FENS.copy()


def run_profile(args: argparse.Namespace) -> dict:
    fens = load_fens(args)
    if args.max_positions:
        fens = fens[: args.max_positions]

    bot = HybridChessBot(
        checkpoint=args.checkpoint,
        depth=args.depth if args.depth > 0 else None,
        time_limit=args.time_limit,
        device=args.device,
        verbose=args.verbose,
        evaluation_mode=args.eval_mode,
    )
    bot.hybrid_evaluator.reset_stats()

    profiles: List[PositionProfile] = []

    for idx, fen in enumerate(fens, start=1):
        board = chess.Board(fen)
        bot.search_engine.reset_statistics()
        start = time.perf_counter()
        move, score = bot.search_engine.iterative_deepening(
            board,
            max_depth=args.depth if args.depth > 0 else None,
            time_limit=args.time_limit if args.time_limit > 0 else None,
        )
        elapsed = time.perf_counter() - start

        search_stats = bot.search_engine.get_statistics()
        profile = PositionProfile.from_stats(
            index=idx,
            fen=fen,
            best_move=move,
            score=score,
            elapsed=elapsed,
            search_stats=search_stats,
        )
        profiles.append(profile)

        if args.verbose:
            print(
                f"[{idx}/{len(fens)}] move={profile.best_move} "
                f"score={profile.score_cp:.1f}cp nodes={profile.nodes} "
                f"tt_hit={profile.tt_hit_rate*100:.1f}% time={profile.elapsed_s:.2f}s"
            )

    aggregate = summarize_profiles(profiles)
    eval_snapshot = bot.hybrid_evaluator.get_stats()
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "args": vars(args),
        "positions": [asdict(p) for p in profiles],
        "summary": aggregate,
        "evaluation_stats": eval_snapshot,
    }


def summarize_profiles(profiles: Iterable[PositionProfile]) -> dict:
    profiles = list(profiles)
    total = len(profiles)
    if total == 0:
        return {}
    avg = lambda attr: sum(getattr(p, attr) for p in profiles) / total
    return {
        "positions": total,
        "avg_nodes": avg("nodes"),
        "avg_nps": avg("nps"),
        "avg_elapsed_s": avg("elapsed_s"),
        "avg_tt_hit_rate": avg("tt_hit_rate"),
        "avg_quiescence_nodes": avg("quiescence_nodes"),
    }


def write_outputs(report: dict, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    fieldnames = list(report["positions"][0].keys()) if report["positions"] else []
    if fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in report["positions"]:
                writer.writerow(row)

    print(f"[profile] Wrote JSON: {json_path}")
    if fieldnames:
        print(f"[profile] Wrote CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the hybrid search stack.")
    parser.add_argument("--checkpoint", default="checkpoints/best_phase2.pt", help="Hybrid checkpoint path.")
    parser.add_argument("--depth", type=int, default=4, help="Search depth per position (<=0 uses bot default).")
    parser.add_argument("--time-limit", type=float, default=0.0, help="Time limit per move in seconds (0 disables).")
    parser.add_argument("--device", default=None, help="Force device (cpu or cuda).")
    parser.add_argument("--fen-file", help="Optional file containing newline-separated FENs.")
    parser.add_argument("--max-positions", type=int, help="Limit number of FENs to profile.")
    parser.add_argument("--eval-mode", default=os.environ.get("HYBRID_EVAL_MODE", "auto"), choices=["auto", "nnue", "transformer"])
    parser.add_argument("--output-dir", default="logs/profiles", help="Directory to store profiling outputs.")
    parser.add_argument("--tag", default=None, help="Optional label appended to output filenames.")
    parser.add_argument("--verbose", action="store_true", help="Print per-position stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_profile(args)
    backend_usage = report.get("evaluation_stats", {}).get("nnue_backend_usage", {})
    if backend_usage:
        print("[profile] NNUE backend usage:")
        for backend_name, data in backend_usage.items():
            print(f"  - {backend_name}: {data.get('calls', 0)} calls ({data.get('time', 0.0):.2f}s)")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"profile_{timestamp}{tag}"
    write_outputs(report, Path(args.output_dir), stem)


if __name__ == "__main__":
    main()
