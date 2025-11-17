"""
NNUE Evaluator - Interfaces with Stockfish's NNUE implementation.

This module now queries the compiled Stockfish binary directly using the
`eval` command, which returns the raw NNUE evaluation without running any
search.  That keeps inference fast and ensures the training pipeline uses
the same numeric targets as inference.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from typing import Tuple, List, Optional

import chess
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chess_utils import board_to_tensor, extract_selection_features  # noqa: E402
import config  # noqa: E402


class StockfishNNUEInterface:
    """
    Lightweight helper that keeps a single Stockfish process alive and
    issues `eval` commands to read the raw NNUE scores.  The command is
    virtually instantaneous compared to `go depth=N` and does not perform
    any search, which removes the bottleneck that previously made every
    evaluation call spin up a full Stockfish analysis.
    """

    _EVAL_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)")

    def __init__(self, binary_path: str):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Stockfish binary not found: {binary_path}")

        self.binary_path = binary_path
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._initialize()

    def _initialize(self):
        self._send_command("uci")
        self._wait_for_token("uciok")

        # Ensure the large NNUE network is active if available.
        if getattr(config, "STOCKFISH_NNUE_PATH", None):
            self._send_command(f"setoption name EvalFile value {config.STOCKFISH_NNUE_PATH}")

        self._send_command("isready")
        self._wait_for_token("readyok")

    def _send_command(self, command: str):
        if self._process.stdin is None:
            raise RuntimeError("Stockfish stdin is closed")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> str:
        if self._process.stdout is None:
            raise RuntimeError("Stockfish stdout is closed")
        line = self._process.stdout.readline()
        if line == "":
            raise RuntimeError("Stockfish process terminated unexpectedly")
        return line.rstrip("\n")

    def _wait_for_token(self, token: str):
        while True:
            line = self._read_line()
            if token in line.strip():
                return

    def close(self):
        if self._process and self._process.poll() is None:
            try:
                self._send_command("quit")
            except Exception:
                pass

    def evaluate_board(self, board: chess.Board) -> float:
        """
        Returns the NNUE evaluation for the given position in centipawns.
        """
        fen = board.fen()
        with self._lock:
            self._send_command(f"position fen {fen}")
            self._send_command("eval")
            value = self._read_eval_block()
            # Drain the engine until it reports ready. This keeps the
            # stdout pipe synchronized with the commands we send.
            self._send_command("isready")
            self._wait_for_token("readyok")

        if value is None:
            raise RuntimeError("Unable to parse NNUE evaluation output")
        return value

    def _read_eval_block(self) -> Optional[float]:
        nnue_value = None
        final_value = None

        while True:
            line = self._read_line()
            stripped = line.strip()
            # Keep reading until Stockfish finishes printing the eval
            # table.  The line that starts with "Final evaluation"
            # indicates that the block is complete.
            if "NNUE evaluation" in stripped:
                parsed = self._parse_value(stripped)
                if parsed is not None:
                    nnue_value = parsed
            elif stripped.startswith("Final evaluation"):
                parsed = self._parse_value(stripped)
                if parsed is not None:
                    final_value = parsed
                break
        return final_value if final_value is not None else nnue_value

    def _parse_value(self, line: str) -> Optional[float]:
        match = self._EVAL_PATTERN.search(line)
        if not match:
            return None
        # Stockfish reports values in pawns. Convert to centipawns so the
        # rest of the engine can keep using the old scale.
        return float(match.group(1)) * 100.0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NNUEEvaluator(nn.Module):
    """
    Provides NNUE accumulator features and Stockfish NNUE evaluations.

    The old implementation used randomly initialised linear layers and
    triggered a full Stockfish search for every call to `forward`.  The
    new implementation produces deterministic feature vectors derived
    from the board representation and queries Stockfish's `eval` command
    to obtain fast, search-free NNUE values.
    """

    def __init__(self, nnue_weights_path: str = None, use_stockfish_engine: bool = True):
        super().__init__()

        self.output_dim = config.NNUE_FEATURE_DIM
        self.use_stockfish_engine = False
        self._warned_fallback = False

        self.register_buffer("_feature_template", torch.zeros(self.output_dim))

        self._stockfish_interface: Optional[StockfishNNUEInterface] = None
        if use_stockfish_engine and hasattr(config, "STOCKFISH_BINARY_PATH"):
            binary_path = config.STOCKFISH_BINARY_PATH
            if os.path.exists(binary_path):
                try:
                    self._stockfish_interface = StockfishNNUEInterface(binary_path)
                    self.use_stockfish_engine = True
                    print(f"✓ Using Stockfish NNUE evals from: {binary_path}")
                except Exception as exc:
                    print(f"⚠ Failed to initialize Stockfish NNUE interface: {exc}")
            else:
                print(f"⚠ Stockfish binary not found at {binary_path}")

    def compute_accumulator(self, board: chess.Board) -> torch.Tensor:
        """
        Build a deterministic 1024-dim feature vector using a flattened
        board tensor followed by repeated fast features (phase, material,
        etc.) so that the projection layer always sees meaningful input.
        """
        device = self._feature_template.device
        features = torch.zeros(self.output_dim, device=device)

        board_tensor = board_to_tensor(board).flatten().to(device)
        base_len = min(board_tensor.numel(), self.output_dim)
        features[:base_len] = board_tensor[:base_len]

        remaining = self.output_dim - base_len
        if remaining > 0:
            selector_feats = extract_selection_features(board, depth_remaining=0).to(device)
            if selector_feats.numel() > 0:
                repeat_count = (remaining + selector_feats.numel() - 1) // selector_feats.numel()
                tiled = selector_feats.repeat(repeat_count)[:remaining]
                features[base_len:] = tiled

        return features

    def forward(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Returns accumulator features and the Stockfish NNUE evaluation.
        """
        features = self.compute_accumulator(board)
        value = None

        if self.use_stockfish_engine and self._stockfish_interface is not None:
            try:
                value = self._stockfish_interface.evaluate_board(board)
            except Exception as exc:
                if not self._warned_fallback:
                    print(f"⚠ Falling back to heuristic NNUE value: {exc}")
                    self._warned_fallback = True

        if value is None:
            value = self._heuristic_value(board)

        return features, float(value)

    def _heuristic_value(self, board: chess.Board) -> float:
        """
        Simple material-based heuristic used only when the Stockfish NNUE
        process is unavailable.  The scale roughly matches centipawns so
        downstream consumers can keep the same thresholds.
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        score = 0
        for piece in board.piece_map().values():
            value = piece_values.get(piece.piece_type, 0)
            score += value if piece.color == chess.WHITE else -value

        # Encourage development and mobility slightly.
        mobility = len(list(board.legal_moves))
        score += 5 * (mobility - 20)

        return float(score)

    def batch_forward(self, boards: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor]:
        features_list = []
        values_list = []

        for board in boards:
            feat, val = self.forward(board)
            features_list.append(feat)
            values_list.append(val)

        features = torch.stack(features_list)
        values = torch.tensor(values_list, dtype=torch.float32)
        return features, values

    def __del__(self):
        if self._stockfish_interface is not None:
            self._stockfish_interface.close()


def create_nnue_evaluator(weights_path: str = None, use_stockfish: bool = True) -> NNUEEvaluator:
    """
    Factory function kept for backward compatibility.
    """
    return NNUEEvaluator(weights_path, use_stockfish_engine=use_stockfish)
