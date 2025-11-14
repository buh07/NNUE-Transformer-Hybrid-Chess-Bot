"""
Stockfish Evaluation Utility for Target Value Generation
"""

import subprocess
import chess
import os
from typing import Optional
import config


class StockfishEvaluator:
    """
    Wrapper for Stockfish engine to get position evaluations
    """
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 10):
        """
        Initialize Stockfish evaluator
        
        Args:
            stockfish_path: Path to Stockfish binary (default from config)
            depth: Search depth for evaluation (default 10)
        """
        if stockfish_path is None:
            stockfish_path = config.STOCKFISH_BINARY_PATH
        
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish binary not found: {stockfish_path}")
        
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.process = None
        self._start_engine()
    
    def _start_engine(self):
        """Start the Stockfish process"""
        self.process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Initialize UCI
        self._send_command("uci")
        self._wait_for_response("uciok")
        self._send_command("setoption name Threads value 1")
        self._send_command("isready")
        self._wait_for_response("readyok")
    
    def _send_command(self, command: str):
        """Send command to Stockfish"""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
    
    def _wait_for_response(self, expected: str) -> str:
        """Wait for expected response from Stockfish"""
        if not self.process or not self.process.stdout:
            return ""
        
        while True:
            line = self.process.stdout.readline().strip()
            if expected in line:
                return line
    
    def _read_until_bestmove(self) -> dict:
        """
        Read output until 'bestmove' line, extracting evaluation info
        
        Returns:
            dict with 'score_cp' (centipawns) or 'score_mate' (moves to mate)
        """
        if not self.process or not self.process.stdout:
            return {}
        
        result = {}
        while True:
            line = self.process.stdout.readline().strip()
            
            # Parse score from info lines
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    score_type = parts[score_idx + 1]
                    score_value = int(parts[score_idx + 2])
                    
                    if score_type == "cp":
                        result['score_cp'] = score_value
                    elif score_type == "mate":
                        result['score_mate'] = score_value
                except (ValueError, IndexError):
                    pass
            
            # Stop at bestmove
            if line.startswith("bestmove"):
                break
        
        return result
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a chess position and return normalized value
        
        Args:
            board: Chess board to evaluate
        
        Returns:
            Normalized evaluation in [-1, 1] range
            Positive values favor current player
        """
        # Set position
        fen = board.fen()
        self._send_command(f"position fen {fen}")
        
        # Start search
        self._send_command(f"go depth {self.depth}")
        
        # Get evaluation
        eval_info = self._read_until_bestmove()
        
        # Convert to normalized value
        if 'score_mate' in eval_info:
            # Mate score: +1.0 if we're mating, -1.0 if being mated
            mate_in = eval_info['score_mate']
            return 1.0 if mate_in > 0 else -1.0
        
        elif 'score_cp' in eval_info:
            # Centipawn score: normalize using tanh
            # This maps centipawns to [-1, 1] with smooth transitions
            # A 3-pawn advantage (~300cp) gives ~0.76
            # A 5-pawn advantage (~500cp) gives ~0.92
            centipawns = eval_info['score_cp']
            normalized = centipawns / 400.0  # Scale factor
            
            # Apply tanh for smooth boundaries
            import math
            return math.tanh(normalized)
        
        else:
            # No score found, return draw evaluation
            return 0.0
    
    def evaluate_batch(self, boards: list) -> list:
        """
        Evaluate multiple positions
        
        Args:
            boards: List of chess.Board objects
        
        Returns:
            List of normalized evaluations
        """
        return [self.evaluate_position(board) for board in boards]
    
    def close(self):
        """Shut down the Stockfish engine"""
        if self.process:
            self._send_command("quit")
            self.process.wait(timeout=5)
            self.process = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


def evaluate_fen(fen: str, depth: int = 10) -> float:
    """
    Convenience function to evaluate a single FEN position
    
    Args:
        fen: FEN string of position
        depth: Search depth
    
    Returns:
        Normalized evaluation
    """
    evaluator = StockfishEvaluator(depth=depth)
    board = chess.Board(fen)
    value = evaluator.evaluate_position(board)
    evaluator.close()
    return value
