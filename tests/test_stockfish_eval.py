"""
Quick test of Stockfish evaluator functionality
"""

import sys
sys.path.append('src')

from utils.stockfish_eval import StockfishEvaluator
import chess

# Create evaluator
print("Initializing Stockfish evaluator...")
evaluator = StockfishEvaluator(depth=10)

# Test positions
test_positions = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Italian opening"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5", "Balanced position"),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Rook endgame, white better"),
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8", "Complex middlegame"),
]

print("\nEvaluating test positions:\n")
print("=" * 80)

for fen, description in test_positions:
    board = chess.Board(fen)
    value = evaluator.evaluate_position(board)
    print(f"\n{description}")
    print(f"FEN: {fen}")
    print(f"Evaluation: {value:.4f}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

print("\n" + "=" * 80)
print("\nStockfish evaluator test completed successfully!")

evaluator.close()
