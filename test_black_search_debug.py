"""
Debug test - manually trace search for Black's first move
"""
import chess
import torch
from HybridChessBot import HybridChessBot

bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    verbose=False,
    depth=2,
    time_limit=5.0
)

board = chess.Board()
board.push_san("e4")

print("Position after 1.e4:")
print(board)
print()
print("Black to move. Let's evaluate some candidate moves:")
print()

# Manually evaluate each move at depth 1
for move in list(board.legal_moves)[:8]:  # Just first 8 moves
    test_board = board.copy()
    test_board.push(move)
    
    # Get raw evaluation (from White's POV)
    score = bot.search_engine.evaluate_position(test_board)
    
    print(f"{move.uci():6s} -> {score:7.3f} cp (White's POV)")

print()
print("Black should prefer moves with MORE NEGATIVE scores")
print("(because those are better for Black)")
print()

# Now run the actual search
print("Running depth-2 search...")
best_move, best_score = bot.search_engine.iterative_deepening(board, max_depth=2, time_limit=5.0)
print(f"Bot chose: {best_move.uci()}, score: {best_score:.3f}")
