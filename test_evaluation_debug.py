"""
Quick test to see what evaluation values are being returned
"""
import chess
import sys
import torch
from HybridChessBot import HybridChessBot

# Create bot
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    verbose=True,
    depth=1,
    time_limit=1.0
)

# Test starting position
board = chess.Board()
print("\n" + "=" * 60)
print("Testing Starting Position")
print("=" * 60)
print(board)
print()

# Get evaluation
with torch.no_grad():
    policy, value, method = bot.hybrid_evaluator.evaluate(board, depth_remaining=1)
    
print(f"Evaluation method: {method}")
print(f"Value type: {type(value)}")
print(f"Value: {value}")
if hasattr(value, 'item'):
    print(f"Value.item(): {value.item()}")
print()

# Try a few different moves
test_moves = ['e2e4', 'g1h3', 'd2d4', 'g1f3']
print("Evaluating different first moves:")
for move_uci in test_moves:
    test_board = chess.Board()
    move = chess.Move.from_uci(move_uci)
    test_board.push(move)
    
    with torch.no_grad():
        policy, value, method = bot.hybrid_evaluator.evaluate(test_board, depth_remaining=1)
    
    if hasattr(value, 'item'):
        val = value.item()
    else:
        val = value
    
    print(f"  {move_uci}: {val:.3f} cp")

# Also test the search evaluation
print("\n" + "=" * 60)
print("Testing Search Evaluation")
print("=" * 60)
board = chess.Board()
score = bot.search_engine.evaluate_position(board)
print(f"evaluate_position() returned: {score}")
print(f"Type: {type(score)}")
