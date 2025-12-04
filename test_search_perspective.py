"""
Test to verify search perspective is correct
"""
import chess
import torch
from HybridChessBot import HybridChessBot

# Create bot with depth limit for faster testing
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    verbose=False,
    depth=3,
    time_limit=2.0
)

print("=" * 70)
print("Testing Search Perspective")
print("=" * 70)

# Test 1: Starting position as White
board = chess.Board()
print("\nTest 1: Starting position (White to move)")
print(board)
print()

move, score = bot.search_engine.iterative_deepening(board, max_depth=3, time_limit=2.0)
print(f"Best move: {move.uci()}")
print(f"Score: {score:.3f} cp (should be positive, good for White)")

# Test 2: Same position but Black to move (by making a null move)
print("\n" + "=" * 70)
print("Test 2: After 1.e4 (Black to move)")
board = chess.Board()
board.push_san("e4")
print(board)
print()

move, score = bot.search_engine.iterative_deepening(board, max_depth=3, time_limit=2.0)
print(f"Best move: {move.uci()}")
print(f"Score: {score:.3f} cp")
print("Note: Score is from White's POV. Negative = good for Black")
print(f"  Black should try to minimize, so picking move with most negative score")

# Test 3: Position where White is winning
print("\n" + "=" * 70)
print("Test 3: White up a queen (White to move)")
board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(board)
print()

move, score = bot.search_engine.iterative_deepening(board, max_depth=3, time_limit=2.0)
print(f"Best move: {move.uci()}")
print(f"Score: {score:.3f} cp (should be large positive, White is winning)")

# Test 4: Position where Black is winning  
print("\n" + "=" * 70)
print("Test 4: Black up a queen (Black to move)")
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1")
print(board)
print()

move, score = bot.search_engine.iterative_deepening(board, max_depth=3, time_limit=2.0)
print(f"Best move: {move.uci()}")
print(f"Score: {score:.3f} cp (should be large negative, Black is winning)")

# Test 5: Verify scores improve with depth
print("\n" + "=" * 70)
print("Test 5: Verify search finds better moves at deeper depths")
board = chess.Board()
print("Starting position - testing depths 1, 2, 3")
print()

for depth in [1, 2, 3]:
    move, score = bot.search_engine.iterative_deepening(board, max_depth=depth, time_limit=1.0)
    print(f"Depth {depth}: move={move.uci()}, score={score:.3f}")

print("\nNote: Scores should not wildly fluctuate. At deeper depths,")
print("      the bot should find better or equally good moves.")

print("\n" + "=" * 70)
print("Tests Complete!")
print("=" * 70)
