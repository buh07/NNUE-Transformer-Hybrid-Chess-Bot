"""
Quick test: One game between Hybrid Bot and Stockfish
"""

import chess
import chess.engine
import sys
from pathlib import Path
from HybridChessBot import HybridChessBot

def quick_test():
    """Quick single game test"""
    
    print("=" * 60)
    print("Quick Test: Hybrid Bot vs Stockfish (1 game)")
    print("=" * 60)
    print()
    
    # Find Stockfish
    stockfish_path = "./Stockfish/src/stockfish"
    if not Path(stockfish_path).exists():
        print(f"ERROR: Stockfish not found at {stockfish_path}")
        sys.exit(1)
    
    print("Initializing bots...")
    
    # Create hybrid bot
    hybrid_bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=4,
        time_limit=3.0,
        verbose=True
    )
    print("✓ Hybrid bot ready\n")
    
    # Create Stockfish (weak for testing)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True})
    engine.configure({"UCI_Elo": 1350})  # Beginner level
    print("✓ Stockfish ready (ELO 1350)\n")
    
    # Play game
    board = chess.Board()
    move_count = 0
    
    print("Starting game: Hybrid (White) vs Stockfish (Black)\n")
    print(board)
    print()
    
    while not board.is_game_over() and move_count < 100:
        move_count += 1
        
        if board.turn == chess.WHITE:
            # Hybrid's turn
            print(f"\nMove {move_count} - Hybrid (White) thinking...")
            move = hybrid_bot.choose_move(board)
            print(f"Hybrid plays: {move.uci()}")
        else:
            # Stockfish's turn
            print(f"\nMove {move_count} - Stockfish (Black) thinking...")
            result = engine.play(board, chess.engine.Limit(time=1.0))
            move = result.move
            print(f"Stockfish plays: {move.uci()}")
        
        board.push(move)
        
        # Show position every 5 moves
        if move_count % 5 == 0:
            print(f"\nPosition after move {move_count}:")
            print(board)
    
    # Game result
    print(f"\n{'='*60}")
    print("Game Over!")
    print(f"{'='*60}")
    print(f"Moves: {move_count}")
    
    if board.is_checkmate():
        winner = "Hybrid" if board.turn == chess.BLACK else "Stockfish"
        print(f"Result: {winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Result: Draw by stalemate")
    elif board.is_insufficient_material():
        print("Result: Draw by insufficient material")
    else:
        print(f"Result: Game ended")
    
    print("\nFinal position:")
    print(board)
    
    # Cleanup
    engine.quit()
    
    print("\nTest complete!")


if __name__ == '__main__':
    quick_test()
