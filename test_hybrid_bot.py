"""
Test script for HybridChessBot
Tests compatibility with ChessGame.py and plays sample games
"""

import chess
from ChessGame import ChessGame
from HybridChessBot import HybridChessBot, create_hybrid_bot


class RandomBot:
    """Simple random bot for testing."""
    def __init__(self):
        import random
        self.random = random
    
    def choose_move(self, board):
        moves = list(board.legal_moves)
        return self.random.choice(moves)


def test_interface():
    """Test that HybridChessBot implements the required interface."""
    print("=" * 60)
    print("TEST 1: Interface Compatibility")
    print("=" * 60)
    
    # Create bot
    bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=3, verbose=False)
    
    # Test choose_move method exists
    assert hasattr(bot, 'choose_move'), "Bot must have choose_move method"
    
    # Test it returns a valid move
    board = chess.Board()
    move = bot.choose_move(board)
    
    assert isinstance(move, chess.Move), "choose_move must return chess.Move"
    assert move in board.legal_moves, "Returned move must be legal"
    
    print("✓ Bot has required choose_move() method")
    print("✓ Returns valid chess.Move object")
    print("✓ Move is legal")
    print("\nInterface test PASSED!\n")


def test_game_integration():
    """Test that bot works with ChessGame class."""
    print("=" * 60)
    print("TEST 2: ChessGame Integration")
    print("=" * 60)
    
    # Create bots
    hybrid_bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=3, verbose=False)
    random_bot = RandomBot()
    
    # Create game (hybrid plays White, random plays Black)
    game = ChessGame(hybrid_bot, random_bot)
    
    print("Created game: HybridBot (White) vs RandomBot (Black)")
    print("\nPlaying 5 moves...\n")
    
    # Play 5 moves
    for i in range(10):  # 10 half-moves = 5 full moves
        if game.is_game_over():
            print("Game ended early")
            break
        
        print(f"Move {i//2 + 1}:", "White" if i % 2 == 0 else "Black")
        game.make_move()
        
        # Show last move
        last_move = game.board.peek()
        print(f"  Played: {last_move.uci()}")
    
    print("\nFinal position:")
    print(game)
    print("ChessGame integration test PASSED!\n")


def test_full_game():
    """Play a complete game."""
    print("=" * 60)
    print("TEST 3: Complete Game")
    print("=" * 60)
    
    # Create bots with different settings
    hybrid_bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=4,
        time_limit=3.0,
        verbose=True
    )
    random_bot = RandomBot()
    
    # Create game
    game = ChessGame(hybrid_bot, random_bot)
    
    print("Playing complete game: HybridBot vs RandomBot")
    print("(This may take a few minutes...)\n")
    
    move_count = 0
    max_moves = 100  # Limit to prevent infinite games
    
    while not game.is_game_over() and move_count < max_moves:
        print(f"\n{'='*40}")
        print(f"Move {move_count//2 + 1}: {'White' if move_count % 2 == 0 else 'Black'}")
        print(f"{'='*40}")
        
        game.make_move()
        move_count += 1
        
        last_move = game.board.peek()
        print(f"Played: {last_move.uci()}")
    
    # Show result
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    print(game)
    
    result = game.board.result()
    print(f"Result: {result}")
    
    if game.board.is_checkmate():
        winner = "Black" if game.board.turn else "White"
        print(f"Checkmate! {winner} wins!")
    elif game.board.is_stalemate():
        print("Stalemate!")
    elif game.board.is_insufficient_material():
        print("Draw by insufficient material")
    elif move_count >= max_moves:
        print(f"Game stopped after {max_moves} moves")
    
    # Print statistics
    print("\n" + "="*60)
    print("Bot Statistics")
    print("="*60)
    stats = hybrid_bot.get_statistics()
    
    print(f"\nSearch Statistics:")
    print(f"  Total nodes: {stats['search']['nodes_searched']:,}")
    print(f"  Average NPS: {stats['search']['nps']:,.0f}")
    
    print(f"\nEvaluation Statistics:")
    total_evals = stats['evaluation']['total_evals']
    nnue_evals = stats['evaluation']['nnue_only_evals']
    hybrid_evals = stats['evaluation']['hybrid_evals']
    
    print(f"  Total evaluations: {total_evals:,}")
    print(f"  NNUE-only: {nnue_evals:,} ({100*nnue_evals/total_evals:.1f}%)")
    print(f"  With Transformer: {hybrid_evals:,} ({100*hybrid_evals/total_evals:.1f}%)")
    
    print("\nComplete game test PASSED!\n")


def test_analysis():
    """Test position analysis feature."""
    print("=" * 60)
    print("TEST 4: Position Analysis")
    print("=" * 60)
    
    bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=5, verbose=False)
    
    # Analyze starting position
    board = chess.Board()
    print("Analyzing starting position...")
    
    analysis = bot.analyze_position(board, depth=5)
    
    print(f"\nBest move: {analysis['best_move']}")
    print(f"Evaluation: {analysis['score']:.2f} centipawns")
    print(f"Depth: {analysis['depth']}")
    print(f"Nodes searched: {analysis['nodes']:,}")
    print(f"Time: {analysis['time']:.2f}s")
    print(f"Nodes per second: {analysis['nps']:,.0f}")
    
    print("\nAnalysis test PASSED!\n")


if __name__ == '__main__':
    """Run all tests."""
    
    print("\n" + "=" * 60)
    print("HYBRID CHESS BOT TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        test_interface()
        test_game_integration()
        test_analysis()
        
        # Optional: Play full game (can be slow)
        response = input("Play a complete game? This may take several minutes. (y/n): ")
        if response.lower() == 'y':
            test_full_game()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe HybridChessBot is fully compatible with ChessGame.py")
        print("and ready to play!\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
