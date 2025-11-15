#!/usr/bin/env python3
"""
Quick start example for HybridChessBot
Demonstrates basic usage with ChessGame
"""

import chess
from ChessGame import ChessGame
from HybridChessBot import HybridChessBot


class SimpleBot:
    """A very simple bot that picks the first legal move."""
    def choose_move(self, board):
        return list(board.legal_moves)[0]


def play_quick_game():
    """Play a quick game between HybridBot and SimpleBot."""
    
    print("=" * 60)
    print("HYBRID CHESS BOT - QUICK START DEMO")
    print("=" * 60)
    
    # Create the hybrid bot
    print("\nInitializing Hybrid Bot...")
    hybrid_bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',  # Trained weights
        depth=4,                                   # Search depth
        time_limit=3.0,                           # 3 seconds per move
        verbose=True                              # Print statistics
    )
    
    # Create a simple opponent
    simple_bot = SimpleBot()
    
    # Create the game (hybrid plays White, simple plays Black)
    game = ChessGame(hybrid_bot, simple_bot)
    
    print("\n" + "=" * 60)
    print("GAME START: HybridBot (White) vs SimpleBot (Black)")
    print("=" * 60)
    
    # Play 10 moves (5 full moves)
    for move_num in range(10):
        if game.is_game_over():
            break
        
        current_player = "White" if move_num % 2 == 0 else "Black"
        print(f"\n{'='*60}")
        print(f"Move {move_num//2 + 1}: {current_player} to move")
        print(f"{'='*60}")
        
        # Make the move
        game.make_move()
        
        # Show the move played
        last_move = game.board.peek()
        print(f"\n{current_player} played: {last_move.uci()}")
        
        # Show board
        print("\n" + str(game))
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    # Show final statistics
    stats = hybrid_bot.get_statistics()
    print("\nHybrid Bot Statistics:")
    print(f"  Total nodes searched: {stats['search']['nodes_searched']:,}")
    print(f"  NNUE evaluations: {stats['evaluation']['nnue_only_evals']:,}")
    print(f"  Transformer evaluations: {stats['evaluation']['hybrid_evals']:,}")
    
    total = stats['evaluation']['total_evals']
    if total > 0:
        transformer_pct = 100 * stats['evaluation']['hybrid_evals'] / total
        print(f"  Transformer usage: {transformer_pct:.1f}%")


def analyze_position():
    """Analyze a specific position."""
    
    print("\n" + "=" * 60)
    print("POSITION ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Create bot
    bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=6,
        verbose=False
    )
    
    # Analyze starting position
    board = chess.Board()
    print("\nAnalyzing starting position:")
    print(board)
    
    print("\nSearching to depth 6...")
    analysis = bot.analyze_position(board, depth=6)
    
    print(f"\nResults:")
    print(f"  Best move: {analysis['best_move']}")
    print(f"  Evaluation: {analysis['score']:.2f} centipawns")
    print(f"  Nodes: {analysis['nodes']:,}")
    print(f"  Time: {analysis['time']:.2f}s")
    print(f"  Speed: {analysis['nps']:,.0f} nodes/second")


def interactive_mode():
    """Play against the bot interactively."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Play against HybridBot!")
    print("=" * 60)
    
    # Create bot
    bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=5,
        time_limit=5.0,
        verbose=True
    )
    
    board = chess.Board()
    
    print("\nYou are White. Enter moves in UCI format (e.g., e2e4)")
    print("Type 'quit' to exit\n")
    
    while not board.is_game_over():
        # Show board
        print("\n" + "=" * 60)
        print(board)
        print("=" * 60)
        
        if board.turn:  # White (human)
            print("\nYour move:")
            move_str = input("> ").strip().lower()
            
            if move_str == 'quit':
                print("Thanks for playing!")
                return
            
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except:
                print("Invalid move format! Use UCI notation (e.g., e2e4)")
                continue
        
        else:  # Black (bot)
            print("\nBot thinking...")
            move = bot.choose_move(board)
            print(f"Bot plays: {move.uci()}")
            board.push(move)
    
    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    print(board)
    print(f"\nResult: {board.result()}")


if __name__ == '__main__':
    """Main menu."""
    
    print("\n" + "=" * 60)
    print("HYBRID CHESS BOT - Quick Start")
    print("=" * 60)
    print("\nWhat would you like to do?")
    print("1. Watch a quick demo game")
    print("2. Analyze a position")
    print("3. Play against the bot (interactive)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        play_quick_game()
    elif choice == '2':
        analyze_position()
    elif choice == '3':
        interactive_mode()
    else:
        print("Goodbye!")
