"""
Play chess games using the trained hybrid NNUE-Transformer model.
"""

import chess
import chess.pgn
import torch
import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.models.hybrid_evaluator import HybridEvaluator
from src.search import AlphaBetaSearch


def load_model(checkpoint_path: str, device: str = 'cuda') -> HybridEvaluator:
    """
    Load the trained hybrid model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded HybridEvaluator
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create evaluator
    evaluator = HybridEvaluator(device=device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    evaluator.projection.load_state_dict(checkpoint['projection_state_dict'])
    evaluator.selector.load_state_dict(checkpoint['selector_state_dict'])
    
    evaluator.projection.eval()
    evaluator.selector.eval()
    
    print(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    
    return evaluator


def play_game(white_search: AlphaBetaSearch, black_search: AlphaBetaSearch,
              max_moves: int = 200, verbose: bool = True) -> chess.pgn.Game:
    """
    Play a game between two search engines.
    
    Args:
        white_search: AlphaBetaSearch for white
        black_search: AlphaBetaSearch for black
        max_moves: Maximum number of moves before draw
        verbose: Print moves as they're played
    
    Returns:
        chess.pgn.Game object
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Hybrid Model Game"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Hybrid-NNUE-Transformer"
    game.headers["Black"] = "Hybrid-NNUE-Transformer"
    
    node = game
    move_count = 0
    
    if verbose:
        print("\nStarting game...")
        print("=" * 60)
        print(board)
        print("=" * 60)
    
    while not board.is_game_over() and move_count < max_moves:
        # Select search engine for current player
        search_engine = white_search if board.turn else black_search
        
        # Get best move
        if verbose:
            player = "White" if board.turn else "Black"
            print(f"\n{player} to move...")
        
        move = search_engine.get_best_move(board, time_limit=5.0)
        
        # Make move
        board.push(move)
        node = node.add_variation(move)
        move_count += 1
        
        if verbose:
            print(f"{player} plays: {move}")
            print(board)
            print("-" * 60)
    
    # Set result
    result = board.result()
    game.headers["Result"] = result
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Game over: {result}")
        if board.is_checkmate():
            print("Checkmate!")
        elif board.is_stalemate():
            print("Stalemate")
        elif board.is_insufficient_material():
            print("Insufficient material")
        elif board.can_claim_draw():
            print("Draw by repetition or 50-move rule")
        print("=" * 60)
    
    return game


def play_against_human(search_engine: AlphaBetaSearch, human_color: str = 'white',
                       verbose: bool = True):
    """
    Play a game against a human player.
    
    Args:
        search_engine: AlphaBetaSearch for the engine
        human_color: 'white' or 'black'
        verbose: Print extra information
    """
    board = chess.Board()
    human_is_white = (human_color.lower() == 'white')
    
    print("\n" + "=" * 60)
    print("Playing against Hybrid NNUE-Transformer Chess Engine")
    print(f"You are playing as {'White' if human_is_white else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("Type 'quit' to exit")
    print("=" * 60)
    print(board)
    print("=" * 60)
    
    while not board.is_game_over():
        if board.turn == human_is_white:
            # Human's turn
            print(f"\nYour turn ({'White' if board.turn else 'Black'})")
            print(f"Legal moves: {', '.join([str(m) for m in list(board.legal_moves)[:10]])}...")
            
            while True:
                move_str = input("Enter your move: ").strip()
                
                if move_str.lower() == 'quit':
                    print("Game ended by player")
                    return
                
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again.")
                except:
                    print("Invalid move format! Use UCI notation (e.g., e2e4)")
        else:
            # Engine's turn
            print(f"\nEngine thinking ({'White' if board.turn else 'Black'})...")
            move = search_engine.get_best_move(board, time_limit=10.0)
            board.push(move)
            print(f"Engine plays: {move}")
        
        print("\n" + board.unicode() + "\n")
        print("-" * 60)
    
    # Game over
    print("\n" + "=" * 60)
    print(f"Game over: {board.result()}")
    if board.is_checkmate():
        winner = "You win!" if board.turn != human_is_white else "Engine wins!"
        print(f"Checkmate! {winner}")
    elif board.is_stalemate():
        print("Stalemate - Draw")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.can_claim_draw():
        print("Draw by repetition or 50-move rule")
    print("=" * 60)


def analyze_position(search_engine: AlphaBetaSearch, fen: str, depth: int = 6):
    """
    Analyze a specific position.
    
    Args:
        search_engine: AlphaBetaSearch engine
        fen: FEN string of position to analyze
        depth: Search depth
    """
    board = chess.Board(fen)
    
    print("\n" + "=" * 60)
    print("Position Analysis")
    print("=" * 60)
    print(board)
    print("=" * 60)
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print("=" * 60)
    
    # Get best move
    print(f"\nAnalyzing to depth {depth}...")
    best_move = search_engine.get_best_move(board, depth=depth)
    
    print(f"\nBest move: {best_move}")
    
    # Show resulting position
    board.push(best_move)
    print("\nResulting position:")
    print(board)


def main():
    parser = argparse.ArgumentParser(description='Play chess with Hybrid NNUE-Transformer model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['self', 'human', 'analyze'],
                       default='human', help='Game mode')
    parser.add_argument('--depth', type=int, default=5,
                       help='Search depth')
    parser.add_argument('--color', type=str, choices=['white', 'black'],
                       default='white', help='Human color (for human mode)')
    parser.add_argument('--fen', type=str, default=None,
                       help='FEN position to analyze (for analyze mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PGN file for self-play games')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pt'):
                    print(f"  - checkpoints/{f}")
        else:
            print("  No checkpoints directory found")
        return
    
    evaluator = load_model(args.checkpoint, device=args.device)
    
    # Create search engine
    search_engine = AlphaBetaSearch(
        evaluator,
        max_depth=args.depth,
        tt_size=1000000,
        use_quiescence=True
    )
    
    # Run selected mode
    if args.mode == 'self':
        # Self-play
        print("\n" + "=" * 60)
        print("Self-play mode")
        print("=" * 60)
        
        game = play_game(search_engine, search_engine, verbose=True)
        
        # Save game if output specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(str(game))
            print(f"\nGame saved to {args.output}")
        else:
            print("\nGame PGN:")
            print(game)
    
    elif args.mode == 'human':
        # Play against human
        play_against_human(search_engine, human_color=args.color, verbose=True)
    
    elif args.mode == 'analyze':
        # Analyze position
        if args.fen is None:
            args.fen = chess.STARTING_FEN
            print("No FEN specified, analyzing starting position")
        
        analyze_position(search_engine, args.fen, depth=args.depth)


if __name__ == '__main__':
    main()
