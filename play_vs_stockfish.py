"""
Play Hybrid Bot vs Stockfish
Test the hybrid bot's performance against Stockfish at various strength levels
"""

import chess
import chess.engine
import time
import sys
import os
import random
import json
from pathlib import Path
from HybridChessBot import HybridChessBot


class StockfishBot:
    """Wrapper for Stockfish engine"""
    
    def __init__(self, stockfish_path: str, elo: int = None, depth: int = None, 
                 time_limit: float = 1.0):
        """
        Initialize Stockfish bot
        
        Args:
            stockfish_path: Path to Stockfish executable
            elo: ELO rating (1350-3190, limits strength)
            depth: Fixed depth (alternative to time limit)
            time_limit: Time limit per move in seconds
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.elo = elo
        self.depth = depth
        self.time_limit = time_limit
        
        # Configure engine
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True})
            self.engine.configure({"UCI_Elo": elo})
            print(f"Stockfish configured to ELO {elo}")
        
        # Get engine info
        print(f"Stockfish: {self.engine.id['name']}")
    
    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose best move"""
        if self.depth is not None:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        else:
            result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move
    
    def quit(self):
        """Close engine"""
        self.engine.quit()


def play_game(white_bot, black_bot, white_name: str, black_name: str,
              max_moves: int = 200, verbose: bool = True) -> dict:
    """
    Play a game between two bots
    
    Args:
        white_bot: Bot playing white (must have choose_move method)
        black_bot: Bot playing black
        white_name: Name for white bot
        black_name: Name for black bot
        max_moves: Maximum moves before draw
        verbose: Print move-by-move output
    
    Returns:
        Game result dictionary
    """
    board = chess.Board()
    move_times = {'white': [], 'black': []}
    move_count = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{white_name} (White) vs {black_name} (Black)")
        print(f"{'='*60}\n")
        print(board)
        print()
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        
        # Determine current player
        if board.turn == chess.WHITE:
            current_bot = white_bot
            bot_name = white_name
            color = 'white'
        else:
            current_bot = black_bot
            bot_name = black_name
            color = 'black'
        
        if verbose:
            print(f"Move {move_count} - {bot_name} to move...")
        
        # Get move with timing
        start_time = time.time()
        try:
            move = current_bot.choose_move(board)
            elapsed = time.time() - start_time
            move_times[color].append(elapsed)

            # Fallback if bot returned None
            if move is None:
                legal = list(board.legal_moves)
                if legal:
                    move = random.choice(legal)
                    if verbose:
                        print(f"  WARNING: {bot_name} returned None; falling back to legal move {move.uci()}")
                else:
                    raise RuntimeError("No legal moves available to fall back to")

            if verbose:
                print(f"  {bot_name} plays: {move.uci()} ({elapsed:.2f}s)")

            # Make move
            board.push(move)

            if verbose and move_count % 10 == 0:
                print(f"\nPosition after move {move_count}:")
                print(board)
                print()

        except Exception as e:
            print(f"ERROR: {bot_name} failed to choose move: {e}")
            # Award loss to player who errored
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            # Ensure result dict includes player names to avoid KeyError later
            return {
                'result': result,
                'termination': 'error',
                'moves': move_count,
                'error': str(e),
                'white_name': white_name,
                'black_name': black_name
            }
    
    # Determine result
    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        termination = 'checkmate'
    elif board.is_stalemate():
        result = "1/2-1/2"
        termination = 'stalemate'
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        termination = 'insufficient_material'
    elif board.is_fifty_moves():
        result = "1/2-1/2"
        termination = '50_move_rule'
    elif board.is_repetition():
        result = "1/2-1/2"
        termination = 'repetition'
    elif move_count >= max_moves:
        result = "1/2-1/2"
        termination = 'max_moves'
    else:
        result = "1/2-1/2"
        termination = 'unknown'
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game Over: {result}")
        print(f"Termination: {termination}")
        print(f"Total moves: {move_count}")
        print(f"{'='*60}\n")
        print("Final position:")
        print(board)
        print()
    
    return {
        'result': result,
        'termination': termination,
        'moves': move_count,
        'white_name': white_name,
        'black_name': black_name,
        'white_avg_time': sum(move_times['white']) / len(move_times['white']) if move_times['white'] else 0,
        'black_avg_time': sum(move_times['black']) / len(move_times['black']) if move_times['black'] else 0,
        'pgn': board.root().variation_san(board.move_stack)
    }


def play_match(hybrid_bot, stockfish_bot, 
               num_games: int = 2,
               hybrid_name: str = "HybridBot",
               stockfish_name: str = "Stockfish") -> dict:
    """
    Play a match (multiple games with alternating colors)
    
    Args:
        hybrid_bot: Your hybrid bot
        stockfish_bot: Stockfish bot
        num_games: Number of games (should be even for fairness)
        hybrid_name: Name for hybrid bot
        stockfish_name: Name for Stockfish
    
    Returns:
        Match statistics
    """
    results = []
    hybrid_score = 0
    stockfish_score = 0
    
    print(f"\n{'='*60}")
    print(f"MATCH: {hybrid_name} vs {stockfish_name}")
    print(f"Games: {num_games}")
    print(f"{'='*60}\n")
    
    for game_num in range(num_games):
        print(f"\n{'#'*60}")
        print(f"Game {game_num + 1}/{num_games}")
        print(f"{'#'*60}")
        
        # Alternate colors
        if game_num % 2 == 0:
            # Hybrid plays white
            result = play_game(
                white_bot=hybrid_bot,
                black_bot=stockfish_bot,
                white_name=hybrid_name,
                black_name=stockfish_name,
                verbose=True
            )
            
            # Score from hybrid's perspective
            if result['result'] == '1-0':
                hybrid_score += 1
            elif result['result'] == '0-1':
                stockfish_score += 1
            else:
                hybrid_score += 0.5
                stockfish_score += 0.5
        else:
            # Hybrid plays black
            result = play_game(
                white_bot=stockfish_bot,
                black_bot=hybrid_bot,
                white_name=stockfish_name,
                black_name=hybrid_name,
                verbose=True
            )
            
            # Score from hybrid's perspective
            if result['result'] == '0-1':
                hybrid_score += 1
            elif result['result'] == '1-0':
                stockfish_score += 1
            else:
                hybrid_score += 0.5
                stockfish_score += 0.5
        
        results.append(result)
        
        print(f"\nCurrent Score: {hybrid_name} {hybrid_score} - {stockfish_score} {stockfish_name}\n")
    
    # Print match summary
    print(f"\n{'='*60}")
    print(f"MATCH RESULTS")
    print(f"{'='*60}")
    print(f"{hybrid_name}: {hybrid_score}/{num_games}")
    print(f"{stockfish_name}: {stockfish_score}/{num_games}")
    print(f"Draws: {num_games - int(hybrid_score + stockfish_score)}")
    
    wins = sum(1 for r in results if (r['result'] == '1-0' and r['white_name'] == hybrid_name) or 
                                     (r['result'] == '0-1' and r['black_name'] == hybrid_name))
    losses = sum(1 for r in results if (r['result'] == '0-1' and r['white_name'] == hybrid_name) or 
                                       (r['result'] == '1-0' and r['black_name'] == hybrid_name))
    draws = num_games - wins - losses
    
    print(f"\n{hybrid_name} record: {wins}W - {losses}L - {draws}D")
    print(f"Win rate: {hybrid_score/num_games*100:.1f}%")
    
    # Average game length
    avg_moves = sum(r['moves'] for r in results) / len(results)
    print(f"Average game length: {avg_moves:.1f} moves")
    
    return {
        'hybrid_score': hybrid_score,
        'stockfish_score': stockfish_score,
        'games': results,
        'wins': wins,
        'losses': losses,
        'draws': draws
    }


def main():
    """Main function to run bot vs Stockfish"""
    
    # Find Stockfish executable
    stockfish_paths = [
        "./Stockfish/src/stockfish",
        "./Stockfish/stockfish",
        "stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish"
    ]
    
    stockfish_path = None
    for path in stockfish_paths:
        if Path(path).exists():
            stockfish_path = path
            break
    
    if stockfish_path is None:
        print("ERROR: Stockfish not found!")
        print("Please ensure Stockfish is installed at one of these locations:")
        for path in stockfish_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"Found Stockfish at: {stockfish_path}\n")
    
    # Configuration
    print("=" * 60)
    print("Hybrid Bot vs Stockfish Match")
    print("=" * 60)
    print()
    print("Configuration options:")
    print("1. Quick test (2 games, Stockfish depth 5)")
    print("2. Fair match (4 games, Stockfish depth 8)")
    print("3. Stockfish ELO 1500 (4 games)")
    print("4. Stockfish ELO 2000 (4 games)")
    print("5. Custom")
    print()
    
    # Non-interactive auto mode: pass 'auto' as a command-line argument or set AUTO_RUN=1
    auto = ('auto' in sys.argv) or (os.environ.get('AUTO_RUN') == '1')
    if auto:
        # Quick test by default in auto mode
        choice = '1'
        print("Auto mode enabled: selecting quick test (1)")
    else:
        choice = input("Choose configuration (1-5): ").strip()
    
    if choice == '1':
        num_games = 2
        sf_elo = None
        sf_depth = 5
        sf_time = None
        hybrid_depth = 5
        hybrid_time = 5.0
    elif choice == '2':
        num_games = 4
        sf_elo = None
        sf_depth = 8
        sf_time = None
        hybrid_depth = 6
        hybrid_time = 10.0
    elif choice == '3':
        num_games = 4
        sf_elo = 1500
        sf_depth = None
        sf_time = 1.0
        hybrid_depth = 5
        hybrid_time = 5.0
    elif choice == '4':
        num_games = 4
        sf_elo = 2000
        sf_depth = None
        sf_time = 1.0
        hybrid_depth = 6
        hybrid_time = 5.0
    else:
        num_games = int(input("Number of games: "))
        sf_elo_input = input("Stockfish ELO (leave empty for full strength): ")
        sf_elo = int(sf_elo_input) if sf_elo_input else None
        sf_depth = int(input("Stockfish depth (0 for time-based): ") or 0) or None
        sf_time = float(input("Stockfish time per move (s): ") or 1.0)
        hybrid_depth = int(input("Hybrid bot depth: ") or 5)
        hybrid_time = float(input("Hybrid bot time per move (s): ") or 5.0)
    
    print(f"\n{'='*60}")
    print("Match Configuration:")
    print(f"  Games: {num_games}")
    print(f"  Stockfish: {'ELO ' + str(sf_elo) if sf_elo else 'Full strength'}")
    print(f"  Stockfish: {'Depth ' + str(sf_depth) if sf_depth else f'Time {sf_time}s'}")
    print(f"  Hybrid: Depth {hybrid_depth}, Time {hybrid_time}s")
    print(f"{'='*60}\n")
    
    if not auto:
        input("Press Enter to start match...")
    else:
        print("Starting match immediately (auto mode)...")
    
    # Create bots
    print("\nInitializing bots...")
    
    hybrid_bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=hybrid_depth,
        time_limit=hybrid_time,
        verbose=False  # Set to True for detailed output
    )
    print("✓ Hybrid bot ready")
    
    stockfish_bot = StockfishBot(
        stockfish_path=stockfish_path,
        elo=sf_elo,
        depth=sf_depth,
        time_limit=sf_time
    )
    print("✓ Stockfish ready\n")
    
    # Play match
    try:
        match_results = play_match(
            hybrid_bot=hybrid_bot,
            stockfish_bot=stockfish_bot,
            num_games=num_games,
            hybrid_name="HybridBot",
            stockfish_name=f"Stockfish{' ELO'+str(sf_elo) if sf_elo else ''}"
        )
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"match_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Match: HybridBot vs Stockfish\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Games: {num_games}\n")
            f.write(f"Result: {match_results['hybrid_score']} - {match_results['stockfish_score']}\n")
            f.write(f"Record: {match_results['wins']}W - {match_results['losses']}L - {match_results['draws']}D\n")
            f.write(f"\nGame details:\n")
            for i, game in enumerate(match_results['games'], 1):
                f.write(f"\nGame {i}:\n")
                f.write(f"  Result: {game['result']}\n")
                f.write(f"  Moves: {game['moves']}\n")
                f.write(f"  Termination: {game['termination']}\n")
        
        # Also collect and save hybrid bot instrumentation/statistics when available
        try:
            bot_stats = hybrid_bot.get_statistics()
            stats_file = results_file.replace('.txt', '_stats.json')
            with open(stats_file, 'w') as sf:
                json.dump(bot_stats, sf, indent=2, default=str)
            print(f"\nResults saved to: {results_file}")
            print(f"Instrumentation stats saved to: {stats_file}")
        except Exception as e:
            # Non-fatal: just report and continue
            print(f"\nResults saved to: {results_file}")
            print(f"Warning: failed to save instrumentation stats: {e}")
        
    finally:
        # Cleanup
        stockfish_bot.quit()
        print("\nMatch complete!")


if __name__ == '__main__':
    main()
