"""
ELO Rating System for Chess Bot
Estimates your bot's ELO by playing against Stockfish at various strength levels
"""

import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time
from HybridChessBot import HybridChessBot
from play_vs_stockfish import StockfishBot, play_game


class ELOCalculator:
    """
    Calculate ELO rating based on match results
    Uses standard ELO formula: E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    """
    
    def __init__(self, k_factor: int = 32):
        """
        Args:
            k_factor: K-factor for ELO calculation (higher = more volatile)
                     32 is standard for established players
                     40 for beginners, 24 for experts
        """
        self.k_factor = k_factor
    
    def expected_score(self, player_elo: float, opponent_elo: float) -> float:
        """
        Calculate expected score for player against opponent
        
        Returns: Expected score (0.0 to 1.0, where 0.5 = equal strength)
        """
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400))
    
    def update_rating(self, player_elo: float, opponent_elo: float, 
                     actual_score: float) -> float:
        """
        Update player's ELO rating based on game result
        
        Args:
            player_elo: Current ELO rating
            opponent_elo: Opponent's ELO rating
            actual_score: Actual score (1.0 = win, 0.5 = draw, 0.0 = loss)
        
        Returns:
            New ELO rating
        """
        expected = self.expected_score(player_elo, opponent_elo)
        new_elo = player_elo + self.k_factor * (actual_score - expected)
        return new_elo
    
    def estimate_elo_from_results(self, results: List[Dict]) -> Tuple[float, float, Dict]:
        """
        Estimate ELO from multiple game results using iterative refinement
        
        Args:
            results: List of game results with 'opponent_elo' and 'score'
        
        Returns:
            (estimated_elo, confidence_interval, detailed_stats)
        """
        if not results:
            return 1500.0, 200.0, {}
        
        # Start with initial estimate based on win rate
        total_score = sum(r['score'] for r in results)
        avg_score = total_score / len(results)
        
        # Quick estimate from average opponent ELO and score
        avg_opponent_elo = sum(r['opponent_elo'] for r in results) / len(results)
        
        # If score = 0.5, ELO = opponent ELO
        # If score > 0.5, ELO > opponent ELO (stronger)
        # If score < 0.5, ELO < opponent ELO (weaker)
        
        # Use logistic regression approach
        # score = 1 / (1 + 10^((opponent - player) / 400))
        # Solve for player: player = opponent - 400 * log10((1/score) - 1)
        
        if avg_score > 0.95:
            avg_score = 0.95  # Cap to avoid division by zero
        elif avg_score < 0.05:
            avg_score = 0.05
        
        initial_elo = avg_opponent_elo - 400 * math.log10((1/avg_score) - 1)
        
        # Iteratively refine using all results
        estimated_elo = initial_elo
        
        for iteration in range(100):  # Converge
            new_elo = estimated_elo
            
            for result in results:
                expected = self.expected_score(new_elo, result['opponent_elo'])
                error = result['score'] - expected
                new_elo += 0.1 * self.k_factor * error  # Small step
            
            # Check convergence
            if abs(new_elo - estimated_elo) < 0.1:
                break
            
            estimated_elo = new_elo
        
        # Calculate confidence interval based on sample size and variance
        # More games = higher confidence
        # More consistent results = higher confidence
        
        expected_scores = [self.expected_score(estimated_elo, r['opponent_elo']) 
                          for r in results]
        actual_scores = [r['score'] for r in results]
        
        squared_errors = [(a - e) ** 2 for a, e in zip(actual_scores, expected_scores)]
        variance = sum(squared_errors) / len(squared_errors)
        
        # Standard error of the mean
        stderr = math.sqrt(variance / len(results))
        
        # 95% confidence interval (approximately 2 standard errors)
        # Convert stderr to ELO points (rough approximation)
        confidence_interval = stderr * 400  # Scale to ELO points
        
        # Detailed statistics
        wins = sum(1 for r in results if r['score'] == 1.0)
        draws = sum(1 for r in results if r['score'] == 0.5)
        losses = sum(1 for r in results if r['score'] == 0.0)
        
        stats = {
            'estimated_elo': estimated_elo,
            'confidence_interval': confidence_interval,
            'total_games': len(results),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'score': total_score,
            'win_rate': avg_score,
            'variance': variance,
            'min_opponent': min(r['opponent_elo'] for r in results),
            'max_opponent': max(r['opponent_elo'] for r in results),
            'avg_opponent': avg_opponent_elo
        }
        
        return estimated_elo, confidence_interval, stats


def run_elo_test(num_games_per_level: int = 4, 
                 bot_config: Dict = None,
                 save_results: bool = True) -> Dict:
    """
    Run comprehensive ELO test against multiple Stockfish strength levels
    
    Args:
        num_games_per_level: Games to play at each strength level
        bot_config: Configuration for HybridChessBot
        save_results: Save results to file
    
    Returns:
        Complete test results and ELO estimate
    """
    
    print("=" * 70)
    print("ELO Rating Test for Hybrid Chess Bot")
    print("=" * 70)
    print()
    
    # Find Stockfish
    stockfish_path = "./Stockfish/src/stockfish"
    if not Path(stockfish_path).exists():
        print(f"ERROR: Stockfish not found at {stockfish_path}")
        sys.exit(1)
    
    # Default bot configuration
    if bot_config is None:
        bot_config = {
            'depth': None,
            'time_limit': 5.0,
            'use_time_management': False
        }
    
    print("Bot Configuration:")
    for key, value in bot_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Test against multiple ELO levels for accurate estimation
    # Use a range of opponents to get better statistical confidence
    test_levels = [
        1350,  # Beginner
        1500,  # Lower intermediate
        1650,  # Intermediate
        1800,  # Upper intermediate
        1950,  # Advanced
        2100,  # Expert
    ]
    
    print(f"Will test against {len(test_levels)} Stockfish strength levels")
    print(f"Games per level: {num_games_per_level}")
    print(f"Total games: {len(test_levels) * num_games_per_level}")
    print()
    
    # Estimate time
    avg_game_time = num_games_per_level * 50 * (bot_config['time_limit'] + 1.0) / 60  # minutes
    total_time = avg_game_time * len(test_levels)
    print(f"Estimated time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print()
    
    # Auto-proceed if called from script, otherwise prompt
    import sys
    if sys.stdin.isatty():
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Test cancelled")
            return None
    else:
        print("Auto-starting test...")
        print()
    
    # Create bot
    print("\nInitializing bot...")
    # Remove verbose from bot_config if present to avoid conflict
    bot_config_copy = bot_config.copy()
    bot_config_copy.pop('verbose', None)
    hybrid_bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        verbose=False,
        **bot_config_copy
    )
    print("Bot ready!\n")
    
    # Store all game results
    all_results = []
    game_number = 0
    start_time = time.time()
    
    # Test each level
    for level_idx, sf_elo in enumerate(test_levels, 1):
        print(f"\n{'#'*70}")
        print(f"Testing Level {level_idx}/{len(test_levels)}: Stockfish ELO {sf_elo}")
        print(f"{'#'*70}\n")
        
        # Create Stockfish at this level
        stockfish = StockfishBot(
            stockfish_path=stockfish_path,
            elo=sf_elo,
            time_limit=1.0
        )
        
        # Play games (alternate colors)
        level_wins = 0
        level_draws = 0
        level_losses = 0
        
        for game_idx in range(num_games_per_level):
            game_number += 1
            is_white = (game_idx % 2 == 0)
            
            print(f"Game {game_number}/{len(test_levels)*num_games_per_level}: ", end='')
            print(f"{'HybridBot (W)' if is_white else 'HybridBot (B)'} vs SF-{sf_elo}")
            
            # Play game
            if is_white:
                result = play_game(
                    white_bot=hybrid_bot,
                    black_bot=stockfish,
                    white_name="HybridBot",
                    black_name=f"SF-{sf_elo}",
                    max_moves=150,
                    verbose=False
                )
            else:
                result = play_game(
                    white_bot=stockfish,
                    black_bot=hybrid_bot,
                    white_name=f"SF-{sf_elo}",
                    black_name="HybridBot",
                    max_moves=150,
                    verbose=False
                )
            
            # Determine score from bot's perspective
            if is_white:
                if result['result'] == '1-0':
                    score = 1.0
                    level_wins += 1
                elif result['result'] == '0-1':
                    score = 0.0
                    level_losses += 1
                else:
                    score = 0.5
                    level_draws += 1
            else:
                if result['result'] == '0-1':
                    score = 1.0
                    level_wins += 1
                elif result['result'] == '1-0':
                    score = 0.0
                    level_losses += 1
                else:
                    score = 0.5
                    level_draws += 1
            
            print(f"  Result: {result['result']} - ", end='')
            print(f"{'WIN' if score == 1.0 else 'DRAW' if score == 0.5 else 'LOSS'}")
            print(f"  Moves: {result['moves']}, Termination: {result['termination']}")
            
            # Store result
            all_results.append({
                'game_number': game_number,
                'opponent_elo': sf_elo,
                'score': score,
                'result': result['result'],
                'moves': result['moves'],
                'termination': result['termination'],
                'bot_color': 'white' if is_white else 'black'
            })
        
        # Level summary
        level_score = level_wins + 0.5 * level_draws
        level_rate = level_score / num_games_per_level * 100
        
        print(f"\nLevel {sf_elo} Summary:")
        print(f"  Record: {level_wins}W - {level_losses}L - {level_draws}D")
        print(f"  Score: {level_score}/{num_games_per_level} ({level_rate:.1f}%)")
        
        # Current ELO estimate
        if len(all_results) >= 4:
            calc = ELOCalculator()
            current_elo, ci, stats = calc.estimate_elo_from_results(all_results)
            print(f"  Current ELO estimate: {current_elo:.0f} ± {ci:.0f}")
        
        stockfish.quit()
    
    # Final ELO calculation
    print(f"\n{'='*70}")
    print("CALCULATING FINAL ELO RATING")
    print(f"{'='*70}\n")
    
    calc = ELOCalculator(k_factor=32)
    estimated_elo, confidence, stats = calc.estimate_elo_from_results(all_results)
    
    elapsed_time = time.time() - start_time
    
    # Print detailed results
    print(f"Total Games Played: {stats['total_games']}")
    print(f"Overall Record: {stats['wins']}W - {stats['losses']}L - {stats['draws']}D")
    print(f"Total Score: {stats['score']:.1f}/{stats['total_games']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print()
    print(f"Opponent Range: ELO {stats['min_opponent']:.0f} - {stats['max_opponent']:.0f}")
    print(f"Average Opponent: ELO {stats['avg_opponent']:.0f}")
    print()
    print(f"{'='*70}")
    print(f"ESTIMATED ELO: {estimated_elo:.0f} ± {confidence:.0f}")
    print(f"{'='*70}")
    print()
    print(f"Rating Range: {estimated_elo - confidence:.0f} - {estimated_elo + confidence:.0f}")
    print()
    
    # Rating interpretation
    print("Rating Classification:")
    if estimated_elo < 1200:
        print("  Novice (< 1200)")
    elif estimated_elo < 1400:
        print("  Beginner (1200-1400)")
    elif estimated_elo < 1600:
        print("  Intermediate (1400-1600)")
    elif estimated_elo < 1800:
        print("  Advanced (1600-1800)")
    elif estimated_elo < 2000:
        print("  Expert (1800-2000)")
    elif estimated_elo < 2200:
        print("  Master (2000-2200)")
    else:
        print("  Grandmaster level (2200+)")
    
    print()
    print(f"Confidence: ", end='')
    if confidence < 50:
        print("Very High (±{:.0f})".format(confidence))
    elif confidence < 100:
        print("High (±{:.0f})".format(confidence))
    elif confidence < 150:
        print("Moderate (±{:.0f})".format(confidence))
    else:
        print("Low (±{:.0f}) - Consider more games".format(confidence))
    
    print()
    print(f"Test Duration: {elapsed_time/60:.1f} minutes")
    
    # Save results
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"elo_test_results_{timestamp}.json"
        
        output = {
            'timestamp': timestamp,
            'bot_config': bot_config,
            'estimated_elo': estimated_elo,
            'confidence_interval': confidence,
            'statistics': stats,
            'all_games': all_results,
            'test_duration_minutes': elapsed_time / 60
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Also save readable summary
        summary_file = f"elo_test_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("ELO Rating Test Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Duration: {elapsed_time/60:.1f} minutes\n\n")
            
            f.write("Bot Configuration:\n")
            for key, value in bot_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Results:\n")
            f.write(f"  Games Played: {stats['total_games']}\n")
            f.write(f"  Record: {stats['wins']}W - {stats['losses']}L - {stats['draws']}D\n")
            f.write(f"  Win Rate: {stats['win_rate']*100:.1f}%\n\n")
            
            f.write("ELO Rating:\n")
            f.write(f"  Estimated ELO: {estimated_elo:.0f} ± {confidence:.0f}\n")
            f.write(f"  Rating Range: {estimated_elo - confidence:.0f} - {estimated_elo + confidence:.0f}\n\n")
            
            f.write("Opponents Faced:\n")
            f.write(f"  Range: ELO {stats['min_opponent']:.0f} - {stats['max_opponent']:.0f}\n")
            f.write(f"  Average: ELO {stats['avg_opponent']:.0f}\n\n")
            
            f.write("Game-by-Game Results:\n")
            for r in all_results:
                f.write(f"  Game {r['game_number']}: vs SF-{r['opponent_elo']} ({r['bot_color']}) - ")
                f.write(f"{'WIN' if r['score'] == 1.0 else 'DRAW' if r['score'] == 0.5 else 'LOSS'} ")
                f.write(f"({r['moves']} moves)\n")
        
        print(f"Summary saved to: {summary_file}")
    
    return {
        'estimated_elo': estimated_elo,
        'confidence': confidence,
        'stats': stats,
        'all_results': all_results
    }


def quick_elo_estimate(num_games: int = 10) -> float:
    """
    Quick ELO estimate with fewer games
    Less accurate but faster
    """
    print("=" * 70)
    print("Quick ELO Estimate (10 games)")
    print("=" * 70)
    print()
    print("This will play 10 games against varied opponents for a rough estimate.")
    print("For accurate rating, use the full test (24+ games).")
    print()
    
    return run_elo_test(
        num_games_per_level=2,  # 2 games at each of 5 levels = 10 total
        bot_config={'depth': 4, 'time_limit': 3.0},
        save_results=True
    )


def main():
    """Main entry point"""
    print("\nHybrid Chess Bot - ELO Rating Test")
    print("=" * 70)
    print()
    print("Options:")
    print("1. Quick estimate (10 games, ~30 min)")
    print("2. Standard test (24 games, ~90 min)")
    print("3. Comprehensive test (36 games, ~2 hours)")
    print("4. Custom configuration")
    print()
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == '1':
        quick_elo_estimate(num_games=10)
    elif choice == '2':
        run_elo_test(
            num_games_per_level=4,
            bot_config={'depth': 5, 'time_limit': 5.0}
        )
    elif choice == '3':
        run_elo_test(
            num_games_per_level=6,
            bot_config={'depth': 6, 'time_limit': 8.0}
        )
    else:
        # Custom
        print("\nCustom Configuration:")
        games_per_level = int(input("Games per strength level (2-10): ") or 4)
        depth = int(input("Bot search depth (3-8): ") or 5)
        time_limit = float(input("Time per move (1-30s): ") or 5.0)
        use_tm = input("Use time management? (y/n): ").strip().lower() == 'y'
        
        bot_config = {
            'depth': depth,
            'time_limit': time_limit,
            'use_time_management': use_tm
        }
        
        if use_tm:
            total_time = float(input("Total game time (60-600s): ") or 300.0)
            bot_config['total_game_time'] = total_time
        
        run_elo_test(
            num_games_per_level=games_per_level,
            bot_config=bot_config
        )


if __name__ == '__main__':
    main()
