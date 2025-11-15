"""
Simple ELO Test - Play a few games to estimate your bot's rating
"""

import sys
from pathlib import Path
from test_elo_rating import run_elo_test

def main():
    print("\n" + "=" * 70)
    print("Simple ELO Rating Test for Hybrid Chess Bot")
    print("=" * 70)
    print()
    print("This will play 12 games (2 at each of 6 strength levels)")
    print("to estimate your bot's ELO rating.")
    print()
    print("Estimated time: 40-50 minutes")
    print()
    
    # Check Stockfish exists
    if not Path("./Stockfish/src/stockfish").exists():
        print("ERROR: Stockfish not found at ./Stockfish/src/stockfish")
        sys.exit(1)
    
    response = input("Start ELO test? (y/n): ").strip().lower()
    if response != 'y':
        print("Test cancelled")
        return
    
    print("\nStarting test...\n")
    
    # Run test with reasonable settings
    results = run_elo_test(
        num_games_per_level=2,  # 2 games per level = 12 total
        bot_config={
            'depth': 5,
            'time_limit': 5.0,
            'use_time_management': False,
            'verbose': False
        },
        save_results=True
    )
    
    if results:
        print("\n" + "=" * 70)
        print("TEST COMPLETE!")
        print("=" * 70)
        print()
        print(f"Your bot's estimated ELO: {results['estimated_elo']:.0f} ± {results['confidence']:.0f}")
        print()
        print("Check the saved files for detailed results.")

if __name__ == '__main__':
    main()
