"""
Automatic ELO Test - No prompts, runs immediately
"""

import sys
from pathlib import Path
from test_elo_rating import run_elo_test

# Check Stockfish exists
if not Path("./Stockfish/src/stockfish").exists():
    print("ERROR: Stockfish not found at ./Stockfish/src/stockfish")
    sys.exit(1)

print("\n" + "=" * 70)
print("Automatic ELO Rating Test for Hybrid Chess Bot")
print("=" * 70)
print()
print("Configuration:")
print("  Games: 12 (2 at each of 6 strength levels)")
print("  Opponents: Stockfish ELO 1350, 1500, 1650, 1800, 1950, 2100")
print("  Bot: Depth 5, Time 5.0s per move")
print("  Estimated time: 40-50 minutes")
print()
print("Starting test automatically...\n")

# Run test with reasonable settings
# Temporarily override stdin to bypass confirmation prompt
import io
sys.stdin = io.StringIO('y\n')

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
