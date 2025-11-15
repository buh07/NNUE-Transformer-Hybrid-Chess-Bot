# Playing Against Stockfish - Guide

## Overview

Test your Hybrid Bot against Stockfish at various strength levels to evaluate performance.

## Quick Start

### 1. Quick Test (Single Game)
```bash
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate
python quick_test_vs_stockfish.py
```

This runs a single game:
- **Hybrid Bot (White)**: Depth 4, 3s per move
- **Stockfish (Black)**: ELO 1350 (beginner level)

### 2. Full Match (Multiple Games)
```bash
python play_vs_stockfish.py
```

Interactive menu with options:
1. Quick test (2 games, Stockfish depth 5)
2. Fair match (4 games, Stockfish depth 8)
3. Stockfish ELO 1500 (4 games)
4. Stockfish ELO 2000 (4 games)
5. Custom configuration

## Configuration Options

### Stockfish Strength Levels

#### By ELO Rating
```python
# Beginner: 1350 ELO
stockfish_bot = StockfishBot(stockfish_path, elo=1350)

# Intermediate: 1500-1800 ELO
stockfish_bot = StockfishBot(stockfish_path, elo=1650)

# Advanced: 1900-2200 ELO
stockfish_bot = StockfishBot(stockfish_path, elo=2000)

# Expert: 2300+ ELO
stockfish_bot = StockfishBot(stockfish_path, elo=2400)
```

#### By Search Depth
```python
# Weak (fast)
stockfish_bot = StockfishBot(stockfish_path, depth=5)

# Medium
stockfish_bot = StockfishBot(stockfish_path, depth=10)

# Strong
stockfish_bot = StockfishBot(stockfish_path, depth=15)

# Very strong (slow)
stockfish_bot = StockfishBot(stockfish_path, depth=20)
```

### Hybrid Bot Configuration

```python
hybrid_bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=5,              # Search depth (3-8 reasonable)
    time_limit=5.0,       # Time per move (seconds)
    verbose=True          # Print move details
)
```

## Example Configurations

### 1. Testing Basic Functionality
```python
# Hybrid: Fast search
# Stockfish: Weak opponent
hybrid = HybridChessBot(depth=3, time_limit=2.0)
stockfish = StockfishBot(stockfish_path, elo=1350)
```

### 2. Competitive Match
```python
# Hybrid: Strong search
# Stockfish: Intermediate opponent
hybrid = HybridChessBot(depth=6, time_limit=10.0)
stockfish = StockfishBot(stockfish_path, elo=1800)
```

### 3. Strength Test
```python
# Hybrid: Best effort
# Stockfish: Strong opponent
hybrid = HybridChessBot(depth=7, time_limit=15.0)
stockfish = StockfishBot(stockfish_path, elo=2200)
```

## Match Formats

### Single Game
```python
from play_vs_stockfish import play_game

result = play_game(
    white_bot=hybrid_bot,
    black_bot=stockfish_bot,
    white_name="Hybrid",
    black_name="Stockfish",
    max_moves=200,
    verbose=True
)

print(f"Result: {result['result']}")
print(f"Moves: {result['moves']}")
```

### Match with Alternating Colors
```python
from play_vs_stockfish import play_match

match_results = play_match(
    hybrid_bot=hybrid_bot,
    stockfish_bot=stockfish_bot,
    num_games=4,  # Even number for fairness
    hybrid_name="HybridBot",
    stockfish_name="Stockfish"
)

print(f"Score: {match_results['hybrid_score']} - {match_results['stockfish_score']}")
```

## Understanding Results

### Game Result Format
```python
{
    'result': '1-0',           # 1-0 (white wins), 0-1 (black wins), 1/2-1/2 (draw)
    'termination': 'checkmate', # checkmate, stalemate, repetition, etc.
    'moves': 42,               # Total moves played
    'white_name': 'HybridBot',
    'black_name': 'Stockfish',
    'white_avg_time': 3.2,     # Average time per move
    'black_avg_time': 1.0,
    'pgn': '1. e4 e5 2. Nf3...' # Game notation
}
```

### Match Statistics
```python
{
    'hybrid_score': 2.5,       # Points scored (1 per win, 0.5 per draw)
    'stockfish_score': 1.5,
    'wins': 2,                 # Hybrid wins
    'losses': 1,               # Hybrid losses
    'draws': 1,
    'games': [...]             # List of individual game results
}
```

## Performance Benchmarks

### Expected Performance vs Stockfish

| Opponent | Hybrid Depth | Hybrid Time | Expected Result |
|----------|--------------|-------------|-----------------|
| SF ELO 1350 | 3-4 | 2-3s | Should win most games |
| SF ELO 1500 | 4-5 | 3-5s | Competitive, slight edge |
| SF ELO 1800 | 5-6 | 5-10s | Competitive match |
| SF ELO 2000 | 6-7 | 10-15s | Challenging opponent |
| SF Depth 10 | 5-6 | 5-10s | Even match |
| SF Depth 15 | 7-8 | 15-30s | Tough match |

### Current Test Results

From initial test game:
```
Hybrid Bot (White, Depth 4, 3s/move) vs Stockfish (Black, ELO 1350)
- Hybrid playing reasonable moves (d4, Nd2, e4)
- Consistent evaluation (~12 cp advantage)
- 0% transformer usage in opening (correct - tactical)
- Avg ~400-600 NPS at depth 2-4
```

## Running Experiments

### Experiment 1: Find Optimal Depth
```bash
# Test different depths against same opponent
for depth in 3 4 5 6; do
    echo "Testing depth $depth"
    python -c "
from play_vs_stockfish import *
hybrid = HybridChessBot(depth=$depth, time_limit=10.0)
sf = StockfishBot('./Stockfish/src/stockfish', elo=1650)
result = play_game(hybrid, sf, 'Hybrid', 'Stockfish', verbose=False)
print(f'Depth $depth: {result[\"result\"]}')
sf.quit()
"
done
```

### Experiment 2: Calibrate Strength
```bash
# Find which Stockfish ELO matches your bot
python play_vs_stockfish.py
# Try configurations 1-4 and see where you get ~50% win rate
```

### Experiment 3: Time Management Impact
```python
# Compare with and without time management
bot_fixed = HybridChessBot(depth=5, time_limit=5.0, use_time_management=False)
bot_dynamic = HybridChessBot(depth=5, use_time_management=True, total_game_time=300.0)

# Play matches with each and compare win rates
```

## Advanced Usage

### Custom Match Script
```python
import chess
from HybridChessBot import HybridChessBot
from play_vs_stockfish import StockfishBot, play_match

# Create bots
hybrid = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=6,
    time_limit=10.0,
    use_time_management=True,
    total_game_time=300.0,
    verbose=True
)

stockfish = StockfishBot(
    stockfish_path='./Stockfish/src/stockfish',
    elo=1800,
    time_limit=2.0
)

# Play 10-game match
results = play_match(
    hybrid_bot=hybrid,
    stockfish_bot=stockfish,
    num_games=10,
    hybrid_name="HybridBot-v1",
    stockfish_name="SF-1800"
)

# Analyze results
print(f"\nFinal Score: {results['hybrid_score']} - {results['stockfish_score']}")
print(f"Win Rate: {results['hybrid_score']/10*100:.1f}%")

# Cleanup
stockfish.quit()
```

### Collect Statistics
```python
# Track performance metrics
games_won = []
avg_moves = []
transformer_usage = []

for game_num in range(10):
    result = play_game(hybrid_bot, stockfish_bot, ...)
    
    # Track metrics
    games_won.append(result['result'] == '1-0')  # assuming hybrid is white
    avg_moves.append(result['moves'])
    
    # Get bot statistics
    stats = hybrid_bot.get_statistics()
    eval_stats = stats['evaluation']
    total_evals = eval_stats['total_evals']
    transformer_pct = eval_stats['hybrid_evals'] / total_evals if total_evals > 0 else 0
    transformer_usage.append(transformer_pct)

print(f"Win rate: {sum(games_won)/len(games_won)*100:.1f}%")
print(f"Avg game length: {sum(avg_moves)/len(avg_moves):.1f} moves")
print(f"Avg transformer usage: {sum(transformer_usage)/len(transformer_usage)*100:.1f}%")
```

## Troubleshooting

### Stockfish Not Found
```bash
# Check if Stockfish exists
ls -la Stockfish/src/stockfish

# If not, build it:
cd Stockfish/src
make build ARCH=x86-64-modern
```

### Bot Too Slow
```python
# Reduce search depth or time limit
hybrid = HybridChessBot(depth=3, time_limit=2.0)

# Or use shallower Stockfish
stockfish = StockfishBot(stockfish_path, depth=5)
```

### Memory Issues
```python
# Clear transposition table between games
hybrid_bot.search_engine.tt.clear()
hybrid_bot.reset_statistics()
```

### Games Too Long
```python
# Set maximum move limit
result = play_game(
    white_bot, black_bot,
    max_moves=100  # Draw after 100 moves
)
```

## Interpreting Results

### What to Look For

**Positive Signs:**
- ✅ Legal moves played
- ✅ Reasonable opening play
- ✅ Tactical awareness (captures good pieces)
- ✅ Wins against weak opponents (ELO 1350-1500)
- ✅ Competitive against intermediate (ELO 1500-1800)
- ✅ Selector working (transformer usage in strategic positions)

**Areas for Improvement:**
- ⚠️ Blunders (hanging pieces)
- ⚠️ Passive play (not developing)
- ⚠️ Missing obvious tactics
- ⚠️ Poor endgame technique
- ⚠️ Time management issues

### Strength Estimation

Based on match results, estimate your bot's ELO:

| Performance vs Stockfish | Estimated Bot Strength |
|--------------------------|------------------------|
| Beats ELO 1350 (80%+) | ~1400-1500 |
| Beats ELO 1500 (60%+) | ~1600-1700 |
| Even with ELO 1800 (50%) | ~1800 |
| Beats ELO 2000 (60%+) | ~2100+ |

## Next Steps

1. **Baseline Test**: Run quick_test_vs_stockfish.py to verify everything works
2. **Find Your Level**: Play matches against different ELO levels
3. **Optimize Configuration**: Tune depth/time for best performance
4. **Collect Data**: Run multiple matches for statistical significance
5. **Analyze Weaknesses**: Review lost games to find improvement areas

## Example Session

```bash
# Start environment
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate

# Quick test
python quick_test_vs_stockfish.py

# Full match
python play_vs_stockfish.py
# Choose option 3 (Stockfish ELO 1500)

# Review results
cat match_results_*.txt
```

Results are saved to `match_results_TIMESTAMP.txt` with full details.

## Files

- **`play_vs_stockfish.py`** - Main match script (interactive)
- **`quick_test_vs_stockfish.py`** - Single game test (automated)
- **`STOCKFISH_TESTING.md`** - This guide
- **`match_results_*.txt`** - Saved match results

Happy testing! 🎯
