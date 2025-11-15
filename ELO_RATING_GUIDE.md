# How to Get Your Bot's ELO Rating

## Overview

Getting an accurate ELO rating requires playing multiple games against opponents with known ratings and using statistical analysis. This guide shows you how to determine your bot's actual chess strength.

## Quick Start

### Option 1: Quick Estimate (30 minutes)
```bash
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate
python test_elo_rating.py
# Choose option 1
```
- **Games:** 10 (2 per level across 5 levels)
- **Time:** ~30 minutes
- **Accuracy:** ±100-150 ELO

### Option 2: Standard Test (90 minutes) - **Recommended**
```bash
python test_elo_rating.py
# Choose option 2
```
- **Games:** 24 (4 per level across 6 levels)
- **Time:** ~90 minutes
- **Accuracy:** ±50-100 ELO

### Option 3: Comprehensive Test (2+ hours)
```bash
python test_elo_rating.py
# Choose option 3
```
- **Games:** 36 (6 per level across 6 levels)
- **Time:** ~2 hours
- **Accuracy:** ±30-50 ELO

## How It Works

### ELO Rating System

The system uses the standard chess ELO formula:

```
Expected Score = 1 / (1 + 10^((Opponent_ELO - Player_ELO) / 400))
```

**Example:**
- If you're ELO 1600 playing ELO 1600: Expected score = 0.50 (50%)
- If you're ELO 1600 playing ELO 1800: Expected score = 0.24 (24%)
- If you're ELO 1800 playing ELO 1600: Expected score = 0.76 (76%)

### Rating Calculation Process

1. **Play against multiple opponents** at different ELO levels (1350-2100)
2. **Record results** (win = 1.0, draw = 0.5, loss = 0.0)
3. **Calculate expected scores** for each match
4. **Use iterative refinement** to find ELO that best fits results
5. **Compute confidence interval** based on consistency and sample size

### Test Opponents

The test plays against Stockfish at these ELO levels:

| Level | ELO | Strength |
|-------|-----|----------|
| 1 | 1350 | Beginner |
| 2 | 1500 | Lower Intermediate |
| 3 | 1650 | Intermediate |
| 4 | 1800 | Upper Intermediate |
| 5 | 1950 | Advanced |
| 6 | 2100 | Expert |

### Why Multiple Levels?

Testing against a range of opponents gives more accurate results:
- **Too weak opponents:** Can't distinguish between 1800 and 2000 if both beat 1400 100%
- **Too strong opponents:** Can't distinguish between 1200 and 1400 if both lose to 2000
- **Range of opponents:** Provides statistical power to pinpoint exact rating

## Understanding Results

### Output Format

```
ESTIMATED ELO: 1650 ± 75
Rating Range: 1575 - 1725

Total Games: 24
Record: 12W - 8L - 4D
Win Rate: 58.3%
```

### What It Means

**Estimated ELO: 1650**
- Your bot plays at approximately 1650 strength

**± 75 (Confidence Interval)**
- 95% confident true ELO is between 1575-1725
- Smaller interval = higher confidence
- More games = smaller interval

**Win Rate: 58.3%**
- Against opponents averaging ~1600 ELO
- 50% = equal strength, >50% = stronger, <50% = weaker

### Confidence Levels

| Confidence | Meaning |
|------------|---------|
| ±30-50 | Very reliable (36+ games) |
| ±50-100 | Reliable (24+ games) |
| ±100-150 | Rough estimate (10-20 games) |
| ±150+ | Uncertain (< 10 games) |

## Rating Classifications

| ELO Range | Classification |
|-----------|----------------|
| < 1200 | Novice |
| 1200-1400 | Beginner |
| 1400-1600 | Intermediate |
| 1600-1800 | Advanced |
| 1800-2000 | Expert |
| 2000-2200 | Master |
| 2200+ | Grandmaster level |

## Example Results

### Example 1: Strong Intermediate Bot
```
Results against Stockfish:
  SF-1350: 4-0 (100%)
  SF-1500: 3-1 (75%)
  SF-1650: 2-2 (50%)
  SF-1800: 1-3 (25%)
  SF-1950: 0-4 (0%)

Estimated ELO: 1650 ± 60
Classification: Advanced
```

**Interpretation:** Bot is clearly stronger than 1500, competitive with 1650, weaker than 1800.

### Example 2: Beginner Bot
```
Results against Stockfish:
  SF-1350: 2-2 (50%)
  SF-1500: 1-3 (25%)
  SF-1650: 0-4 (0%)

Estimated ELO: 1350 ± 80
Classification: Beginner
```

**Interpretation:** Bot struggles against 1500+, competitive at 1350 level.

## Factors Affecting Rating

### Bot Configuration

**Search Depth:**
- Depth 3-4: ~1300-1500 ELO
- Depth 5-6: ~1500-1700 ELO
- Depth 7-8: ~1700-1900 ELO (slower)

**Time per Move:**
- 1-2s: Lower strength (less calculation)
- 3-5s: Balanced
- 10-30s: Higher strength (more calculation)

**Time Management:**
- Fixed time: Consistent but suboptimal
- Dynamic allocation: +50-100 ELO improvement

### Architecture Factors

**Your Hybrid System:**
- ✅ Selector accuracy (97.56% = good routing)
- ✅ NNUE for tactics (fast, accurate for sharp positions)
- ✅ Transformer for strategy (deep understanding)
- ⚠️ Search speed (~400-600 NPS, slower than pure NNUE)

**Expected Performance:**
- Strong tactical awareness (NNUE)
- Good positional understanding (Transformer)
- Limited by search depth due to speed

## Running Custom Tests

### Test Specific Configuration

```python
from test_elo_rating import run_elo_test

# Test with your preferred settings
results = run_elo_test(
    num_games_per_level=4,
    bot_config={
        'depth': 6,
        'time_limit': 10.0,
        'use_time_management': True,
        'total_game_time': 300.0
    },
    save_results=True
)

print(f"Estimated ELO: {results['estimated_elo']:.0f}")
```

### Compare Configurations

```python
# Test with different depths
configs = [
    {'depth': 4, 'time_limit': 3.0},
    {'depth': 5, 'time_limit': 5.0},
    {'depth': 6, 'time_limit': 10.0}
]

for config in configs:
    results = run_elo_test(num_games_per_level=3, bot_config=config)
    print(f"Depth {config['depth']}: ELO {results['estimated_elo']:.0f}")
```

## Improving Accuracy

### More Games = Better Accuracy

| Games | Expected Confidence | Time Required |
|-------|---------------------|---------------|
| 10 | ±100-150 | 30 min |
| 20 | ±75-100 | 60 min |
| 30 | ±50-75 | 90 min |
| 50 | ±30-50 | 2.5 hours |

### Tips for Accurate Testing

1. **Use even number of games per level** (alternate colors)
2. **Test against range of opponents** (not just one level)
3. **Consistent bot configuration** (don't change settings mid-test)
4. **Let games finish naturally** (don't interrupt)
5. **Review suspicious games** (crashes, timeouts)

## Interpreting Edge Cases

### Very High Win Rate (>90%)
```
All opponents too weak → Test against stronger opponents
Your bot might be 200+ ELO above highest tested
```

### Very Low Win Rate (<10%)
```
All opponents too strong → Test against weaker opponents  
Your bot might be 200+ ELO below lowest tested
```

### Inconsistent Results
```
High variance (large confidence interval)
Possible causes:
- Bot performance varies by position type
- Search time too variable
- Bugs causing occasional blunders

Solution: More games or investigate bot consistency
```

## Output Files

The test generates two files:

### 1. JSON Results (`elo_test_results_TIMESTAMP.json`)
```json
{
  "estimated_elo": 1650,
  "confidence_interval": 75,
  "statistics": {
    "total_games": 24,
    "wins": 12,
    "losses": 8,
    "draws": 4,
    "win_rate": 0.583
  },
  "all_games": [...]
}
```

### 2. Text Summary (`elo_test_summary_TIMESTAMP.txt`)
```
ELO Rating Test Results
======================================================================

Estimated ELO: 1650 ± 75
Rating Range: 1575 - 1725

Games Played: 24
Record: 12W - 8L - 4D
Win Rate: 58.3%

Game-by-Game Results:
  Game 1: vs SF-1350 (white) - WIN (42 moves)
  Game 2: vs SF-1350 (black) - WIN (38 moves)
  ...
```

## Validating Results

### Cross-Check Methods

1. **Play on chess servers** (lichess, chess.com) to get independent rating
2. **Compare win rates** against known bots
3. **Test against other engines** (not just Stockfish)
4. **Repeat test** after some time to check consistency

### Expected Correlations

If your bot is ELO 1650:
- Should beat 1400 opponents ~85%
- Should be even with 1650 opponents ~50%
- Should lose to 1900 opponents ~25%

Use this to validate the estimate makes sense.

## Advanced: ELO Calculator Usage

### Calculate ELO from Custom Results

```python
from test_elo_rating import ELOCalculator

calc = ELOCalculator(k_factor=32)

# Your game results
results = [
    {'opponent_elo': 1500, 'score': 1.0},  # Win
    {'opponent_elo': 1500, 'score': 0.5},  # Draw
    {'opponent_elo': 1650, 'score': 0.0},  # Loss
    {'opponent_elo': 1650, 'score': 0.5},  # Draw
]

elo, confidence, stats = calc.estimate_elo_from_results(results)
print(f"Estimated ELO: {elo:.0f} ± {confidence:.0f}")
```

### Expected Score Calculator

```python
# What's my expected score against 1700 opponent?
expected = calc.expected_score(player_elo=1600, opponent_elo=1700)
print(f"Expected: {expected:.2%}")  # e.g., 36%

# How many points should I score in 10 games?
print(f"Expected score: {expected * 10:.1f}/10")  # e.g., 3.6/10
```

## Troubleshooting

### Test Takes Too Long
```python
# Use quick estimate
python test_elo_rating.py
# Choose option 1 (10 games)

# Or reduce time per move
bot_config = {'depth': 4, 'time_limit': 2.0}
```

### Bot Crashes During Test
```
Check logs for errors
Reduce depth or time limit
Ensure checkpoint is valid
```

### Unexpected Rating
```
Review individual games for patterns
Check if bot has consistent performance
Verify Stockfish is working correctly
Consider hardware limitations (CPU/GPU)
```

## Recommendations

### For First Test
1. Run **Standard Test** (option 2)
2. Takes ~90 minutes
3. Gives ±50-100 ELO accuracy
4. Good baseline for future comparisons

### For Development
1. Quick estimates (option 1) to track progress
2. Standard test after major changes
3. Comprehensive test for final evaluation

### For Publication/Comparison
1. Comprehensive test (option 3)
2. Multiple runs for consistency
3. Document exact configuration
4. Save all game records

## Example Session

```bash
$ cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
$ source chess_env/bin/activate
$ python test_elo_rating.py

Hybrid Chess Bot - ELO Rating Test
======================================================================

Options:
1. Quick estimate (10 games, ~30 min)
2. Standard test (24 games, ~90 min)
3. Comprehensive test (36 games, ~2 hours)
4. Custom configuration

Choose option (1-4): 2

Bot Configuration:
  depth: 5
  time_limit: 5.0
  use_time_management: False

Will test against 6 Stockfish strength levels
Games per level: 4
Total games: 24

Estimated time: 90.0 minutes (1.5 hours)

Continue? (y/n): y

[... games play ...]

======================================================================
ESTIMATED ELO: 1650 ± 75
======================================================================

Rating Range: 1575 - 1725
Classification: Advanced
Confidence: High (±75)

Results saved to: elo_test_results_20251115_163045.json
Summary saved to: elo_test_summary_20251115_163045.txt
```

## Summary

**To get your bot's ELO rating:**

1. ✅ **Run:** `python test_elo_rating.py`
2. ✅ **Choose:** Standard test (option 2) - 90 minutes
3. ✅ **Wait:** Test plays 24 games against 6 strength levels
4. ✅ **Get:** Estimated ELO ± confidence interval
5. ✅ **Review:** Detailed results in saved files

**The test automatically:**
- Plays against multiple Stockfish levels
- Alternates colors for fairness
- Calculates ELO using standard formula
- Provides confidence intervals
- Saves complete results

**You get:**
- Accurate ELO rating (±50-100 points)
- Performance breakdown by opponent strength
- Game-by-game results
- Statistical confidence metrics
- Rating classification

Ready to test! 🎯
