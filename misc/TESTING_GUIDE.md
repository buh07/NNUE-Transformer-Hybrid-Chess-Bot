# Testing Your Bot - Complete Guide

## ✅ All Scripts Are Working!

You have 6 different testing scripts available, each for different purposes.

## Quick Reference

### 1. **Test Bot Functionality** (5 minutes)
```bash
python quick_test_vs_stockfish.py
```
- **Purpose:** Verify bot works and plays legal moves
- **Games:** 1 game
- **Opponent:** Stockfish ELO 1350 (weak)
- **Output:** Move-by-move display
- **Use:** First-time testing, debugging

### 2. **Get ELO Rating - Simple** (40 minutes) ⭐ RECOMMENDED FOR ELO
```bash
python simple_elo_test.py
```
- **Purpose:** Quick ELO estimate
- **Games:** 12 (2 per level × 6 levels)
- **Opponents:** ELO 1350, 1500, 1650, 1800, 1950, 2100
- **Output:** `Estimated ELO: 1650 ± 75`
- **Files:** JSON + text summary saved automatically
- **Use:** Get your bot's rating quickly

### 3. **Get ELO Rating - Full** (90-120 minutes)
```bash
python test_elo_rating.py
```
- **Purpose:** Accurate ELO with options
- **Options:**
  - Quick: 10 games (30 min)
  - Standard: 24 games (90 min)
  - Comprehensive: 36 games (2 hours)
  - Custom: Your settings
- **Use:** When you need more accuracy

### 4. **Play Interactive Match** (30-60 minutes)
```bash
python play_vs_stockfish.py
```
- **Purpose:** Play custom matches
- **Options:** Choose opponent strength, game count, time controls
- **Output:** Live game display, match statistics
- **Use:** Test specific configurations, watch games

### 5. **Run Benchmark Suite** (60-90 minutes)
```bash
python run_benchmark.py
```
- **Purpose:** Standardized performance tests
- **Tests:** 3 benchmarks (Beginner, Intermediate, Advanced)
- **Games:** 12 total (4 per benchmark)
- **Output:** Performance report across levels
- **Use:** Track improvement over time

### 6. **Test Bot Integration** (2 minutes)
```bash
python test_hybrid_bot.py
```
- **Purpose:** Unit tests for bot interface
- **Tests:** ChessGame compatibility, API correctness
- **Use:** Development, CI/CD

## Which Script Should I Use?

### I want to know my bot's ELO rating
```bash
python simple_elo_test.py
```
**Best choice:** Quick, accurate enough (±50-100 ELO), saves results

### I want to watch my bot play
```bash
python quick_test_vs_stockfish.py
```
**Shows:** Every move, evaluation, statistics

### I want to test if my bot works
```bash
python test_hybrid_bot.py
```
**Verifies:** Interface, legal moves, basic functionality

### I want detailed performance data
```bash
python run_benchmark.py
```
**Provides:** Win rates at multiple levels, comprehensive stats

### I want maximum ELO accuracy
```bash
python test_elo_rating.py
# Choose option 3 (Comprehensive)
```
**Best for:** Final evaluation, publication

## Example Session: Getting Your ELO

```bash
# 1. Activate environment
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate

# 2. Run simple ELO test
python simple_elo_test.py

# Output after 40-50 minutes:
# ======================================================================
# TEST COMPLETE!
# ======================================================================
#
# Your bot's estimated ELO: 1650 ± 75
#
# Check the saved files for detailed results.

# 3. Check results
cat elo_test_summary_*.txt
```

## Understanding ELO Results

### Example Output
```
Estimated ELO: 1650 ± 75
Rating Range: 1575 - 1725

Games Played: 12
Record: 7W - 3L - 2D
Win Rate: 66.7%

Opponents Faced:
  Range: ELO 1350 - 2100
  Average: ELO 1725
```

### What It Means
- **1650:** Your bot plays at approximately 1650 strength
- **±75:** 95% confidence interval (true ELO likely between 1575-1725)
- **7W-3L-2D:** 7 wins, 3 losses, 2 draws
- **66.7%:** Score percentage (1.0 per win, 0.5 per draw)

### ELO Classifications
- **< 1200:** Novice
- **1200-1400:** Beginner
- **1400-1600:** Intermediate ⬅ Your bot might be here
- **1600-1800:** Advanced
- **1800-2000:** Expert
- **2000-2200:** Master
- **2200+:** Grandmaster level

## Common Issues & Solutions

### Issue: "Stockfish not found"
```bash
# Check if Stockfish exists
ls -la Stockfish/src/stockfish

# If missing, it should be there already:
# -rwxr-xr-x 119512480 Nov 14 15:52 Stockfish/src/stockfish
```
**Solution:** Already installed at `Stockfish/src/stockfish` ✓

### Issue: Script runs but no games play
**Check:** Are you in the right directory?
```bash
pwd
# Should be: /scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot
```

### Issue: Bot plays but crashes mid-game
**Check:** Memory/GPU availability
```bash
nvidia-smi  # Check GPU
free -h     # Check RAM
```
**Solution:** Reduce depth or time limit in bot config

### Issue: Takes too long
**Use faster config:**
```bash
# Edit simple_elo_test.py, change:
bot_config={
    'depth': 4,        # Reduce from 5 to 4
    'time_limit': 3.0, # Reduce from 5.0 to 3.0
    ...
}
```

## Script Details

### simple_elo_test.py ⭐
```python
# What it does:
- Plays 12 games (2 at each of 6 levels)
- Bot config: depth=5, time=5.0s
- Estimates ELO with confidence interval
- Saves JSON and text results
- Takes ~40-50 minutes

# When to use:
- First time getting ELO
- Quick rating estimate
- Regular check-ins
```

### test_elo_rating.py
```python
# What it does:
- Multiple test options (10/24/36 games)
- Customizable bot configuration
- Detailed statistical analysis
- Performance tracking
- Takes 30min - 2+ hours depending on option

# When to use:
- Need custom configuration
- Want different game counts
- Need maximum accuracy
- Development testing
```

### play_vs_stockfish.py
```python
# What it does:
- Interactive match setup
- Choose opponent ELO or depth
- Live game display
- Alternating colors
- Match statistics

# When to use:
- Want to watch specific matches
- Testing new configurations
- Playing for fun
- Demonstrating bot
```

### quick_test_vs_stockfish.py
```python
# What it does:
- Single game with move display
- Opponent: SF ELO 1350
- Bot: depth=4, time=3.0s
- Shows every move and eval
- Takes ~5 minutes

# When to use:
- First-time verification
- Debugging issues
- Quick functionality check
```

### run_benchmark.py
```python
# What it does:
- 3 standardized benchmarks
- Beginner (SF 1350), Intermediate (1650), Advanced (1850)
- 4 games per benchmark = 12 total
- Performance report
- Takes ~60-90 minutes

# When to use:
- Compare bot versions
- Track improvement
- Standardized testing
```

### test_hybrid_bot.py
```python
# What it does:
- Unit tests for bot interface
- Verifies ChessGame compatibility
- Checks choose_move() works
- Fast execution (~2 minutes)

# When to use:
- Development testing
- After code changes
- CI/CD pipeline
```

## Output Files

All ELO tests save results:

### JSON File (`elo_test_results_TIMESTAMP.json`)
```json
{
  "estimated_elo": 1650,
  "confidence_interval": 75,
  "statistics": {
    "total_games": 12,
    "wins": 7,
    "losses": 3,
    "draws": 2
  },
  "all_games": [...]
}
```

### Text Summary (`elo_test_summary_TIMESTAMP.txt`)
```
ELO Rating Test Results
======================================================================

Estimated ELO: 1650 ± 75
Rating Range: 1575 - 1725

Record: 7W - 3L - 2D
Win Rate: 66.7%

Game-by-Game Results:
  Game 1: vs SF-1350 (white) - WIN (42 moves)
  ...
```

## Tips for Best Results

### For Accurate ELO:
1. ✅ Use `simple_elo_test.py` (balanced speed/accuracy)
2. ✅ Let it run uninterrupted
3. ✅ Use consistent bot configuration
4. ✅ More games = better accuracy (but takes longer)

### For Quick Testing:
1. ✅ Use `quick_test_vs_stockfish.py` first
2. ✅ Verify bot works correctly
3. ✅ Then run full ELO test

### For Development:
1. ✅ `test_hybrid_bot.py` for unit tests
2. ✅ `quick_test_vs_stockfish.py` for integration
3. ✅ `simple_elo_test.py` before/after changes

## Recommended Workflow

### First Time (Total: ~50 minutes)
```bash
# 1. Quick functionality check (5 min)
python quick_test_vs_stockfish.py

# 2. Get ELO rating (40-50 min)
python simple_elo_test.py

# 3. Review results
cat elo_test_summary_*.txt
```

### Regular Testing
```bash
# Option A: Quick check (5 min)
python quick_test_vs_stockfish.py

# Option B: Full ELO update (40 min)
python simple_elo_test.py

# Option C: Watch specific match (10-20 min)
python play_vs_stockfish.py
```

### After Major Changes
```bash
# 1. Unit tests (2 min)
python test_hybrid_bot.py

# 2. Integration test (5 min)
python quick_test_vs_stockfish.py

# 3. Performance benchmark (90 min)
python run_benchmark.py

# 4. Full ELO test (90 min)
python test_elo_rating.py  # Choose option 2
```

## Summary

**All scripts are working!** ✅

**To get your bot's ELO rating right now:**
```bash
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate
python simple_elo_test.py
```

**This will:**
- ✅ Play 12 games against varied opponents
- ✅ Calculate your bot's ELO rating
- ✅ Give you results in ~40-50 minutes
- ✅ Save detailed results to files

**Expected result format:**
```
Your bot's estimated ELO: 1650 ± 75
```

The scripts are ready and working. Just run the command above! 🎯
