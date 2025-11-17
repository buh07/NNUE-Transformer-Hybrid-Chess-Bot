# Time Management System - Documentation

## Overview

The time management system dynamically allocates time per move based on **position complexity metrics** derived from your hybrid architecture. This gives your bot a significant advantage by spending more time on critical positions and less on simple ones.

## Key Insight

**Your hybrid architecture already computes complexity metrics!** The selector network learns to distinguish tactical vs strategic positions, and these same metrics can guide time allocation:

- **Selector Probability (0-1)**: How strategic/complex the position is
- **Selector Confidence (0-1)**: How certain the model is (uncertainty → more time)
- **20+ Position Features**: Material, mobility, phase, tactics, structure, etc.

## Available Metrics

### 1. **Selector-Based Metrics** (Unique to Your Architecture)

| Metric | Range | Meaning | Use for Time |
|--------|-------|---------|--------------|
| `selector_probability` | 0.0 - 1.0 | 1.0 = strategic (transformer), 0.0 = tactical (NNUE) | High = more time |
| `selector_confidence` | 0.0 - 1.0 | 1.0 = very certain, 0.0 = uncertain | Low = more time |
| `tactical_complexity` | 0.0 - 1.0 | Checks, captures, threats | High = more time |
| `strategic_complexity` | 0.0 - 1.0 | Positional factors | High = more time |

### 2. **Position Features** (from `extract_selection_features`)

The selector uses 20 features that are also useful for time management:

```python
Features extracted for each position:
[0]  move_count          # Game phase indicator
[1]  material_balance    # Imbalance → complexity
[2]  num_legal_moves     # More options → more time
[3]  is_check            # Forcing → needs calculation
[4]  num_attackers       # Tactical complexity
[5]  num_defenders       # Tactical complexity
[6]  center_control      # Strategic importance
[7]  mobility            # Piece activity
[8]  king_safety         # Critical metric
[9]  piece_activity      # Coordination
[10] pawn_structure      # Long-term factors
[11] is_endgame          # Phase indicator
[12] has_passed_pawns    # Winning chances
[13] king_tropism        # Attack potential
[14] depth_remaining     # Search state
[15] num_pieces          # Complexity indicator
[16] bishops_vs_knights  # Positional factor
[17] queen_on_board      # Tactical sharpness
[18] rooks_on_open_files # Strategic advantage
[19] connected_rooks     # Coordination
```

### 3. **Derived Metrics**

The TimeManager computes additional metrics:

- **Overall Complexity**: Weighted combination of tactical, strategic, and selector metrics
- **Material Imbalance**: `|material_balance|` (unbalanced = harder to evaluate)
- **Phase Multiplier**: Opening (0.7x), Middlegame (1.2x), Endgame (1.0x)
- **Time Pressure**: Reduces allocation when running low on time

## Time Allocation Formula

```python
allocated_time = base_time × complexity × uncertainty × phase × pressure

Where:
  base_time = (remaining_time - emergency_reserve) / estimated_moves + increment * 0.8
  complexity = 0.5 + overall_complexity * 1.5  # Range: 0.5 to 2.0
  uncertainty = 1.0 + (1.0 - selector_confidence) * 0.5  # Range: 1.0 to 1.5
  phase = {0.7 (opening), 1.2 (middlegame), 1.0 (endgame)}
  pressure = {1.0 (plenty), 0.85 (medium), 0.7 (low), 0.5 (very low)}
```

### Example Calculations

**Simple Opening Position** (e4 e5 Nf3):
- Selector: 0.2 (tactical) → complexity_mult = 0.8
- Confidence: 0.9 (certain) → uncertainty_mult = 1.05
- Phase: opening → phase_mult = 0.7
- Pressure: 1.0
- **Result: 0.59x base time** (e.g., 0.6s if base = 1.0s)

**Complex Middlegame** (closed center, maneuvering):
- Selector: 0.8 (strategic) → complexity_mult = 1.7
- Confidence: 0.4 (uncertain) → uncertainty_mult = 1.3
- Phase: middlegame → phase_mult = 1.2
- Pressure: 1.0
- **Result: 2.65x base time** (e.g., 2.7s if base = 1.0s)

**Sharp Tactical Position** (unclear sacrifice):
- Selector: 0.5 (uncertain) → complexity_mult = 1.25
- Confidence: 0.1 (very uncertain) → uncertainty_mult = 1.45
- Phase: middlegame → phase_mult = 1.2
- Pressure: 1.0
- **Result: 2.18x base time** (e.g., 2.2s if base = 1.0s)

**Ratio: 2.7s / 0.6s = 4.5x more time for complex positions!**

## Usage

### Basic Usage (Enabled)

```python
from HybridChessBot import HybridChessBot

# Create bot with time management
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=5,
    use_time_management=True,  # Enable dynamic allocation
    total_game_time=300.0,      # 5 minutes total
    verbose=True
)

# The bot now automatically adjusts time per move!
move = bot.choose_move(board)
```

### Advanced Usage (Custom Parameters)

```python
from time_manager import TimeManager

# Create custom time manager
time_mgr = TimeManager(
    total_time=300.0,        # Total time (seconds)
    increment=2.0,           # Increment per move
    moves_to_go=None,        # Auto-estimate moves remaining
    emergency_time=5.0       # Keep 5s reserve
)

# Get allocation for current position
allocation = time_mgr.allocate_time(
    board=board,
    selector_model=bot.selector,
    depth_remaining=10
)

print(f"Allocated: {allocation['allocated_time']:.2f}s")
print(f"Complexity: {allocation['complexity_score']:.2f}")
print(f"Confidence: {allocation['selector_confidence']:.2f}")
```

### Without Time Management (Fixed Time)

```python
# Traditional fixed time per move
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=5,
    time_limit=3.0,          # Fixed 3 seconds per move
    use_time_management=False  # Disabled (default)
)
```

## API Reference

### TimeManager Class

```python
class TimeManager:
    def __init__(self, 
                 total_time: float,
                 increment: float = 0.0,
                 moves_to_go: Optional[int] = None,
                 emergency_time: float = 5.0)
```

**Methods:**

#### `allocate_time(board, selector_model, depth_remaining) -> Dict`

Returns allocation dictionary:
```python
{
    'allocated_time': float,      # Recommended time for this move
    'min_time': float,            # Minimum time (30% of base)
    'max_time': float,            # Maximum time (20% of remaining)
    'base_time': float,           # Base allocation
    'complexity_score': float,    # Overall complexity [0-1]
    'selector_confidence': float, # Selector confidence [0-1]
    'selector_probability': float,# Transformer probability [0-1]
    'time_pressure': float,       # Time pressure factor [0-1]
    'estimated_moves_remaining': int,
    'multipliers': {
        'complexity': float,
        'uncertainty': float,
        'phase': float,
        'time_pressure': float
    }
}
```

#### `update_after_move(time_used: float)`

Update time manager after move is played.

#### `get_statistics() -> Dict`

Get time usage statistics:
```python
{
    'total_time_remaining': float,
    'moves_played': int,
    'total_time_used': float,
    'avg_time_per_move': float,
    'min_time_used': float,
    'max_time_used': float,
    'avg_selector_confidence': float,
    'available_time': float
}
```

## Benefits

### 1. **Efficient Time Usage**
- Spend more time on critical positions
- Spend less time on routine positions
- Automatically adjust to position complexity

### 2. **Leverages Existing Architecture**
- Uses selector metrics already computed
- No additional overhead
- Selector is 97.56% accurate at position classification

### 3. **Adaptive to Game State**
- Adjusts for time pressure
- Considers game phase (opening/middlegame/endgame)
- Estimates moves remaining

### 4. **Improved Strength**
- Deeper search in complex positions
- Faster play in simple positions
- Better time distribution = better overall play

## Performance Characteristics

### Time Distribution Examples

For a 5-minute (300s) game:

| Position Type | Base Time | Allocated | Actual Range |
|---------------|-----------|-----------|--------------|
| Simple Opening | 1.0s | 0.6s | 0.3s - 1.8s |
| Normal Middlegame | 1.0s | 1.2s | 0.3s - 3.0s |
| Complex Middlegame | 1.0s | 2.7s | 0.3s - 6.0s |
| Tactical Crisis | 1.0s | 3.5s | 0.3s - 6.0s |
| Simple Endgame | 1.0s | 1.0s | 0.3s - 3.0s |

### Typical Game Flow

```
Moves 1-10 (Opening):     Avg 0.7s/move = 7s total
Moves 11-30 (Middlegame): Avg 2.5s/move = 50s total
Moves 31-50 (Endgame):    Avg 1.5s/move = 30s total
Emergency reserve:        5s
---
Total: ~92s / 300s used for 50 moves
Remaining: 208s for extension or longer games
```

## Comparison

### Fixed Time (3s per move)
```
All positions get 3s, regardless of complexity
- Simple position: 3s (wasted 2.4s)
- Complex position: 3s (needed 6s, got only 3s)
- 40-move game: 120s total
```

### Dynamic Time Management
```
Time allocated by position needs
- Simple position: 0.6s (saved 2.4s)
- Complex position: 6.0s (extra 3s when needed)
- 40-move game: 90-150s total (adaptive)
- Saved time used for critical positions!
```

## Tuning Parameters

You can adjust these in `time_manager.py`:

```python
# Complexity multiplier range
complexity_multiplier = 0.5 + complexity * 1.5  # 0.5 to 2.0

# Uncertainty bonus
uncertainty_multiplier = 1.0 + (1.0 - confidence) * 0.5  # 1.0 to 1.5

# Phase multipliers
PHASE_MULTIPLIERS = {
    'opening': 0.7,      # Play faster
    'middlegame': 1.2,   # Think more
    'endgame': 1.0       # Normal
}

# Time pressure thresholds
if available_time < 10: pressure = 0.5
elif available_time < 30: pressure = 0.7
elif available_time < 60: pressure = 0.85
else: pressure = 1.0

# Emergency reserve
emergency_time = 5.0  # Keep 5s for time scramble
```

## Future Enhancements

Potential improvements:

1. **Move Importance**: Detect critical moves (passed pawn push, king safety)
2. **Opponent Model**: Spend more time after opponent's unexpected moves
3. **Tree Statistics**: Use search tree branching factor for complexity
4. **Learning**: Adjust multipliers based on game outcomes
5. **Position Volatility**: Detect sharp tactical changes

## Examples

See `demo_time_management.py` for:
- Full interactive demo
- Comparison with/without time management
- Metrics explanation
- Performance analysis

Run with:
```bash
python demo_time_management.py           # Full demo
python demo_time_management.py --explain  # Just show metrics
python demo_time_management.py --compare  # Compare fixed vs dynamic
```

## Summary

**Your hybrid architecture provides unique metrics for intelligent time management:**

✅ **Selector probability** → Position complexity
✅ **Selector confidence** → Uncertainty (more time when uncertain)
✅ **20+ position features** → Tactical/strategic complexity
✅ **97.56% accurate** selector → Reliable complexity assessment

**Result:** Bot automatically spends 3-5x more time on complex positions and 2-3x less on simple positions, improving overall play quality while staying within time controls.
