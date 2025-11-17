# HybridChessBot Documentation

## Overview

`HybridChessBot` is a chess engine that uses your trained hybrid NNUE-Transformer architecture. It combines:
- **NNUE** for fast tactical evaluation
- **Transformer** for strategic positions (endgames, closed positions)
- **Adaptive Selector** (97.56% accuracy) that decides when to use each evaluator
- **Alpha-Beta Search** with transposition table and move ordering

## Quick Start

### Basic Usage

```python
from ChessGame import ChessGame
from HybridChessBot import HybridChessBot

# Create the bot
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=5,
    time_limit=5.0
)

# Use with ChessGame
opponent = SomeOtherBot()
game = ChessGame(bot, opponent)

while not game.is_game_over():
    game.make_move()
```

### Direct Move Selection

```python
import chess
from HybridChessBot import HybridChessBot

bot = HybridChessBot()
board = chess.Board()

# Get best move
move = bot.choose_move(board)
print(f"Best move: {move.uci()}")
```

## Constructor Parameters

```python
HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',  # Path to trained weights
    depth=5,                                   # Maximum search depth
    time_limit=5.0,                           # Time limit per move (seconds)
    device=None,                              # 'cuda'/'cpu' (auto-detect if None)
    verbose=False                             # Print search statistics
)
```

### Parameters Explained

- **checkpoint**: Path to your trained model checkpoint
  - Recommended: `'checkpoints/best_phase2.pt'` (best validation loss + 97.56% selector accuracy)
  - Alternative: `'checkpoints/final_model.pt'` (latest trained model)

- **depth**: Search depth (1-20)
  - `3-4`: Fast play (~1-2 seconds per move)
  - `5-6`: Standard play (~3-5 seconds per move)
  - `7-10`: Strong play (~10-30 seconds per move)
  - `10+`: Very strong but slow

- **time_limit**: Maximum time per move in seconds
  - Acts as a hard limit; search stops when exceeded
  - Useful for timed games
  - Set to `None` for no limit

- **device**: Computing device
  - `'cuda'`: Use GPU (much faster if available)
  - `'cpu'`: Use CPU (slower but works everywhere)
  - `None`: Auto-detect (uses GPU if available)

- **verbose**: Print detailed statistics
  - `True`: Shows search depth, nodes, time, NPS, evaluation stats
  - `False`: Silent operation

## Required Interface

The bot implements the required `choose_move()` interface:

```python
def choose_move(self, board: chess.Board) -> chess.Move:
    """
    Choose the best move for the current position.
    
    Args:
        board: chess.Board object representing current game state
        
    Returns:
        chess.Move object representing the chosen move
    """
```

This makes it compatible with `ChessGame`:

```python
from ChessGame import ChessGame

player1 = HybridChessBot()
player2 = SomeOtherBot()
game = ChessGame(player1, player2)
```

## Additional Methods

### Position Analysis

```python
analysis = bot.analyze_position(board, depth=6)

print(f"Best move: {analysis['best_move']}")
print(f"Score: {analysis['score']} centipawns")
print(f"Nodes: {analysis['nodes']}")
print(f"Time: {analysis['time']}s")
print(f"NPS: {analysis['nps']}")
```

### Statistics

```python
stats = bot.get_statistics()

# Search statistics
print(f"Nodes: {stats['search']['nodes_searched']}")
print(f"Time: {stats['search']['time_elapsed']}s")
print(f"NPS: {stats['search']['nps']}")
print(f"TT hit rate: {stats['search']['tt_hit_rate']}")

# Evaluation statistics
print(f"NNUE evals: {stats['evaluation']['nnue_only_evals']}")
print(f"Transformer evals: {stats['evaluation']['hybrid_evals']}")
```

### Reset Statistics

```python
bot.reset_statistics()  # Clear all counters
```

## Examples

### Example 1: Play Against Another Bot

```python
from ChessGame import ChessGame
from HybridChessBot import HybridChessBot

class RandomBot:
    def choose_move(self, board):
        import random
        return random.choice(list(board.legal_moves))

# Create bots
hybrid = HybridChessBot(depth=5, verbose=True)
random = RandomBot()

# Play game
game = ChessGame(hybrid, random)

while not game.is_game_over():
    print(game)
    game.make_move()

print(f"Result: {game.board.result()}")
```

### Example 2: Analyze Multiple Positions

```python
from HybridChessBot import HybridChessBot
import chess

bot = HybridChessBot(depth=6)

positions = [
    chess.Board(),  # Starting position
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")  # King endgame
]

for i, board in enumerate(positions, 1):
    analysis = bot.analyze_position(board)
    print(f"\nPosition {i}:")
    print(f"  Best move: {analysis['best_move']}")
    print(f"  Score: {analysis['score']:.2f}")
```

### Example 3: Timed Game

```python
from HybridChessBot import HybridChessBot
import chess

# Create fast bot for bullet chess
bot = HybridChessBot(
    depth=4,           # Lower depth
    time_limit=1.0,    # 1 second per move
    verbose=False      # No verbose output
)

board = chess.Board()

# Make 5 moves quickly
for _ in range(5):
    move = bot.choose_move(board)
    print(f"Move: {move.uci()}")
    board.push(move)
```

### Example 4: Compare Evaluation Methods

```python
from HybridChessBot import HybridChessBot
import chess

bot = HybridChessBot(depth=5, verbose=True)

# Play through a game
board = chess.Board()
for _ in range(10):
    move = bot.choose_move(board)
    board.push(move)

# Check statistics
stats = bot.get_statistics()
total = stats['evaluation']['total_evals']
transformer = stats['evaluation']['hybrid_evals']

print(f"\nTransformer usage: {100*transformer/total:.1f}%")
```

## Performance Characteristics

### Speed Benchmarks

| Depth | Nodes | Time (avg) | NPS |
|-------|-------|------------|-----|
| 3 | ~5K | 0.5s | ~10K |
| 4 | ~20K | 2.0s | ~10K |
| 5 | ~80K | 8.0s | ~10K |
| 6 | ~300K | 30s | ~10K |

*Note: Times vary based on position complexity and transformer usage*

### Evaluation Methods

The bot adaptively chooses between:

1. **NNUE-only** (~70-90% of positions)
   - Tactical positions
   - Open positions with captures
   - Positions with checks/threats
   - Fast: ~0.1ms per evaluation

2. **Hybrid (NNUE + Transformer)** (~10-30% of positions)
   - Endgames (few pieces)
   - Closed positions (few legal moves)
   - Strategic positions
   - Slower: ~50ms per evaluation

The **selector network** (97.56% accuracy) automatically decides which method to use.

## Troubleshooting

### "Checkpoint not found"
```
[WARNING] Checkpoint not found: checkpoints/best_phase2.pt
```
**Solution**: Check that the checkpoint file exists. Use absolute path if needed:
```python
bot = HybridChessBot(checkpoint='/full/path/to/checkpoints/best_phase2.pt')
```

### "CUDA out of memory"
**Solution**: Use CPU instead:
```python
bot = HybridChessBot(device='cpu')
```

### Bot plays very slowly
**Solutions**:
- Lower depth: `depth=3` or `depth=4`
- Set time limit: `time_limit=3.0`
- Use CPU if GPU is slow: `device='cpu'`

### Bot makes illegal moves
This shouldn't happen - the bot only returns legal moves. If it does:
```python
import chess
move = bot.choose_move(board)
assert move in board.legal_moves, f"Illegal move: {move}"
```

## Integration with ChessGame

The bot is fully compatible with `ChessGame.py`:

```python
from ChessGame import ChessGame
from HybridChessBot import HybridChessBot

# Required: choose_move(board) -> chess.Move
# ✓ Implemented

# Create players
player1 = HybridChessBot(depth=5)
player2 = SomeOtherBot()

# Create game
game = ChessGame(player1, player2)

# Play
while not game.is_game_over():
    game.make_move()

print(game.board.result())
```

## Advanced Usage

### Custom Checkpoint

```python
# Use a different training checkpoint
bot = HybridChessBot(checkpoint='checkpoints/phase1_epoch25.pt')
```

### Adjust Search Aggressiveness

```python
# Aggressive: Deep search, long time
aggressive_bot = HybridChessBot(depth=8, time_limit=30.0)

# Defensive: Shallow search, quick moves
defensive_bot = HybridChessBot(depth=3, time_limit=1.0)
```

### Monitor Performance

```python
bot = HybridChessBot(verbose=True)

for _ in range(10):
    move = bot.choose_move(board)
    board.push(move)
    
    # Verbose mode prints:
    # - Depth and score
    # - Nodes searched
    # - Time and NPS
    # - NNUE vs Transformer usage
```

## Files Created

- **HybridChessBot.py**: Main bot implementation
- **test_hybrid_bot.py**: Comprehensive test suite
- **quick_start.py**: Interactive examples
- **BOT_USAGE.md**: This documentation

## Testing

Run the test suite:
```bash
python test_hybrid_bot.py
```

Run quick examples:
```bash
python quick_start.py
```

## Architecture

```
HybridChessBot
├── NNUE Evaluator (frozen, pre-trained)
├── Transformer Model (frozen, pre-trained)  
├── Projection Layer (trained, 525K params)
├── Selector Network (trained, 3.5K params)
└── Alpha-Beta Search Engine
    ├── Transposition Table (1M entries)
    ├── Move Ordering (MVV-LVA)
    ├── Iterative Deepening
    └── Quiescence Search
```

## Credits

- Training: 50 epochs (25 projection + 25 joint)
- Data: 1M+ Lichess Elite games (2500+ Elo)
- Validation: 8838 loss, 97.56% selector accuracy
- Architecture: Hybrid NNUE-Transformer with learned routing
