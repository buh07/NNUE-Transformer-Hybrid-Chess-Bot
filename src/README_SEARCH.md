# Alpha-Beta Search Implementation

This directory contains the search engine for the Hybrid NNUE-Transformer Chess Bot.

## Components

### `search.py`

Complete alpha-beta search implementation with:

#### Features
- **Alpha-Beta Pruning**: Efficient tree search with pruning
- **Transposition Table**: LRU cache for position evaluations (1M entries default)
- **Move Ordering**: Prioritizes captures (MVV-LVA), checks, and promotions
- **Iterative Deepening**: Progressively deeper searches with time management
- **Quiescence Search**: Searches captures to avoid horizon effect
- **Time Management**: Respects time limits with graceful interruption

#### Classes

**`TranspositionTable`**
- Caches position evaluations with Zobrist hashing
- Stores: score, depth, entry type (exact/alpha/beta), best move
- LRU eviction when full
- Hit rate tracking for performance analysis

**`AlphaBetaSearch`**
- Main search engine using hybrid evaluator
- Configurable max depth and table size
- Search statistics (nodes, time, nps)
- Three search modes:
  - `get_best_move()`: Simple interface for best move
  - `alpha_beta()`: Core recursive search with pruning
  - `search_iterative_deepening()`: Progressive deepening with stats

### `play.py`

Game playing interface with three modes:

#### Modes

**Self-Play**
```bash
python src/play.py --mode self --depth 5 --output game.pgn
```
- Engine plays against itself
- Useful for testing and generating training data
- Saves games to PGN format

**Human vs Engine**
```bash
python src/play.py --mode human --color white --depth 5
```
- Play against the trained model
- UCI move input (e.g., e2e4, g1f3)
- Displays board after each move

**Position Analysis**
```bash
python src/play.py --mode analyze --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 6
```
- Analyze a specific position
- Shows best move and evaluation
- Supports custom FEN strings

## Usage

### After Training

Once you have a trained model checkpoint:

```bash
# Play as white against the engine
python src/play.py --checkpoint checkpoints/final_model.pt --mode human --color white

# Self-play for testing
python src/play.py --checkpoint checkpoints/best_phase2.pt --mode self --depth 6

# Analyze a position
python src/play.py --checkpoint checkpoints/final_model.pt --mode analyze \
    --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" \
    --depth 8
```

### Command Line Options

```
--checkpoint PATH    Path to model checkpoint (default: checkpoints/final_model.pt)
--mode MODE         Game mode: self, human, analyze (default: human)
--depth DEPTH       Search depth (default: 5)
--color COLOR       Human color: white, black (default: white)
--fen FEN          FEN position for analysis mode
--output PATH       Output PGN file for self-play mode
--device DEVICE     Device: cuda, cpu (default: cuda if available)
```

## Performance

### Search Statistics

Typical performance on test positions:

```
Depth 5 (starting position):
- Nodes: ~12,000
- Time: ~0.6s
- NPS: ~20,000 nodes/second
- TT hit rate: ~13%

Depth 6 (starting position):
- Nodes: ~70,000
- Time: ~3.5s
- NPS: ~20,000 nodes/second
- TT hit rate: ~15%
```

### Move Ordering Effectiveness

With MVV-LVA and check prioritization:
- Captures evaluated first
- Promotions prioritized
- Checks explored early
- Results in better pruning and faster search

## Testing

Run the test suite:

```bash
python test_search.py
```

Tests cover:
- ✓ Transposition table functionality
- ✓ Move ordering (captures, checks, promotions)
- ✓ Basic alpha-beta search
- ✓ Iterative deepening
- ✓ Checkmate detection
- ✓ Time-limited search

All tests should pass.

## Search Improvements (Future)

Potential enhancements:
- [ ] Killer move heuristic
- [ ] History heuristic for move ordering
- [ ] Null move pruning
- [ ] Late move reductions (LMR)
- [ ] Multi-PV search for analysis
- [ ] Pondering (thinking on opponent's time)
- [ ] Opening book integration
- [ ] Endgame tablebase probing

## Architecture Notes

### Integration with Hybrid Evaluator

The search engine uses the hybrid evaluator for position evaluation:

```python
policy_logits, value, use_transformer = evaluator.evaluate(board)
```

- `value`: Position evaluation (centipawns from white's perspective)
- `policy_logits`: Move probabilities (could be used for move ordering)
- `use_transformer`: Whether transformer was used (for statistics)

### Quiescence Search

Searches only capture moves to reach "quiet" positions:
- Prevents horizon effect (missing tactics just beyond search depth)
- Default max depth: 4 plies
- Stand-pat evaluation for beta cutoffs
- Critical for tactical positions

### Time Management

Iterative deepening allows graceful time management:
1. Search depth 1, store best move
2. Search depth 2, update best move
3. Continue until time limit or max depth
4. Return best move from deepest complete iteration

Even if interrupted, always has a valid move from previous iteration.

## Example Session

```
$ python src/play.py --mode human --color white --depth 5

============================================================
Playing against Hybrid NNUE-Transformer Chess Engine
You are playing as White
Enter moves in UCI format (e.g., e2e4, g1f3)
Type 'quit' to exit
============================================================
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
============================================================

Your turn (White)
Legal moves: g1h3, g1f3, b1c3, b1a3, h2h3...
Enter your move: e2e4

Engine thinking (Black)...
Depth 1: score=0.523, move=e7e5, nodes=20, time=0.00s, nps=25000
Depth 2: score=-0.412, move=e7e5, nodes=82, time=0.01s, nps=18500
Depth 3: score=0.498, move=e7e5, nodes=615, time=0.03s, nps=20000
Depth 4: score=-0.401, move=e7e5, nodes=2104, time=0.11s, nps=19127
Depth 5: score=0.510, move=e7e5, nodes=13580, time=0.68s, nps=19970

Best move: e7e5 (score: 0.510)
Engine plays: e7e5

r n b q k b n r
p p p p . p p p
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
------------------------------------------------------------
```
