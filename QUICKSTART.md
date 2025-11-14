# Quick Start Guide

## Prerequisites

Virtual environment activated: `chess_env`
Dependencies installed: `torch`, `chess`, `numpy`, `tqdm`
Training data downloaded: 1,064,917 games (955 MB)
GPU available: RTX 6000 Ada Generation (CUDA)

## Training the Model

### Step 1: Verify Setup

```bash
# Test all components
python test_implementation.py

# Test training pipeline
python test_training.py

# Test search engine
python test_search.py
```

All tests should pass ✓

### Step 2: Start Training

```bash
# Full training (50 epochs total: 25 Phase 1 + 25 Phase 2)
python src/train.py
```

**Expected Timeline**: ~2 days on RTX 6000

**Training Process**:
1. **Phase 1** (25 epochs): Train projection layer only
   - Selector frozen
   - Learns NNUE → Transformer feature mapping
   - Saves: `checkpoints/best_phase1.pt`

2. **Phase 2** (25 epochs): Joint training
   - Both projection and selector trainable
   - Selector learns position complexity
   - Saves: `checkpoints/best_phase2.pt`, `checkpoints/final_model.pt`

**Monitoring**:
```
Phase 1, Epoch 1/25: Train Loss: 5286.66 | Val Loss: 3707.98
Phase 1, Epoch 2/25: Train Loss: 3615.83 | Val Loss: 3355.85
...
Phase 2, Epoch 1/25: Train Loss: 3149.49 | Val Loss: 3564.78, Selector Acc: 42.50%
Phase 2, Epoch 2/25: Train Loss: 3095.91 | Val Loss: 3608.17, Selector Acc: 45.00%
...
```

### Step 3: Monitor Training

Check logs for:
- ✓ Loss decreasing each epoch
- ✓ Validation loss stable (not overfitting)
- ✓ Selector accuracy improving (>40%)
- ✓ GPU utilization high
- ✓ Checkpoints saved regularly

## Playing Games

### After Training Completes

Choose your checkpoint:
- `checkpoints/best_phase1.pt` - Best projection layer performance
- `checkpoints/best_phase2.pt` - Best joint training performance  
- `checkpoints/final_model.pt` - Final trained model

### Mode 1: Human vs Engine

```bash
# Play as White
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode human --color white --depth 5

# Play as Black
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode human --color black --depth 5
```

**Gameplay**:
- Enter moves in UCI format: `e2e4`, `g1f3`, `e7e8q` (promotion)
- Type `quit` to exit
- Board displayed after each move

### Mode 2: Self-Play (Testing)

```bash
# Engine plays against itself
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode self --depth 5 --output test_game.pgn
```

**Use Cases**:
- Test model strength
- Generate training data
- Debug search behavior
- Save games for analysis

### Mode 3: Position Analysis

```bash
# Analyze starting position
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode analyze --depth 6

# Analyze custom position (FEN)
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode analyze --depth 6 \
                   --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
```

**Output**:
- Best move found
- Evaluation score
- Search statistics (nodes, time, depth)
- Resulting position

## Configuration

### Search Depth

Deeper = Stronger but slower:
- `--depth 3`: Fast (~0.1s per move)
- `--depth 5`: Balanced (~0.6s per move) **[Recommended]**
- `--depth 6`: Strong (~3.5s per move)
- `--depth 8`: Very strong (~30s per move)

### Time Control

```bash
# Use time limit instead of fixed depth
python src/play.py --checkpoint checkpoints/final_model.pt \
                   --mode human --time-limit 5.0  # 5 seconds per move
```

## Troubleshooting

### Issue: Model not found

```
Error: Checkpoint not found: checkpoints/final_model.pt
```

**Solution**: Train the model first or specify correct checkpoint path:
```bash
python src/train.py  # Train model
# OR
python src/play.py --checkpoint checkpoints/best_phase1.pt
```

### Issue: Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 128  # Instead of 256
```

### Issue: Slow search

```
Search taking too long per move
```

**Solution**: Reduce depth or use time limit:
```bash
python src/play.py --depth 4  # Instead of 5
# OR
python src/play.py --time-limit 3.0  # Max 3 seconds
```

## Performance Expectations

### Search Performance
| Depth | Nodes    | Time  | Strength          |
|-------|----------|-------|-------------------|
| 3     | ~600     | 0.1s  | Beginner          |
| 4     | ~2K      | 0.2s  | Intermediate      |
| 5     | ~12K     | 0.6s  | Club player       |
| 6     | ~70K     | 3.5s  | Strong club       |
| 7     | ~400K    | 20s   | Expert            |
| 8     | ~2M      | 100s  | Master level      |

### Training Performance
- **RTX 6000 Ada**: ~2 days for 50 epochs
- **Batch size 256**: ~20s per epoch
- **80K training positions**: Good coverage
- **GPU utilization**: ~80-90%

## Next Steps

### Experiment with Hyperparameters

Edit `config.py`:
```python
# Try different learning rates
LEARNING_RATE = 5e-4  # Lower = more stable
SELECTOR_LR_MULTIPLIER = 0.05  # Slower selector learning

# Try different architectures
SELECTOR_HIDDEN_DIM_1 = 128  # Bigger selector
SELECTOR_DROPOUT = 0.3  # More regularization
```

### Download More Training Data

```bash
# Get more recent months
python download_lichess_data.py
```

Edit the script to download additional months from:
https://database.nikonoel.fr/

### Analyze Your Games

```bash
# Save your game
python src/play.py --mode human --output mygame.pgn

# Analyze critical positions from your game
# Extract FEN from game and analyze
python src/play.py --mode analyze --fen "YOUR_FEN_HERE" --depth 8
```

## Tips for Strong Play

1. **Use adequate depth**: Minimum depth 5 for serious games
2. **Time management**: Use time limits for consistent play
3. **Opening preparation**: First 10 moves are from training data patterns
4. **Tactical positions**: Model should shine in complex middlegames
5. **Endgames**: May be weaker in theoretical endgames (consider tablebase integration)

## Getting Help

If you encounter issues:

1. Check test files pass: `python test_*.py`
2. Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check checkpoint exists: `ls -lh checkpoints/*.pt`
4. Review error messages for hints
5. Check `config.py` for correct paths

## Resources

- Implementation details: `hybrid_nnue_transformer_implementation.md`
- Search documentation: `src/README_SEARCH.md`
- Data documentation: `data/README.md`
- Main README: `README.md`

---

**Ready to train?** Start with: `python src/train.py`

**Already trained?** Start playing with: `python src/play.py --mode human --color white`
