# NNUE Transformer Hybrid Chess Bot

A state-of-the-art hybrid chess engine combining NNUE (Efficiently Updatable Neural Network) and Transformer architectures with an adaptive selector mechanism. The system intelligently chooses between fast NNUE evaluation and deep Transformer analysis based on position complexity.

## 🎯 Key Features

### Hybrid Architecture
- **NNUE Evaluator**: Fast position evaluation using Stockfish NNUE weights
- **Transformer Model**: Deep analysis for complex positions
- **Trainable Projection Layer**: Bridges NNUE features to Transformer input (525K parameters)
- **Adaptive Selector**: Learns when to use Transformer vs NNUE-only (3.5K parameters)

### Search Engine
- **Alpha-Beta Search** with pruning for efficient move selection
- **Transposition Table** (1M entries) with LRU eviction
- **Move Ordering**: MVV-LVA captures, checks, promotions
- **Iterative Deepening** with time management
- **Quiescence Search** to avoid horizon effect

### Training
- **Two-Phase Training**: Projection layer first, then joint optimization
- **High-Quality Data**: 1M+ games from Lichess Elite (2500+ Elo)
- **GPU Accelerated**: Optimized for CUDA (tested on RTX 6000)

## 📁 Project Structure

```
NNUE-Transformer-Hybrid-Chess-Bot/
├── config.py                    # Hyperparameters and paths
├── src/
│   ├── models/
│   │   ├── nnue_evaluator.py    # Stockfish NNUE wrapper
│   │   ├── transformer_model.py # Transformer wrapper
│   │   ├── projection_layer.py  # Trainable bridge (525K params)
│   │   ├── selector.py          # Adaptive selector (3.5K params)
│   │   └── hybrid_evaluator.py  # Combined system
│   ├── utils/
│   │   └── chess_utils.py       # Move encoding, feature extraction
│   ├── dataset.py               # PGN data loading
│   ├── train.py                 # Two-phase training pipeline
│   ├── search.py                # Alpha-beta search engine
│   └── play.py                  # Game playing interface
├── data/
│   ├── lichess_elite_2024-*.pgn # Training data (1M+ games)
│   └── README.md                # Data documentation
├── checkpoints/                 # Model checkpoints
├── Stockfish/                   # Stockfish NNUE repository
├── chess-transformers/          # Transformer repository
└── tests/
    ├── test_implementation.py   # Component tests
    ├── test_training.py         # Training pipeline tests
    └── test_search.py           # Search engine tests
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/buh07/NNUE-Transformer-Hybrid-Chess-Bot.git
cd "NNUE Transformer Hybrid Chess Bot"

# Activate virtual environment
source chess_env/bin/activate  # Already set up

# Install dependencies (if needed)
pip install torch chess numpy tqdm requests
```

### 2. Training

The training data is already downloaded (1,064,917 games, 955 MB).

```bash
# Train the model (two-phase: projection → joint)
python src/train.py

# Expected: ~2 days on RTX 6000 Ada
# Phase 1: 25 epochs (projection only)
# Phase 2: 25 epochs (joint training)
```

### 3. Playing

```bash
# Play as white against the engine
python src/play.py --checkpoint checkpoints/final_model.pt --mode human --color white

# Self-play for testing
python src/play.py --mode self --depth 5 --output game.pgn

# Analyze a position
python src/play.py --mode analyze --fen "POSITION_FEN" --depth 6
```

## 🎮 Usage Examples

### Human vs Engine

```bash
$ python src/play.py --mode human --color white --depth 5

============================================================
Playing against Hybrid NNUE-Transformer Chess Engine
You are playing as White
Enter moves in UCI format (e.g., e2e4, g1f3)
============================================================

Your turn (White)
Enter your move: e2e4

Engine thinking (Black)...
Depth 5: score=0.510, move=e7e5, nodes=13580, time=0.68s, nps=19970
Best move: e7e5 (score: 0.510)
Engine plays: e7e5
```

### Position Analysis

```bash
$ python src/play.py --mode analyze --fen "START" --depth 6

Position Analysis
============================================================
Analyzing to depth 6...
Best move: e2e4
Resulting position:
[shows resulting board]
```

## 🧪 Testing

```bash
# Test all components
python test_implementation.py

# Test training pipeline
python test_training.py

# Test search engine
python test_search.py
```

All tests should pass ✓

## 📊 Training Data

**Source**: [Lichess Elite Database](https://database.nikonoel.fr/)

| File | Games | Size | Rating Range |
|------|-------|------|--------------|
| lichess_elite_2024-10.pgn | 274,426 | 247 MB | 2500+ vs 2300+ |
| lichess_elite_2024-09.pgn | 259,278 | 232 MB | 2500+ vs 2300+ |
| lichess_elite_2024-08.pgn | 273,032 | 245 MB | 2500+ vs 2300+ |
| lichess_elite_2024-07.pgn | 258,181 | 231 MB | 2500+ vs 2300+ |
| **Total** | **1,064,917** | **955 MB** | High quality |

Download more data:
```bash
python download_lichess_data.py
```

## 🏗️ Architecture Details

### Hybrid Evaluation Pipeline

```
Chess Position
    ↓
┌───────────────────────┐
│  NNUE Evaluator       │  Fast evaluation
│  (Frozen Weights)     │  → Features [1024]
└───────────────────────┘
    ↓
┌───────────────────────┐
│  Selector Network     │  Decides: Use transformer?
│  (Trainable: 3.5K)    │  → Boolean decision
└───────────────────────┘
    ↓
    ├─[If complex position]─┐
    │                        ↓
    │               ┌──────────────────┐
    │               │ Projection Layer │
    │               │ (Trainable: 525K)│
    │               │ 1024 → 512       │
    │               └──────────────────┘
    │                        ↓
    │               ┌──────────────────┐
    │               │ Transformer      │
    │               │ (Frozen Weights) │
    │               │ → Policy, Value  │
    │               └──────────────────┘
    │                        ↓
    └────────────────────────┴────────
                    ↓
        Final Policy + Value Prediction
```

### Key Parameters

- **NNUE Feature Dim**: 1024
- **Transformer Dim**: 512
- **Move Space**: 1858 (all legal chess moves)
- **Selection Features**: 20 (game phase, material, etc.)
- **Batch Size**: 256
- **Learning Rate**: 1e-3 (0.1x for selector)

## 📈 Performance

### Search Statistics
- **Depth 5**: ~12K nodes, ~0.6s, 20K nps
- **Depth 6**: ~70K nodes, ~3.5s, 20K nps
- **TT Hit Rate**: ~13-15%

### Training Progress
- **Phase 1**: Projection layer learns NNUE→Transformer mapping
- **Phase 2**: Selector learns position complexity patterns
- **Expected Accuracy**: Selector ~45-50% on validation (baseline)

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PHASE1_EPOCHS = 25  # Projection only
PHASE2_EPOCHS = 25  # Joint training

# Search
max_depth = 5
tt_size = 1000000
use_quiescence = True

# Data
MAX_TRAIN_POSITIONS = 80000
MAX_VAL_POSITIONS = 20000
```

## 🤝 Contributing

This project is part of a research implementation. See `hybrid_nnue_transformer_implementation.md` for detailed architecture documentation.

## 📝 License

Creative Commons CC0 (training data from Lichess)

## 🙏 Acknowledgments

- **Stockfish Team**: NNUE architecture and weights
- **chess-transformers**: Transformer model implementation
- **Lichess**: Elite game database
- **nikonoel**: Curated elite database

## 📚 References

- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish)
- [chess-transformers](https://github.com/sgrvinod/chess-transformers)
- [Lichess Database](https://database.lichess.org/)
- Implementation details: `hybrid_nnue_transformer_implementation.md`

---

**Status**: ✅ Implementation Complete | 🎯 Ready for Training | 🎮 Game Playing Ready
