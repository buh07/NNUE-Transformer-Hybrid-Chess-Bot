"""
Configuration file for Hybrid NNUE-Transformer Chess Engine
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
STOCKFISH_DIR = os.path.join(PROJECT_ROOT, 'Stockfish')
CHESS_TRANSFORMERS_DIR = os.path.join(PROJECT_ROOT, 'chess-transformers')

# Model Paths  
STOCKFISH_NNUE_PATH = os.path.join(STOCKFISH_DIR, 'src', 'nn-49c1193b131c.nnue')  # Latest big network (best)
TRANSFORMER_WEIGHTS_PATH = os.path.join(CHESS_TRANSFORMERS_DIR, 'checkpoints', 'CT-EFT-85', 'CT-EFT-85.pt')  # CT-EFT-85 (~19M params, best available)
# Note: CT-EFT-85 downloaded from chess-transformers official checkpoint

# Model Architecture
NNUE_FEATURE_DIM = 1024  # NNUE accumulator output dimension
TRANSFORMER_DIM = 512    # Transformer hidden dimension
NUM_MOVES = 1858         # Total possible chess moves (including promotions)
SELECTION_FEATURE_DIM = 20  # Number of features for selection function

# Training Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 5e-4  # Base learning rate
SELECTOR_LR_MULTIPLIER = 0.02  # Much lower LR for selector stability (was 0.5, caused instability)
NUM_EPOCHS = 50  # Total epochs (25 Phase 1 + 25 Phase 2) - reduced from 100
GRADIENT_CLIP_NORM = 1.0

# Early Stopping
EARLY_STOPPING_PATIENCE = 8  # Stop if no improvement for 8 epochs
EARLY_STOPPING_MIN_DELTA = 0.01  # Minimum change to qualify as improvement

# Learning Rate Scheduling
LR_SCHEDULER_PATIENCE = 4  # Reduce LR if no improvement for 4 epochs
LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor when reducing
LR_SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate

# Dataset
MAX_TRAIN_POSITIONS = 80000
MAX_VAL_POSITIONS = 20000
PGN_FILES = [
    'data/lichess_elite_2024-10.pgn',
    'data/lichess_elite_2024-09.pgn',
    'data/lichess_elite_2024-08.pgn',
    'data/lichess_elite_2024-07.pgn',
]  # Lichess Elite games: 2500+ vs 2300+ players

# Training Schedule
PHASE1_EPOCHS = 25  # Projection layer only (back to original)
PHASE2_EPOCHS = 25  # Projection + selector (back to original)

# Loss Weights
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
SELECTOR_LOSS_WEIGHT = 0.5

# Selector Parameters
SELECTOR_THRESHOLD = 0.5  # Threshold for using transformer
SELECTOR_POS_WEIGHT = 1.2  # Weight for positive class (transformer helps) - reduced from 1.5
SELECTOR_HIDDEN_DIM_1 = 128  # Hidden layer 1 size
SELECTOR_HIDDEN_DIM_2 = 64   # Hidden layer 2 size
SELECTOR_DROPOUT = 0.2       # Dropout for regularization (increased from 0.15)

# Projection Layer Parameters
PROJECTION_DROPOUT = 0.1

# Search Parameters
MAX_SEARCH_DEPTH = 18
DEFAULT_TIME_PER_MOVE = 5.0  # seconds

# Value Blending
NNUE_VALUE_WEIGHT = 0.3
TRANSFORMER_VALUE_WEIGHT = 0.7

# Inference
TEMPERATURE = 1.0  # Temperature for softmax during move selection

# Device
DEVICE = 'cuda'  # Will check availability at runtime

# Logging
LOG_INTERVAL = 100  # Log every N batches
SAVE_BEST_MODEL = True

# Evaluation
NUM_EVAL_GAMES = 100
EVAL_TIME_PER_MOVE = 3.0
