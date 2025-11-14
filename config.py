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
LEARNING_RATE = 5e-4  # Reduced for better stability with larger selector
SELECTOR_LR_MULTIPLIER = 0.5  # Selector gets moderate learning rate (was too low at 0.1)
NUM_EPOCHS = 100  # Total epochs (50 Phase 1 + 50 Phase 2)
GRADIENT_CLIP_NORM = 1.0

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
PHASE1_EPOCHS = 50  # Projection layer only (increased for better convergence)
PHASE2_EPOCHS = 50  # Projection + selector (increased for selector learning)

# Loss Weights
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
SELECTOR_LOSS_WEIGHT = 0.5

# Selector Parameters
SELECTOR_THRESHOLD = 0.5  # Threshold for using transformer
SELECTOR_POS_WEIGHT = 1.5  # Weight for positive class (transformer helps) - compensates for imbalance
SELECTOR_HIDDEN_DIM_1 = 128  # Increased capacity for complex decision
SELECTOR_HIDDEN_DIM_2 = 64   # Increased capacity
SELECTOR_DROPOUT = 0.15      # Reduced dropout for better learning

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
