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
STOCKFISH_NNUE_PATH = os.path.join(STOCKFISH_DIR, 'src', 'nn-0000000000a0.nnue')
TRANSFORMER_WEIGHTS_PATH = os.path.join(CHESS_TRANSFORMERS_DIR, 'checkpoints')

# Model Architecture
NNUE_FEATURE_DIM = 1024  # NNUE accumulator output dimension
TRANSFORMER_DIM = 512    # Transformer hidden dimension
NUM_MOVES = 1858         # Total possible chess moves (including promotions)
SELECTION_FEATURE_DIM = 20  # Number of features for selection function

# Training Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SELECTOR_LR_MULTIPLIER = 0.1  # Selector gets lower learning rate
NUM_EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0

# Dataset
MAX_TRAIN_POSITIONS = 80000
MAX_VAL_POSITIONS = 20000
PGN_FILES = []  # Will be populated with actual PGN file paths

# Training Schedule
PHASE1_EPOCHS = 25  # Projection layer only
PHASE2_EPOCHS = 25  # Projection + selector

# Loss Weights
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
SELECTOR_LOSS_WEIGHT = 0.5

# Selector Parameters
SELECTOR_THRESHOLD = 0.5  # Threshold for using transformer
SELECTOR_HIDDEN_DIM_1 = 64
SELECTOR_HIDDEN_DIM_2 = 32
SELECTOR_DROPOUT = 0.2

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
