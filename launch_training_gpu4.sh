#!/bin/bash
# Launch training on GPU 4 with fixed NNUE (using Stockfish)

# Set GPU to use
export CUDA_VISIBLE_DEVICES=4

# Activate environment
source chess_env/bin/activate

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_fixed_nnue_${TIMESTAMP}.log"

echo "======================================================================="
echo "Starting Training with Fixed NNUE (Stockfish-based)"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  GPU: 4"
echo "  Training positions: 250,000 (from Lichess elite games)"
echo "  NNUE: Using Stockfish engine for all evaluations"
echo "  Consistency: Training and inference now match!"
echo "  Log file: ${LOG_FILE}"
echo ""
echo "Note: Training will be slower due to Stockfish calls, but accuracy will be MUCH better"
echo ""
echo "Starting training..."
echo ""

# Run training with unbuffered output
cd src && python -u train.py 2>&1 | tee "../${LOG_FILE}"

echo ""
echo "Training complete! Check ${LOG_FILE} for details"
