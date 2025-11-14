#!/bin/bash
# Quick Start: Launch New Training with Fixes
# This script starts training with all the improvements applied

set -e  # Exit on error

PROJECT_DIR="/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "Starting Hybrid NNUE-Transformer Training (FIXED VERSION)"
echo "======================================================================"
echo ""
echo "Fixes Applied:"
echo "  ✓ Reduced selector LR multiplier: 0.5 → 0.02 (25x reduction)"
echo "  ✓ Added early stopping (patience: 8 epochs)"
echo "  ✓ Added learning rate scheduling (ReduceLROnPlateau)"
echo "  ✓ Reduced total epochs: 100 → 50"
echo "  ✓ Better regularization and monitoring"
echo ""
echo "Expected improvements:"
echo "  - Validation loss: ~3,300-3,700 (was 9,527)"
echo "  - Selector accuracy: >55% (was 47%)"
echo "  - Training time: 1-2 hours (was 4.77 hours)"
echo "  - No overfitting: train/val gap <1,000"
echo ""
echo "======================================================================"
echo ""

# Activate environment
echo "Activating environment..."
source chess_env/bin/activate

# Backup old checkpoints
if [ -d "checkpoints" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="checkpoints_backup_${TIMESTAMP}"
    echo "Backing up old checkpoints to ${BACKUP_DIR}..."
    cp -r checkpoints "$BACKUP_DIR"
fi

# Start training
echo ""
echo "Starting training..."
echo "Log will be saved to: logs/training_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Create logs directory
mkdir -p logs

# Run training with output to both terminal and log file
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
python src/train.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "======================================================================"
echo "Check results:"
echo "  - Checkpoints: checkpoints/"
echo "  - Training history: checkpoints/training_history.json"
echo "  - Summary: checkpoints/training_summary.json"
echo "  - Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review training_summary.json for metrics"
echo "  2. Test the model: python src/test_implementation.py"
echo "  3. Play games against baseline: python src/play_game.py"
echo ""
