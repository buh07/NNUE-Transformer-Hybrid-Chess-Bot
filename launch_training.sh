#!/bin/bash
# Launch complete training run in tmux on GPU 4

# Configuration
SESSION_NAME="chess_train"
GPU_ID=4
PHASE1_EPOCHS=100
PHASE2_EPOCHS=100
CHECKPOINT="best_phase2.pt"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
    echo "Error: tmux session '$SESSION_NAME' already exists!"
    echo "Please attach to it with: tmux attach -t $SESSION_NAME"
    echo "Or kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session and run training
echo "Starting training in tmux session: $SESSION_NAME"
echo "GPU: $GPU_ID"
echo "Phase 1 epochs: $PHASE1_EPOCHS"
echo "Phase 2 epochs: $PHASE2_EPOCHS"
echo "Resume from checkpoint: $CHECKPOINT"
echo ""
echo "Commands to manage the session:"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Detach:  Ctrl-b d"
echo "  Kill:    tmux kill-session -t $SESSION_NAME"
echo ""

# Get the project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create tmux session with training command
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "source chess_env/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting training on GPU $GPU_ID...'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Phase 1: $PHASE1_EPOCHS epochs | Phase 2: $PHASE2_EPOCHS epochs'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo '----------------------------------------'" C-m
tmux send-keys -t $SESSION_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID python src/train_resume.py --checkpoint $CHECKPOINT --phase1-epochs $PHASE1_EPOCHS --phase2-epochs $PHASE2_EPOCHS 2>&1 | tee logs/training_\$(date +%Y%m%d_%H%M%S).log" C-m

echo "Training started in tmux session: $SESSION_NAME"
echo ""
echo "To view progress, attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
