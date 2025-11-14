# Training Status and Complete Training Guide

## Current Training Status (Test Run Completed)

### Test Run Summary
- **Duration**: 1.21 hours
- **Epochs Completed**: 50 (25 Phase 1 + 25 Phase 2)
- **GPU Used**: GPU 4 (NVIDIA RTX 6000 Ada Generation, 50.88 GB)
- **Dataset**: 100,000 positions from 4 PGN files

### Final Metrics from Test Run
```
Final train loss:     8345.8792
Final val loss:       8252.1513
Selector accuracy:    59.70%

Best Phase 1 val loss: 8248.4867 (epoch 18)
Best Phase 2 val loss: 8249.3604 (epoch 3)
```

### Saved Checkpoints
Located in `checkpoints/` directory:
- `best_phase1.pt` - Best model from Phase 1 (projection only)
- `best_phase2.pt` - Best model from Phase 2 (joint training) ✓ USE THIS
- `final_model.pt` - Final model after 50 epochs
- `training_history.json` - Complete training history
- Periodic checkpoints every 5 epochs

---

## Complete Training Plan

### Training Configuration
We will extend training from the current checkpoint for a full production run:

- **Phase 1 (Projection Only)**: 100 epochs total
- **Phase 2 (Joint Training)**: 100 epochs total
- **Total Training**: 200 epochs (~4-5 hours on GPU 4)
- **Starting Point**: Resume from `best_phase2.pt` checkpoint

### What This Means
- Already completed: 50 epochs (25+25)
- Remaining: 150 epochs (75 Phase 1 + 75 Phase 2)
- The training will continue from the best Phase 2 checkpoint

---

## How to Start Complete Training

### Option 1: Automated Launch (Recommended)
```bash
# From the project directory
./launch_training.sh
```

This will:
1. Create a tmux session named `chess_train`
2. Activate the virtual environment
3. Start training on GPU 4
4. Save logs to `logs/training_YYYYMMDD_HHMMSS.log`
5. Resume from `best_phase2.pt` checkpoint

### Option 2: Manual Launch
```bash
# Create tmux session
tmux new -s chess_train

# Activate environment
source chess_env/bin/activate

# Run training on GPU 4
CUDA_VISIBLE_DEVICES=4 python src/train_resume.py \
    --checkpoint best_phase2.pt \
    --phase1-epochs 100 \
    --phase2-epochs 100

# Detach from tmux
Ctrl-b d
```

---

## Managing the Training Session

### Check on Training Progress
```bash
# Attach to the tmux session
tmux attach -t chess_train

# Detach again (training continues)
Ctrl-b d
```

### View Training Logs
```bash
# View latest log file
tail -f logs/training_*.log

# Or view specific log
tail -f logs/training_20241113_143022.log
```

### Monitor GPU Usage
```bash
# In a separate terminal
watch -n 2 nvidia-smi

# Or check specific GPU
watch -n 2 "nvidia-smi | grep -A 10 'GPU.*4'"
```

### Stop Training (If Needed)
```bash
# Attach to session and press Ctrl-C
tmux attach -t chess_train
# Then press Ctrl-C

# Or kill the entire session
tmux kill-session -t chess_train
```

---

## Training Progress Tracking

### Real-time Monitoring
During training, you'll see:
```
Epoch 51 (joint): 100%|█| 313/313 [01:10<00:00, 4.43it/s, loss=8112.98...]
Validating: 100%|███████████████| 79/79 [00:16<00:00, 4.71it/s]

Epoch 51/100
  Train Loss: 8351.0168 | Val Loss: 8250.8051
  Train Policy: 3.2116 | Val Policy: 3.2118
  Train Value: 8347.4576 | Val Value: 8247.2505
  Train Selector: 0.6952 | Val Selector: 0.6856
  Selector Accuracy: 57.91%
  ✓ Saved best Phase 2 model (val_loss=8250.8051)
```

### Checkpoints Saved During Training
- `best_phase1.pt` - Best validation loss during Phase 1
- `best_phase2.pt` - Best validation loss during Phase 2
- `phase1_epochX.pt` - Saved every 5 epochs
- `phase2_epochX.pt` - Saved every 5 epochs
- `final_model_epoch200.pt` - Final model at end
- `training_history_epoch200.json` - Complete history
- `training_summary_epoch200.json` - Final summary statistics

---

## Expected Timeline

Based on test run performance (~1.2 hours for 50 epochs):
- **Complete 200 epochs**: ~4.8 hours
- **Remaining 150 epochs**: ~3.6 hours

Speed: ~4.4-4.5 iterations/second on GPU 4

---

## After Training Completes

### Check Final Results
```bash
# View final summary
cat checkpoints/training_summary_epoch200.json

# Plot training curves (if needed)
python scripts/plot_training.py checkpoints/training_history_epoch200.json
```

### Test the Trained Model
```bash
# Play against the model
python src/play.py --checkpoint checkpoints/best_phase2.pt

# Run test suite
python test_implementation.py
python test_search.py
```

---

## Troubleshooting

### If Training Crashes
The checkpoint system saves progress, so you can resume:
```bash
# Check which epoch was last saved
ls -lt checkpoints/phase2_epoch*.pt | head -1

# Resume from last checkpoint
CUDA_VISIBLE_DEVICES=4 python src/train_resume.py \
    --checkpoint phase2_epochXX.pt \
    --phase1-epochs 100 \
    --phase2-epochs 100
```

### If Session is Lost
```bash
# List all tmux sessions
tmux ls

# Attach to any session
tmux attach -t chess_train
```

### If GPU Runs Out of Memory
Edit `config.py`:
```python
BATCH_SIZE = 128  # Reduce from 256
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Start training | `./launch_training.sh` |
| Check progress | `tmux attach -t chess_train` |
| Detach | `Ctrl-b d` |
| Check GPU | `nvidia-smi` |
| View logs | `tail -f logs/training_*.log` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t chess_train` |

---

## Notes

- Training runs on **GPU 4** specifically
- Virtual environment is automatically activated
- All outputs are logged to `logs/` directory
- Checkpoints are saved to `checkpoints/` directory
- Training history is preserved and extended
- You can safely disconnect and reconnect at any time
