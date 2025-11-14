# Quick Reference: Chess Bot Training on GPU 4

## Current Status
✅ **Training is RUNNING in tmux session: `chess_train`**
- GPU: 4 (NVIDIA RTX 6000 Ada Generation)
- Session started: 2025-11-13 23:21:08
- Log file: `logs/training_20251113_232108.log`

## Training Configuration
- **Resume from**: best_phase2.pt (50 epochs completed)
- **Phase 1 Target**: 100 epochs (projection layer only)
- **Phase 2 Target**: 100 epochs (joint training)
- **Total**: 200 epochs (~4-5 hours)
- **Already completed**: 50 epochs (from test run)
- **Remaining**: 150 epochs

## Essential Commands

### View Training Progress
```bash
# Attach to the tmux session
tmux attach -t chess_train

# Detach (training keeps running)
Ctrl-b d
```

### Monitor Without Attaching
```bash
# View latest output from log
tail -f logs/training_20251113_232108.log

# Check GPU usage
nvidia-smi

# Watch GPU continuously
watch -n 2 nvidia-smi
```

### Session Management
```bash
# List all tmux sessions
tmux ls

# Kill session if needed
tmux kill-session -t chess_train
```

## Training Progress Indicators

You'll see output like this:
```
Epoch 51 (joint): 100%|█| 313/313 [01:10<00:00, 4.43it/s]
Validating: 100%|████| 79/79 [00:16<00:00, 4.71it/s]

Epoch 51/100
  Train Loss: 8351.0168 | Val Loss: 8250.8051
  Selector Accuracy: 57.91%
  ✓ Saved best Phase 2 model
```

## When Training Completes

Check results:
```bash
# View training summary
cat checkpoints/training_summary_epoch200.json

# View final metrics
tail -100 logs/training_20251113_232108.log
```

## Files Being Created
- `checkpoints/best_phase1.pt` - Best Phase 1 model
- `checkpoints/best_phase2.pt` - Best Phase 2 model
- `checkpoints/phase1_epochX.pt` - Periodic checkpoints
- `checkpoints/phase2_epochX.pt` - Periodic checkpoints
- `checkpoints/final_model_epoch200.pt` - Final model
- `checkpoints/training_history_epoch200.json` - Complete history
- `checkpoints/training_summary_epoch200.json` - Summary stats
- `logs/training_20251113_232108.log` - Full training log

## Expected Timeline
- Test run: 50 epochs in 1.21 hours
- Full run: 200 epochs ≈ 4.8 hours
- Remaining: 150 epochs ≈ 3.6 hours
- Speed: ~4.4 iterations/second

## Initial Metrics (From Test Run)
```
Phase 1 Best Val Loss: 8248.4867
Phase 2 Best Val Loss: 8249.3604
Final Selector Accuracy: 59.70%
```

## Troubleshooting

If you lose connection:
```bash
ssh lisplab-1
tmux attach -t chess_train
```

If training stops unexpectedly:
```bash
# Check what happened
tail -50 logs/training_20251113_232108.log

# Restart from last checkpoint
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
source chess_env/bin/activate
CUDA_VISIBLE_DEVICES=4 python src/train_resume.py \
    --checkpoint best_phase2.pt \
    --phase1-epochs 100 \
    --phase2-epochs 100
```

---
**Remember**: You can safely disconnect from SSH. The training will continue running in tmux!
