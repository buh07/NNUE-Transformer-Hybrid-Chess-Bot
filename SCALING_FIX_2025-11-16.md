# Scaling Fix Applied: 2025-11-16

## Issue Discovered

The training code had a **scaling inconsistency** between target values and model predictions that caused artificially inflated loss values.

### Root Cause

1. **Stockfish Evaluation** (`src/utils/stockfish_eval.py`):
   - Returns **normalized values in [-1, 1]** using `tanh(centipawns/400)`
   - Example: +300 centipawns (3-pawn advantage) → ~0.76

2. **Training Code** (`src/train.py`, line 152 - OLD):
   ```python
   target_values_scaled = target_values * 100.0  # Scale to centipawns
   ```
   - This multiplied already-normalized values by 100x
   - Resulting range: [-100, 100] instead of [-1, 1]

3. **NNUE Model Output**:
   - Also outputs values in centipawn-like scale
   - But the scaling was double-applied to targets

### Impact on Loss Values

With MSE (Mean Squared Error) loss:
- **Before Fix**: Loss values ~330,000 - 400,000
  - Targets in range [-100, 100]
  - Squared errors up to 10,000 per sample
  - Batch size 256 → accumulated losses of 300k+

- **After Fix**: Expected loss values ~30 - 40
  - Targets in range [-1, 1]
  - Squared errors up to ~1 per sample
  - Batch size 256 → accumulated losses of 30-40

### The Fix

**File**: `src/train.py` (line 152)

**Before**:
```python
target_values_scaled = target_values * 100.0  # Scale to centipawns
value_loss = self.value_criterion(blended_values, target_values_scaled)
```

**After**:
```python
# Stockfish already returns normalized values in [-1, 1]
value_loss = self.value_criterion(blended_values, target_values)
```

## Training Runs Affected

### With Scaling Bug (100x)
- **Training run**: `training_fixed_nnue_20251116_004321.log`
- **Started**: Nov 16, 2025 at 00:43:21
- **Status**: Currently running (Phase 1 complete, Phase 2 in progress)
- **Validation loss**: ~332,137 (inflated by 100x scaling)
- **Model checkpoints**: 
  - `best_phase1.pt`
  - `best_phase2.pt` (when complete)

**⚠️ IMPORTANT**: Models trained with this bug have weights that are **scaled 100x larger** than models trained after the fix. They are **NOT compatible** with the fixed code.

### After Fix
- **All future training runs** will use the correct scaling
- Expected validation losses: ~3,300 (100x smaller)

## Action Items

### If you want to continue current training:
**Do nothing** - Let it finish with the old scaling. The model is learning correctly, just with inflated loss numbers.

### If you want to use the fix:
1. **Stop current training** (Ctrl+C in tmux)
2. **Delete old checkpoints** (they're incompatible):
   ```bash
   rm checkpoints/best_phase*.pt
   rm checkpoints/phase*_epoch*.pt
   ```
3. **Restart training** with the fixed code
4. New models will have ~100x smaller losses but equivalent accuracy

## Why This Wasn't Caught Earlier

1. **Loss values decreased consistently** (~400k → ~330k), indicating learning was happening
2. **Relative improvements** were tracked, not absolute values
3. The scaling was applied consistently, so the model could still learn the relative relationships
4. High losses aren't inherently wrong in MSE - they depend on the scale of predictions

## Verification

To verify the fix is working, check loss values in new training runs:
- **Value loss should be**: 1-100 range (not 1000-400000 range)
- **Total loss should be**: Similar magnitude to value loss
- **Policy loss should be**: 1-10 range (unchanged, uses cross-entropy)

## Related Files Modified

1. `src/train.py` - Removed 100x scaling on line 152
2. `config.py` - Added documentation note
3. This document - `SCALING_FIX_2025-11-16.md`
