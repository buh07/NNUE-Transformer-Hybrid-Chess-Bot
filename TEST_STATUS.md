# Test Status Summary - Strategic Routing Framework

## Tests Running

### 1. Projection Architecture Test (WITH Stockfish)
**Status:** ⏳ Running  
**File:** `test_projection_with_stockfish.log`  
**Command:** `python test_projection_architectures.py`

**Changes Made:**
- ✅ Updated to use Stockfish depth-10 targets
- ✅ Now matches training setup with strategic routing
- ✅ Tests 4 projection architectures: V1-V4

**Expected:** Test will take ~10-15 minutes with Stockfish evaluations

---

### 2. Blending Weights Test (WITH Stockfish)
**Status:** ⏳ Running (may have stalled)
**File:** `test_blending_strategic.log`  
**Command:** `python test_blending_weights.py`

**What it Tests:**
- ✅ Tests 10 NNUE/Transformer blending ratios
- ✅ Tests with BOTH Stockfish AND game outcome targets
- ✅ Compares results to detect bias

**Expected:** Test will take ~30-40 minutes (20 dataloader passes with Stockfish)

---

## Strategic Routing Changes Implemented

### Feature Enhancements (20 features total):
- **Feature 16:** Forcing move ratio (tactical indicator)
- **Feature 17:** Pawn chains (closed structure)
- **Feature 18:** Piece coordination (strategic harmony)
- **Feature 19:** Material imbalance (precision needed)
- **Feature 20:** Mobility (open vs closed)

### Selector Training Logic:
```python
# NEW: Label positions as strategic
endgame = features[4] > 0.5          # 6-12 pieces
few_captures = features[1] < 0.2     # Low tactical complexity
not_in_check = features[7] < 0.5     # No forcing sequences
closed = features[0] < 0.4           # Few legal moves

is_strategic = (endgame | closed) & few_captures & not_in_check
selector_labels = is_strategic
```

---

## Why Tests are Slow

**Stockfish Evaluation:** Each position requires:
1. Spawning Stockfish process
2. Running depth-10 search (~1-4 seconds per position)
3. Parsing evaluation result

**Test Volumes:**
- Projection test: 5000 positions × 4 architectures = 20,000 evaluations
- Blending test: 5000 positions × 10 configs × 2 target types = 100,000 evaluations

**Time Estimates:**
- Projection: ~10-15 minutes
- Blending: ~30-60 minutes (with Stockfish)

---

## Previous Test Results (Game Outcomes Only)

### Projection Test (Before Stockfish):
- **All architectures:** 8168.80 loss (identical!)
- **Reason:** Using placeholder NNUE weights + game outcomes
- **Conclusion:** Need Stockfish targets to see real differences

### Blending Test (Before Bias Fix):
- **Pure NNUE (1.0/0.0):** 8214.84 loss
- **Current (0.3/0.7):** Much worse
- **Reason:** Tested against game outcomes, but now training with Stockfish

---

## Current Training Status

**Main Training:** Running in tmux session 2
- Epoch 1, batch 454/782 (~58% of phase 1)
- Using: V3 projection + pure NNUE blending + Stockfish targets + strategic routing
- Training data: 200K positions (782 batches)
- Validation: 50K positions (196 batches)

---

## Next Steps

1. ⏳ Wait for projection test to complete (~10 min)
2. ⏳ Wait for blending test to complete (~30 min)
3. 📊 Compare results with/without Stockfish targets
4. 📊 Validate strategic routing features are working
5. 🎯 Continue main training until Phase 2 (joint training)
6. 🎯 Phase 2 will learn strategic routing with new selector labels

---

Generated: November 14, 2025
