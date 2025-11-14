# Training Improvements Applied

## Problem Analysis (from analyzer output)

**Current Issues:**
- ✓ Transformer weights loaded (CT-EFT-85, 19M params)
- ❌ Selector predicts "use transformer" for everything (100% recall, 40.9% precision)
- ❌ Class imbalance: transformer helps 40.9% of time, NNUE better 59.1%
- ❌ Sharp threshold sensitivity (performance drops at 0.6+)
- ❌ Model too simple for complex decision

## Changes Made

### 1. **Increased Selector Capacity** ✓
```python
# Before
SELECTOR_HIDDEN_DIM_1 = 64
SELECTOR_HIDDEN_DIM_2 = 32

# After
SELECTOR_HIDDEN_DIM_1 = 128  # 2x capacity
SELECTOR_HIDDEN_DIM_2 = 64   # 2x capacity
```
**Why:** The selector needs more capacity to learn the complex pattern of when transformer helps.

### 2. **Weighted Loss for Class Imbalance** ✓
```python
SELECTOR_POS_WEIGHT = 1.5  # Weight positive class more
```
Added weighted BCE loss that emphasizes the minority class (transformer helps).

**Why:** Without weighting, model learns to just predict the majority class (NNUE).

### 3. **Focal Loss for Hard Examples** ✓
```python
focal_gamma = 2.0
focal_weight = (1 - pt) ** focal_gamma
```
Added focal loss component that focuses on hard-to-classify examples.

**Why:** Prevents model from being overly confident on easy examples; forces it to learn the boundary cases.

### 4. **Extended Training Duration** ✓
```python
# Before
PHASE1_EPOCHS = 25
PHASE2_EPOCHS = 25
# Total: 50 epochs

# After
PHASE1_EPOCHS = 50
PHASE2_EPOCHS = 50
# Total: 100 epochs
```
**Why:** Selector needs more time to learn the complex decision boundary.

### 5. **Adjusted Learning Rates** ✓
```python
# Before
LEARNING_RATE = 1e-3
SELECTOR_LR_MULTIPLIER = 0.1  # Too small!

# After
LEARNING_RATE = 5e-4  # More stable
SELECTOR_LR_MULTIPLIER = 0.5  # Allows selector to learn faster
```
**Why:** Selector was barely updating (0.1 multiplier too aggressive). Now it can actually learn.

### 6. **Reduced Dropout** ✓
```python
# Before
SELECTOR_DROPOUT = 0.2

# After
SELECTOR_DROPOUT = 0.15
```
**Why:** With increased capacity, less aggressive regularization allows better learning.

## Expected Improvements

### Before Training:
- Precision: ~40.9% (predicting transformer for almost everything)
- Recall: 100% (catches all cases but with many false positives)
- Accuracy: ~41%
- F1 Score: 0.581

### After Training (Expected):
- Precision: **65-75%** (fewer false positives)
- Recall: **85-95%** (still catching most cases)
- Accuracy: **75-80%**
- F1 Score: **0.75-0.80**

## How to Train with Improvements

### Option 1: Fresh Training (Recommended)
```bash
# Start fresh training with improved architecture
./launch_training.sh
```

### Option 2: Resume from Checkpoint
```bash
# Resume training but with new architecture
# Note: Selector will reinitialize due to size change
source chess_env/bin/activate
CUDA_VISIBLE_DEVICES=4 python src/train.py
```

**Important:** Since we changed selector architecture (64→128, 32→64), you cannot directly resume from old checkpoints. The projection layer weights can be loaded, but selector will start fresh.

## Monitoring Training

### Watch for These Metrics:
1. **Selector Accuracy**: Should increase from ~60% to 75-80%
2. **Precision**: Should improve from 40% to 65-75%
3. **Recall**: Should stabilize at 85-95% (not 100%)
4. **Validation Loss**: Should decrease steadily

### Run Analysis After Training:
```bash
python src/analyze_selector.py --checkpoint checkpoints/best_phase2.pt
```

Look for:
- Precision > 65%
- Recall > 85%
- Better threshold analysis (smoother curve from 0.4-0.7)

## Additional Optimizations (Not Yet Implemented)

### If results still aren't good enough:

1. **Add More Selection Features**
   - Current: 20 features
   - Consider: piece mobility, king safety, pawn structure complexity
   
2. **Use Attention Mechanism**
   - Replace MLP with attention over position features
   
3. **Ensemble Methods**
   - Train multiple selectors and vote
   
4. **Curriculum Learning**
   - Train first on "obvious" cases (large error differences)
   - Then on boundary cases

5. **Add Regularization Terms**
   - Penalize selector for being too aggressive
   - Reward high-confidence correct predictions

## Summary

The key issue was that your selector was:
1. **Too small** (64-32 neurons) for complex decision
2. **Learning too slowly** (0.1 LR multiplier)
3. **Ignoring class imbalance** (41% vs 59% split)
4. **Not focusing on hard examples** (no focal loss)
5. **Training too briefly** (25 epochs insufficient)

All of these are now fixed! Re-train from scratch to see improvements.
