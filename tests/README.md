# Test Suite

This directory contains all test files for the Hybrid NNUE-Transformer Chess Bot.

## Test Files

### Architecture Tests
- **`test_projection_architectures.py`** - Compare different projection layer architectures (V1-V4)
- **`test_projection_fast.py`** - Fast projection testing using game outcomes

### Blending Tests
- **`test_blending_weights.py`** - Find optimal NNUE/Transformer value blending weights
- **`test_blending_fast.py`** - Fast blending weight optimization

### Component Tests
- **`test_implementation.py`** - Test basic hybrid evaluator implementation
- **`test_pretrained_models.py`** - Verify pretrained model loading
- **`test_stockfish_eval.py`** - Test Stockfish evaluation integration
- **`test_strategic_routing.py`** - Test strategic routing/selection function

### System Tests
- **`test_search.py`** - Test search algorithm
- **`test_training.py`** - Test training pipeline

## Running Tests

From the project root directory:

```bash
# Run a specific test
python tests/test_projection_architectures.py

# Run with logging
python tests/test_blending_fast.py 2>&1 | tee logs/test_output.log
```

## Test Results

Test logs are stored in the `../logs/` directory.

## Latest Test Results (Nov 14, 2025)

### Projection Architecture Test
- **Best**: V1 Simple (1024→512) - Loss: 2244.53, RMSE: 47.38 cp
- V1 has 7x fewer parameters than V3 multi-layer

### Blending Weight Test
- **Best**: NNUE=1.0, Transformer=0.0 - Loss: 8187.97, RMSE: 90.49 cp
- Pure NNUE performs best for value estimation
