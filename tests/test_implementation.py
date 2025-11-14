#!/usr/bin/env python3
"""
Test script to verify the Hybrid NNUE-Transformer implementation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import chess

print("=" * 70)
print("HYBRID NNUE-TRANSFORMER CHESS ENGINE - COMPONENT TESTS")
print("=" * 70)

# Test 1: Configuration
print("\n[1/8] Testing configuration...")
try:
    import config
    print(f"  ✓ Config loaded")
    print(f"    - NNUE dim: {config.NNUE_FEATURE_DIM}")
    print(f"    - Transformer dim: {config.TRANSFORMER_DIM}")
    print(f"    - Batch size: {config.BATCH_SIZE}")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    sys.exit(1)

# Test 2: Utilities
print("\n[2/8] Testing utilities...")
try:
    from utils.chess_utils import (
        move_to_index, get_legal_move_mask, 
        extract_selection_features, legal_softmax
    )
    
    board = chess.Board()
    mask = get_legal_move_mask(board)
    features = extract_selection_features(board, 10)
    
    move = chess.Move.from_uci('e2e4')
    idx = move_to_index(move)
    
    print(f"  ✓ Utilities working")
    print(f"    - Legal moves in start position: {mask.sum().item()}")
    print(f"    - Selection features shape: {features.shape}")
    print(f"    - Move e2e4 index: {idx}")
except Exception as e:
    print(f"  ✗ Utilities failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: NNUE Evaluator
print("\n[3/8] Testing NNUE evaluator...")
try:
    from models.nnue_evaluator import create_nnue_evaluator
    
    nnue = create_nnue_evaluator()
    board = chess.Board()
    features, value = nnue.forward(board)
    
    print(f"  ✓ NNUE evaluator working")
    print(f"    - Features shape: {features.shape}")
    print(f"    - Start position value: {value:.2f} cp")
except Exception as e:
    print(f"  ✗ NNUE evaluator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Transformer Model
print("\n[4/8] Testing transformer model...")
try:
    from models.transformer_model import create_transformer_model
    
    transformer = create_transformer_model()
    strategic_features = torch.randn(config.TRANSFORMER_DIM)
    policy_logits, value = transformer.forward(strategic_features)
    
    print(f"  ✓ Transformer model working")
    print(f"    - Policy logits shape: {policy_logits.shape}")
    print(f"    - Value: {value:.2f} cp")
except Exception as e:
    print(f"  ✗ Transformer model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Projection Layer
print("\n[5/8] Testing projection layer...")
try:
    from models.projection_layer import create_projection_layer
    
    projection = create_projection_layer()
    nnue_features = torch.randn(config.NNUE_FEATURE_DIM)
    projected = projection(nnue_features)
    
    print(f"  ✓ Projection layer working")
    print(f"    - Input shape: {nnue_features.shape}")
    print(f"    - Output shape: {projected.shape}")
    print(f"    - Trainable parameters: {sum(p.numel() for p in projection.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"  ✗ Projection layer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Selector
print("\n[6/8] Testing selection function...")
try:
    from models.selector import create_selector
    
    selector = create_selector()
    board = chess.Board()
    decision = selector.predict_from_board(board, depth_remaining=10)
    
    print(f"  ✓ Selector working")
    print(f"    - Start position decision: {'use transformer' if decision else 'use NNUE only'}")
    print(f"    - Trainable parameters: {sum(p.numel() for p in selector.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"  ✗ Selector failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Hybrid Evaluator
print("\n[7/8] Testing hybrid evaluator...")
try:
    from models.hybrid_evaluator import create_hybrid_evaluator
    
    evaluator = create_hybrid_evaluator()
    board = chess.Board()
    policy, value, method = evaluator.evaluate(board)
    
    print(f"  ✓ Hybrid evaluator working")
    print(f"    - Evaluation method: {method}")
    print(f"    - Value: {value:.2f} cp")
    print(f"    - Policy shape: {policy.shape}")
    print(f"    - Top move probability: {policy.max().item():.4f}")
    
    # Test multiple positions
    boards = []
    for moves in [['e4'], ['e4', 'e5'], ['e4', 'e5', 'Nf3']]:
        b = chess.Board()
        for m in moves:
            b.push_san(m)
        boards.append(b)
    
    for b in boards:
        _, _, method = evaluator.evaluate(b)
    
    evaluator.print_stats()
    
except Exception as e:
    print(f"  ✗ Hybrid evaluator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Dataset
print("\n[8/8] Testing dataset...")
try:
    from dataset import create_dummy_dataset
    from torch.utils.data import DataLoader
    from dataset import collate_fn
    
    dataset = create_dummy_dataset(num_positions=50)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    
    print(f"  ✓ Dataset working")
    print(f"    - Dataset size: {len(dataset)}")
    print(f"    - Batch size: {len(batch['boards'])}")
    print(f"    - Selection features shape: {batch['selection_features'].shape}")
    
except Exception as e:
    print(f"  ✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nThe hybrid model implementation is ready.")
print("\nNext steps:")
print("  1. Obtain PGN files with high-quality games")
print("  2. Run training: python src/train.py (to be implemented)")
print("  3. Test the trained model: python src/play.py (to be implemented)")
print("\nFor now, the model uses randomly initialized trainable components.")
print("NNUE and Transformer models are placeholder implementations.")
