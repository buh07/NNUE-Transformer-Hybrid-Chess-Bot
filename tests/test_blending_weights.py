"""
Test different value blending weights to find optimal NNUE vs Transformer balance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.models.nnue_evaluator import create_nnue_evaluator
from src.models.transformer_model import create_transformer_model
from src.models.projection_layer import create_projection_layer
from src.dataset import create_dataloaders
import config


def test_blending_weights(nnue_weight, transformer_weight, nnue_model, transformer_model, projection, val_loader, device):
    """Test a specific blending weight configuration"""
    
    # Ensure weights sum to 1.0
    total = nnue_weight + transformer_weight
    nnue_w = nnue_weight / total
    trans_w = transformer_weight / total
    
    print(f"\n{'='*60}")
    print(f"Testing: NNUE={nnue_w:.1f}, Transformer={trans_w:.1f}")
    print(f"{'='*60}")
    
    value_criterion = nn.MSELoss()
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            boards = batch['boards']
            target_values = batch['target_values'].to(device)
            
            # Get NNUE features
            nnue_features_list = []
            nnue_values_list = []
            for board in boards:
                feat, val = nnue_model.forward(board)
                nnue_features_list.append(feat.cpu())
                nnue_values_list.append(val)
            
            nnue_features = torch.stack(nnue_features_list).to(device)
            nnue_values = torch.tensor(nnue_values_list, dtype=torch.float32).to(device)
            
            # Project and get transformer values
            projected_features = projection(nnue_features)
            _, transformer_values = transformer_model.forward(projected_features)
            
            # Blend values with test weights
            blended_values = nnue_w * nnue_values + trans_w * transformer_values
            
            # Compute loss
            target_values_scaled = target_values * 100.0
            value_loss = value_criterion(blended_values, target_values_scaled)
            
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_value_loss = total_value_loss / num_batches
    centipawns_error = avg_value_loss ** 0.5
    pawns_error = centipawns_error / 100.0
    
    print(f"Average Value Loss: {avg_value_loss:.2f}")
    print(f"RMSE Error: {centipawns_error:.2f} centipawns ({pawns_error:.3f} pawns)")
    
    return avg_value_loss


def main():
    print("Value Blending Weight Optimization")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    nnue_model = create_nnue_evaluator().to(device)
    transformer_model = create_transformer_model().to(device)
    projection = create_projection_layer().to(device)
    
    # Set to eval mode
    nnue_model.eval()
    transformer_model.eval()
    projection.eval()
    
    # Load data (small subset for quick testing)
    print("Loading validation data...")
    original_train_pos = config.MAX_TRAIN_POSITIONS
    original_val_pos = config.MAX_VAL_POSITIONS
    original_use_stockfish = config.USE_STOCKFISH_TARGETS
    
    config.MAX_TRAIN_POSITIONS = 10000
    config.MAX_VAL_POSITIONS = 5000
    
    train_files = config.PGN_FILES[:2]
    
    # Test with BOTH target types to check for bias
    print("\n" + "="*60)
    print("Testing with Stockfish targets (matches training):")
    print("="*60)
    config.USE_STOCKFISH_TARGETS = True
    _, val_loader_stockfish = create_dataloaders(
        train_pgn_files=train_files,
        val_pgn_files=train_files,
        use_stockfish_targets=True,
        stockfish_depth=config.STOCKFISH_TARGET_DEPTH
    )
    
    print("\n" + "="*60)
    print("Testing with game outcome targets (for comparison):")
    print("="*60)
    config.USE_STOCKFISH_TARGETS = False
    _, val_loader_gameoutcome = create_dataloaders(
        train_pgn_files=train_files,
        val_pgn_files=train_files,
        use_stockfish_targets=False
    )
    
    config.MAX_TRAIN_POSITIONS = original_train_pos
    config.MAX_VAL_POSITIONS = original_val_pos
    config.USE_STOCKFISH_TARGETS = original_use_stockfish
    
    # Test different blending ratios
    # Format: (NNUE_weight, Transformer_weight)
    weight_configs = [
        (1.0, 0.0),   # Pure NNUE
        (0.8, 0.2),   # 80% NNUE, 20% Transformer
        (0.7, 0.3),   # 70% NNUE, 30% Transformer
        (0.6, 0.4),   # 60% NNUE, 40% Transformer
        (0.5, 0.5),   # Equal blend
        (0.4, 0.6),   # 40% NNUE, 60% Transformer
        (0.3, 0.7),   # Current default (30% NNUE, 70% Transformer)
        (0.2, 0.8),   # 20% NNUE, 80% Transformer
        (0.1, 0.9),   # 10% NNUE, 90% Transformer
        (0.0, 1.0),   # Pure Transformer
    ]
    
    # Test with Stockfish targets first (matches training)
    print("\n" + "="*60)
    print("TESTING WITH STOCKFISH TARGETS (Primary - Matches Training)")
    print("="*60)
    results_stockfish = []
    for nnue_w, trans_w in weight_configs:
        try:
            loss = test_blending_weights(
                nnue_w, trans_w, nnue_model, transformer_model, projection, val_loader_stockfish, device
            )
            results_stockfish.append((nnue_w, trans_w, loss))
        except Exception as e:
            print(f"Error testing weights ({nnue_w}, {trans_w}): {e}")
            continue
    
    # Test with game outcome targets for comparison
    print("\n" + "="*60)
    print("TESTING WITH GAME OUTCOME TARGETS (Secondary - For Comparison)")
    print("="*60)
    results_gameoutcome = []
    for nnue_w, trans_w in weight_configs:
        try:
            loss = test_blending_weights(
                nnue_w, trans_w, nnue_model, transformer_model, projection, val_loader_gameoutcome, device
            )
            results_gameoutcome.append((nnue_w, trans_w, loss))
        except Exception as e:
            print(f"Error testing weights ({nnue_w}, {trans_w}): {e}")
            continue
    
    # Print summaries
    print("\n" + "="*60)
    print("SUMMARY - STOCKFISH TARGETS (Use this for training decisions)")
    print("="*60)
    
    if not results_stockfish:
        print("No configurations tested successfully!")
    else:
        print(f"{'NNUE Weight':<15} {'Transformer Weight':<20} {'Loss':<12}")
        print("-"*60)
        
        for nnue_w, trans_w, loss in sorted(results_stockfish, key=lambda x: x[2]):
            print(f"{nnue_w:<15.1f} {trans_w:<20.1f} {loss:<12.2f}")
        
        best_stockfish = min(results_stockfish, key=lambda x: x[2])
        print("\n" + "="*60)
        print("BEST CONFIGURATION (Stockfish targets):")
        print(f"  NNUE Weight: {best_stockfish[0]}")
        print(f"  Transformer Weight: {best_stockfish[1]}")
        print(f"  Value Loss: {best_stockfish[2]:.2f}")
        print("="*60)
    
    print("\n" + "="*60)
    print("SUMMARY - GAME OUTCOME TARGETS (For comparison only)")
    print("="*60)
    
    if not results_gameoutcome:
        print("No configurations tested successfully!")
    else:
        print(f"{'NNUE Weight':<15} {'Transformer Weight':<20} {'Loss':<12}")
        print("-"*60)
        
        for nnue_w, trans_w, loss in sorted(results_gameoutcome, key=lambda x: x[2]):
            print(f"{nnue_w:<15.1f} {trans_w:<20.1f} {loss:<12.2f}")
        
        best_gameoutcome = min(results_gameoutcome, key=lambda x: x[2])
        print("\n" + "="*60)
        print("BEST CONFIGURATION (Game outcome targets):")
        print(f"  NNUE Weight: {best_gameoutcome[0]}")
        print(f"  Transformer Weight: {best_gameoutcome[1]}")
        print(f"  Value Loss: {best_gameoutcome[2]:.2f}")
        print("="*60)
        
        # Compare the two approaches
        if results_stockfish and results_gameoutcome:
            print("\n" + "="*60)
            print("COMPARISON BETWEEN TARGET TYPES:")
            print("="*60)
            print(f"Best with Stockfish targets: NNUE={best_stockfish[0]}, Trans={best_stockfish[1]}, Loss={best_stockfish[2]:.2f}")
            print(f"Best with game outcomes:      NNUE={best_gameoutcome[0]}, Trans={best_gameoutcome[1]}, Loss={best_gameoutcome[2]:.2f}")
            
            if best_stockfish[0] != best_gameoutcome[0]:
                print("\n⚠️  WARNING: Different optimal blending weights for different target types!")
                print("    This suggests there IS bias. Use Stockfish targets result since that matches training.")
            else:
                print("\n✓ Both target types agree on optimal weights - less bias concern.")


if __name__ == '__main__':
    main()
