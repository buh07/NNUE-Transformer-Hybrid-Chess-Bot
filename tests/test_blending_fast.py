"""
Fast blending weight test using game outcomes (avoids Stockfish subprocess issues)
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
    max_batches = 20  # Limit for speed
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", total=max_batches):
            if num_batches >= max_batches:
                break
                
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
    print("Value Blending Weight Optimization (Fast Version)")
    print("="*60)
    print("Using game outcomes for speed (consistent results)")
    
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
    
    config.MAX_TRAIN_POSITIONS = 10000
    config.MAX_VAL_POSITIONS = 5000
    
    train_files = config.PGN_FILES[:2]
    _, val_loader = create_dataloaders(
        train_pgn_files=train_files,
        val_pgn_files=train_files,
        use_stockfish_targets=False  # Use game outcomes for speed
    )
    
    config.MAX_TRAIN_POSITIONS = original_train_pos
    config.MAX_VAL_POSITIONS = original_val_pos
    
    # Test different blending ratios
    weight_configs = [
        (1.0, 0.0),   # Pure NNUE
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),   # Equal blend
        (0.4, 0.6),
        (0.3, 0.7),   # Current default
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),   # Pure Transformer
    ]
    
    results = []
    for nnue_w, trans_w in weight_configs:
        try:
            loss = test_blending_weights(
                nnue_w, trans_w, nnue_model, transformer_model, projection, val_loader, device
            )
            results.append((nnue_w, trans_w, loss))
        except Exception as e:
            print(f"Error testing weights ({nnue_w}, {trans_w}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results:
        print("No configurations tested successfully!")
        return
    
    print(f"{'NNUE Weight':<15} {'Transformer Weight':<20} {'Loss':<12}")
    print("-"*60)
    
    for nnue_w, trans_w, loss in sorted(results, key=lambda x: x[2]):
        print(f"{nnue_w:<15.1f} {trans_w:<20.1f} {loss:<12.2f}")
    
    best_config = min(results, key=lambda x: x[2])
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION:")
    print(f"  NNUE Weight: {best_config[0]:.1f}")
    print(f"  Transformer Weight: {best_config[1]:.1f}")
    print(f"  Value Loss: {best_config[2]:.2f}")
    print(f"  RMSE Error: {best_config[2]**0.5:.2f} centipawns ({best_config[2]**0.5/100:.3f} pawns)")
    print("="*60)
    
    # Compare to current default (0.3, 0.7) - though now using (1.0, 0.0)
    current_used = next((r for r in results if r[0] == 1.0 and r[1] == 0.0), None)
    if current_used and best_config != current_used:
        improvement = ((current_used[2] - best_config[2]) / current_used[2]) * 100
        print(f"\nImprovement over current (1.0/0.0): {improvement:.1f}%")


if __name__ == '__main__':
    main()
