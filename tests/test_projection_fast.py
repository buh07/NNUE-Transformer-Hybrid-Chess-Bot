"""
Optimized test for projection architectures with proper Stockfish handling
Avoids subprocess hanging by using game outcomes as proxy (faster)
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
from src.dataset import create_dataloaders
import config


# Test projection architectures (same as before)
class ProjectionLayerV1(nn.Module):
    """Simple projection (current implementation)"""
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV2(nn.Module):
    """Deeper with more capacity"""
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT),
            nn.Linear(2048, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV3(nn.Module):
    """Multi-layer with intermediate bottleneck"""
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, 1536),
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT),
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT),
            nn.Linear(1024, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV4(nn.Module):
    """Residual connections"""
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()
        self.expand = nn.Linear(nnue_dim, 1536)
        self.norm1 = nn.LayerNorm(1536)
        self.hidden = nn.Linear(1536, 1536)
        self.norm2 = nn.LayerNorm(1536)
        self.project = nn.Linear(1536, transformer_dim)
        self.norm3 = nn.LayerNorm(transformer_dim)
        self.dropout = nn.Dropout(config.PROJECTION_DROPOUT)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        out = self.expand(x)
        out = self.norm1(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        
        identity = out
        out = self.hidden(out)
        out = self.norm2(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = out + identity
        
        out = self.project(out)
        out = self.norm3(out)
        return out


def test_architecture(projection_class, name, nnue_model, transformer_model, val_loader, device):
    """Test a specific projection architecture"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Create projection layer
    projection = projection_class().to(device)
    projection.eval()
    
    param_count = sum(p.numel() for p in projection.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Test on validation set
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
            for board in boards:
                feat, _ = nnue_model.forward(board)
                nnue_features_list.append(feat.cpu())
            
            nnue_features = torch.stack(nnue_features_list).to(device)
            
            # Project and get transformer values
            projected_features = projection(nnue_features)
            _, transformer_values = transformer_model.forward(projected_features)
            
            # Compute value loss
            target_values_scaled = target_values * 100.0
            value_loss = value_criterion(transformer_values, target_values_scaled)
            
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_value_loss = total_value_loss / num_batches
    centipawns_error = avg_value_loss ** 0.5
    pawns_error = centipawns_error / 100.0
    
    print(f"Average Value Loss: {avg_value_loss:.2f}")
    print(f"RMSE Error: {centipawns_error:.2f} centipawns ({pawns_error:.3f} pawns)")
    
    return avg_value_loss, param_count


def main():
    print("Projection Layer Architecture Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    nnue_model = create_nnue_evaluator().to(device)
    transformer_model = create_transformer_model().to(device)
    
    nnue_model.eval()
    transformer_model.eval()
    
    # Load data - USE GAME OUTCOMES (fast, consistent)
    print("Loading validation data...")
    print("Using game outcome targets for faster testing (consistent with previous tests)")
    
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
    
    # Test architectures
    architectures = [
        (ProjectionLayerV1, "V1: Current (Simple 1024->512)"),
        (ProjectionLayerV2, "V2: Deeper (1024->2048->512)"),
        (ProjectionLayerV3, "V3: Multi-layer (1024->1536->1024->512)"),
        (ProjectionLayerV4, "V4: Residual (1024->1536->1536->512)")
    ]
    
    results = []
    for proj_class, name in architectures:
        try:
            loss, params = test_architecture(
                proj_class, name, nnue_model, transformer_model, val_loader, device
            )
            results.append((name, loss, params))
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results:
        print("No architectures tested successfully!")
        return
    
    print(f"{'Architecture':<45} {'Loss':<12} {'Params':<12}")
    print("-"*60)
    
    # Sort by loss
    for name, loss, params in sorted(results, key=lambda x: x[1]):
        print(f"{name:<45} {loss:<12.2f} {params:<12,}")
    
    # Best architecture
    best = min(results, key=lambda x: x[1])
    print("\n" + "="*60)
    print(f"BEST ARCHITECTURE: {best[0]}")
    print(f"Value Loss: {best[1]:.2f}")
    print(f"Parameters: {best[2]:,}")
    print("="*60)


if __name__ == '__main__':
    main()
