"""
Test different projection layer architectures to find the best one for reducing value loss
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


class ProjectionLayerV1(nn.Module):
    """Current architecture: Simple single-layer projection"""
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
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV2(nn.Module):
    """Deeper architecture: Add intermediate 2048-dim layer"""
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT),
            nn.Linear(2048, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV3(nn.Module):
    """Even deeper: 1024 -> 1536 -> 1024 -> 512 (bottleneck)"""
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
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)


class ProjectionLayerV4(nn.Module):
    """Residual connections for better gradient flow"""
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
        # Expand
        out = self.expand(x)
        out = self.norm1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        # Residual block
        identity = out
        out = self.hidden(out)
        out = self.norm2(out)
        out = torch.relu(out)
        out = out + identity  # Residual connection
        out = self.dropout(out)
        
        # Project to transformer dim
        out = self.project(out)
        out = self.norm3(out)
        return out


def test_architecture(projection_class, name, nnue_model, transformer_model, val_loader, device):
    """Test a projection architecture and return value loss"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    projection = projection_class().to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in projection.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Set to eval mode
    nnue_model.eval()
    transformer_model.eval()
    projection.eval()
    
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
                nnue_features_list.append(feat.cpu())  # Ensure on CPU first
                nnue_values_list.append(val)
            
            nnue_features = torch.stack(nnue_features_list).to(device)  # Then move to device
            nnue_values = torch.tensor(nnue_values_list, dtype=torch.float32).to(device)
            
            # Project and get transformer values
            projected_features = projection(nnue_features)
            _, transformer_values = transformer_model.forward(projected_features)
            
            # Blend values
            blended_values = (
                config.NNUE_VALUE_WEIGHT * nnue_values +
                config.TRANSFORMER_VALUE_WEIGHT * transformer_values
            )
            
            # Compute loss
            target_values_scaled = target_values * 100.0
            value_loss = value_criterion(blended_values, target_values_scaled)
            
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_value_loss = total_value_loss / num_batches
    centipawns_error = avg_value_loss ** 0.5  # RMSE in centipawns
    pawns_error = centipawns_error / 100.0
    
    print(f"Average Value Loss: {avg_value_loss:.2f}")
    print(f"RMSE Error: {centipawns_error:.2f} centipawns ({pawns_error:.3f} pawns)")
    
    return avg_value_loss, num_params


def main():
    print("Projection Layer Architecture Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    nnue_model = create_nnue_evaluator().to(device)
    transformer_model = create_transformer_model().to(device)
    
    # Load data (use a subset for quick testing)
    print("Loading validation data...")
    
    # Temporarily reduce dataset size for quick testing
    original_train_pos = config.MAX_TRAIN_POSITIONS
    original_val_pos = config.MAX_VAL_POSITIONS
    original_use_stockfish = config.USE_STOCKFISH_TARGETS
    
    config.MAX_TRAIN_POSITIONS = 10000
    config.MAX_VAL_POSITIONS = 5000
    
    train_files = config.PGN_FILES[:2]  # Use subset of files
    
    # Use Stockfish targets to match training (NEW: strategic routing)
    print("Using Stockfish depth-10 targets (matches training with strategic routing)")
    _, val_loader = create_dataloaders(
        train_pgn_files=train_files,
        val_pgn_files=train_files,
        use_stockfish_targets=True,
        stockfish_depth=config.STOCKFISH_TARGET_DEPTH
    )
    
    # Restore original values
    config.MAX_TRAIN_POSITIONS = original_train_pos
    config.MAX_VAL_POSITIONS = original_val_pos
    config.USE_STOCKFISH_TARGETS = original_use_stockfish
    
    # Test architectures
    architectures = [
        (ProjectionLayerV1, "V1: Current (Simple 1024->512)"),
        (ProjectionLayerV2, "V2: Deeper (1024->2048->512)"),
        (ProjectionLayerV3, "V3: Multi-layer (1024->1536->1024->512)"),
        (ProjectionLayerV4, "V4: Residual (1024->1536->1536->512)"),
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
    
    for name, loss, params in sorted(results, key=lambda x: x[1]):
        print(f"{name:<45} {loss:<12.2f} {params:<12,}")
    
    best_arch = min(results, key=lambda x: x[1])
    print("\n" + "="*60)
    print(f"BEST ARCHITECTURE: {best_arch[0]}")
    print(f"Value Loss: {best_arch[1]:.2f}")
    print(f"Parameters: {best_arch[2]:,}")
    print("="*60)


if __name__ == '__main__':
    main()
