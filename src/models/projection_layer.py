"""
Projection Layer - Learnable bridge between NNUE and Transformer
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ProjectionLayer(nn.Module):
    """
    Learns to project NNUE features (1024-dim) to transformer space (512-dim)
    This is the bridge between tactical NNUE and strategic transformer
    
    This is one of the TWO trainable components (along with SelectionFunction)
    """
    
    def __init__(self, nnue_dim: int = None, transformer_dim: int = None):
        super(ProjectionLayer, self).__init__()
        
        if nnue_dim is None:
            nnue_dim = config.NNUE_FEATURE_DIM
        if transformer_dim is None:
            transformer_dim = config.TRANSFORMER_DIM
        
        self.nnue_dim = nnue_dim
        self.transformer_dim = transformer_dim
        
        # V1: Simple single-layer projection (1024->512)
        # Test results show this is optimal: 2244.53 loss vs 2264.14 for multi-layer
        # Benefits: 7x fewer parameters (525K vs 3.7M), faster, slightly better accuracy
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(config.PROJECTION_DROPOUT)
        )
        
        # Initialize with Xavier/Glorot initialization for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, nnue_features: torch.Tensor) -> torch.Tensor:
        """
        Project NNUE features to transformer space
        
        Args:
            nnue_features: Tensor [batch_size, nnue_dim] or [nnue_dim]
        
        Returns:
            transformer_features: Tensor [batch_size, transformer_dim] or [transformer_dim]
        """
        return self.projection(nnue_features)
    
    def save(self, path: str):
        """Save projection layer weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'nnue_dim': self.nnue_dim,
            'transformer_dim': self.transformer_dim
        }, path)
        print(f"Saved projection layer to {path}")
    
    def load(self, path: str):
        """Load projection layer weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded projection layer from {path}")
    
    @staticmethod
    def from_checkpoint(path: str) -> 'ProjectionLayer':
        """Create projection layer from checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        projection = ProjectionLayer(
            nnue_dim=checkpoint['nnue_dim'],
            transformer_dim=checkpoint['transformer_dim']
        )
        projection.load_state_dict(checkpoint['state_dict'])
        return projection


def create_projection_layer() -> ProjectionLayer:
    """
    Factory function to create projection layer
    
    Returns:
        projection: ProjectionLayer instance
    """
    return ProjectionLayer()


if __name__ == '__main__':
    # Test projection layer
    print("Testing Projection Layer...")
    
    projection = create_projection_layer()
    
    # Test with single sample
    nnue_features = torch.randn(config.NNUE_FEATURE_DIM)
    transformer_features = projection(nnue_features)
    
    print(f"Input shape: {nnue_features.shape}")
    print(f"Output shape: {transformer_features.shape}")
    
    # Test with batch
    batch_size = 8
    nnue_batch = torch.randn(batch_size, config.NNUE_FEATURE_DIM)
    transformer_batch = projection(nnue_batch)
    
    print(f"\nBatch input shape: {nnue_batch.shape}")
    print(f"Batch output shape: {transformer_batch.shape}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    projection.save(temp_path)
    loaded_projection = ProjectionLayer.from_checkpoint(temp_path)
    
    # Verify loaded weights match
    with torch.no_grad():
        original_output = projection(nnue_features)
        loaded_output = loaded_projection(nnue_features)
        diff = (original_output - loaded_output).abs().max().item()
    
    print(f"\nSave/Load test - Max difference: {diff:.6f}")
    
    # Clean up
    os.remove(temp_path)
    
    print("\nProjection layer test complete!")
