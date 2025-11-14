"""
Selection Function - Learns when to use transformer evaluation
"""

import torch
import torch.nn as nn
import chess
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chess_utils import extract_selection_features
import config


class SelectionFunction(nn.Module):
    """
    Learns to predict when transformer evaluation is beneficial
    
    Input: Fast features about position complexity (20 features)
    Output: Probability that transformer should be used
    
    This is one of the TWO trainable components (along with ProjectionLayer)
    """
    
    def __init__(self, feature_dim: int = None):
        super(SelectionFunction, self).__init__()
        
        if feature_dim is None:
            feature_dim = config.SELECTION_FEATURE_DIM
        
        self.feature_dim = feature_dim
        
        # Multi-layer perceptron with dropout
        self.network = nn.Sequential(
            nn.Linear(feature_dim, config.SELECTOR_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(config.SELECTOR_DROPOUT),
            nn.Linear(config.SELECTOR_HIDDEN_DIM_1, config.SELECTOR_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(config.SELECTOR_DROPOUT),
            nn.Linear(config.SELECTOR_HIDDEN_DIM_2, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, position_features: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of using transformer
        
        Args:
            position_features: Tensor [batch_size, feature_dim] or [feature_dim]
                Features include: move count, material balance, phase, etc.
        
        Returns:
            probability: Tensor [batch_size, 1] or scalar
                Probability that transformer should be used
        """
        single_sample = False
        if position_features.dim() == 1:
            position_features = position_features.unsqueeze(0)
            single_sample = True
        
        prob = self.network(position_features)
        
        if single_sample:
            prob = prob.squeeze(0)
        
        return prob
    
    def should_use_transformer(self, position_features: torch.Tensor, 
                              threshold: float = None) -> bool:
        """
        Binary decision for inference
        
        Args:
            position_features: Tensor [feature_dim]
            threshold: Decision threshold (default from config)
        
        Returns:
            use_transformer: Boolean decision
        """
        if threshold is None:
            threshold = config.SELECTOR_THRESHOLD
        
        with torch.no_grad():
            prob = self.forward(position_features)
            return prob.item() > threshold
    
    def should_use_transformer_batch(self, position_features: torch.Tensor,
                                    threshold: float = None) -> torch.Tensor:
        """
        Batch binary decisions
        
        Args:
            position_features: Tensor [batch_size, feature_dim]
            threshold: Decision threshold
        
        Returns:
            decisions: Boolean tensor [batch_size]
        """
        if threshold is None:
            threshold = config.SELECTOR_THRESHOLD
        
        with torch.no_grad():
            probs = self.forward(position_features)
            return probs.squeeze(-1) > threshold
    
    def predict_from_board(self, board: chess.Board, depth_remaining: int,
                          threshold: float = None) -> bool:
        """
        Predict directly from chess board
        
        Args:
            board: Chess board
            depth_remaining: Depth left in search
            threshold: Decision threshold
        
        Returns:
            use_transformer: Boolean decision
        """
        features = extract_selection_features(board, depth_remaining)
        return self.should_use_transformer(features, threshold)
    
    def save(self, path: str):
        """Save selector weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'feature_dim': self.feature_dim
        }, path)
        print(f"Saved selection function to {path}")
    
    def load(self, path: str):
        """Load selector weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded selection function from {path}")
    
    @staticmethod
    def from_checkpoint(path: str) -> 'SelectionFunction':
        """Create selector from checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        selector = SelectionFunction(feature_dim=checkpoint['feature_dim'])
        selector.load_state_dict(checkpoint['state_dict'])
        return selector


def create_selector() -> SelectionFunction:
    """
    Factory function to create selection function
    
    Returns:
        selector: SelectionFunction instance
    """
    return SelectionFunction()


if __name__ == '__main__':
    # Test selection function
    print("Testing Selection Function...")
    
    selector = create_selector()
    
    # Test with single sample
    features = torch.randn(config.SELECTION_FEATURE_DIM)
    prob = selector.forward(features)
    decision = selector.should_use_transformer(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Output probability: {prob.item():.4f}")
    print(f"Decision (use transformer): {decision}")
    
    # Test with batch
    batch_size = 8
    features_batch = torch.randn(batch_size, config.SELECTION_FEATURE_DIM)
    probs_batch = selector.forward(features_batch)
    decisions_batch = selector.should_use_transformer_batch(features_batch)
    
    print(f"\nBatch input shape: {features_batch.shape}")
    print(f"Batch output shape: {probs_batch.shape}")
    print(f"Batch decisions: {decisions_batch.sum().item()}/{batch_size} use transformer")
    
    # Test with actual chess position
    board = chess.Board()
    decision = selector.predict_from_board(board, depth_remaining=10)
    print(f"\nStarting position - Use transformer: {decision}")
    
    # Test after several moves
    board.push_san('e4')
    board.push_san('e5')
    board.push_san('Nf3')
    board.push_san('Nc6')
    decision = selector.predict_from_board(board, depth_remaining=10)
    print(f"After 1. e4 e5 2. Nf3 Nc6 - Use transformer: {decision}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    selector.save(temp_path)
    loaded_selector = SelectionFunction.from_checkpoint(temp_path)
    
    # Verify loaded weights match
    with torch.no_grad():
        original_output = selector(features)
        loaded_output = loaded_selector(features)
        diff = (original_output - loaded_output).abs().item()
    
    print(f"\nSave/Load test - Difference: {diff:.6f}")
    
    # Clean up
    os.remove(temp_path)
    
    print("\nSelection function test complete!")
