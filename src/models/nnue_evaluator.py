"""
NNUE Evaluator - Interface to Stockfish NNUE model
"""

import torch
import torch.nn as nn
import chess
import chess.engine
import numpy as np
from typing import Tuple, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chess_utils import extract_halfka_features
import config


class NNUEEvaluator(nn.Module):
    """
    Pre-trained NNUE model (e.g., from Stockfish)
    Architecture: HalfKAv2 features -> Feature Transformer -> Hidden layers
    
    For now, this is a simplified placeholder that will need to be connected
    to actual Stockfish NNUE weights or use python-chess-nnue library
    """
    
    def __init__(self, nnue_weights_path: str = None, use_stockfish_engine: bool = False):
        super(NNUEEvaluator, self).__init__()
        
        self.feature_dim = 45056  # HalfKAv2 input dimension (sparse)
        self.accumulator_dim = 512  # Per-side accumulator
        self.output_dim = config.NNUE_FEATURE_DIM  # 1024 (both accumulators)
        self.use_stockfish_engine = use_stockfish_engine
        self.engine: Optional[chess.engine.SimpleEngine] = None
        
        # Feature transformer (sparse -> dense accumulator)
        # This would normally load from NNUE binary file
        self.feature_transformer = nn.Linear(self.feature_dim, self.accumulator_dim, bias=True)
        
        # Hidden layers for value computation
        self.hidden1 = nn.Linear(self.output_dim, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Try to use Stockfish engine if requested and available
        if use_stockfish_engine and hasattr(config, 'STOCKFISH_BINARY_PATH'):
            if os.path.exists(config.STOCKFISH_BINARY_PATH):
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_BINARY_PATH)
                    print(f"✓ Using Stockfish engine for NNUE evaluations: {config.STOCKFISH_BINARY_PATH}")
                except Exception as e:
                    print(f"⚠ Failed to load Stockfish engine: {e}")
                    print("  Falling back to learned feature representation")
                    self.use_stockfish_engine = False
            else:
                print(f"⚠ Stockfish binary not found at {config.STOCKFISH_BINARY_PATH}")
                print("  Falling back to learned feature representation")
                self.use_stockfish_engine = False
        
        # Try to load weights if path provided (for learned representation)
        if not self.use_stockfish_engine:
            if nnue_weights_path and os.path.exists(nnue_weights_path):
                self.load_nnue_weights(nnue_weights_path)
            else:
                print(f"⚠ NNUE file: {os.path.basename(nnue_weights_path) if nnue_weights_path else 'Not specified'}")
                print("  Using learned feature representation (will be trained)")
        
        # Freeze parameters - no training for NNUE
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def load_nnue_weights(self, path: str):
        """
        Load NNUE weights from Stockfish binary format
        
        Note: This is a placeholder. Real implementation would need:
        - Binary parser for Stockfish NNUE format
        - Or use python-chess with NNUE support
        - Or convert NNUE to PyTorch format
        """
        print(f"TODO: Implement NNUE weight loading from {path}")
        print("For now using placeholder weights")
        # Future: Parse binary NNUE file and load weights
        pass
    
    def compute_accumulator(self, board: chess.Board) -> torch.Tensor:
        """
        Compute NNUE accumulator for both sides
        
        Args:
            board: Chess position
            
        Returns:
            accumulator: Tensor [1024] (512 white + 512 black)
        """
        # Extract sparse features for both sides
        white_features = extract_halfka_features(board, chess.WHITE)
        black_features = extract_halfka_features(board, chess.BLACK)
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Convert to dense representation (sparse would be more efficient)
        white_input = torch.zeros(self.feature_dim, device=device)
        black_input = torch.zeros(self.feature_dim, device=device)
        
        for idx in white_features:
            if idx < self.feature_dim:
                white_input[idx] = 1.0
        
        for idx in black_features:
            if idx < self.feature_dim:
                black_input[idx] = 1.0
        
        # Feature transformer (incremental update would be faster)
        white_accumulator = self.activation(self.feature_transformer(white_input))
        black_accumulator = self.activation(self.feature_transformer(black_input))
        
        # Concatenate both accumulators
        accumulator = torch.cat([white_accumulator, black_accumulator])
        
        return accumulator
    
    def forward(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Evaluate position with NNUE
        
        Args:
            board: Chess position
            
        Returns:
            features: Tensor [1024] (accumulator output)
            value: Scalar evaluation score (centipawns)
        """
        # Get accumulator features (always computed for consistency)
        features = self.compute_accumulator(board)
        
        # Get value from Stockfish engine if available, otherwise use network
        if self.use_stockfish_engine and self.engine is not None:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(depth=10))
                score = info["score"].relative
                if score.is_mate():
                    # Convert mate scores to large centipawn values
                    value = 10000.0 if score.mate() > 0 else -10000.0
                else:
                    value = float(score.score())
            except Exception as e:
                # Fallback to network evaluation if engine fails
                value = self._evaluate_with_network(features)
        else:
            value = self._evaluate_with_network(features)
        
        return features, value
    
    def _evaluate_with_network(self, features: torch.Tensor) -> float:
        """Evaluate using the neural network"""
        x = self.activation(self.hidden1(features))
        x = self.activation(self.hidden2(x))
        value = self.output_layer(x).item()
        # Scale to centipawns (approximately)
        return value * 100.0
    
    def incremental_update(self, accumulator: torch.Tensor, 
                          move: chess.Move, board: chess.Board) -> torch.Tensor:
        """
        Fast incremental update when position changes by one move
        
        This is much faster than full recomputation but requires
        tracking which features changed
        
        Args:
            accumulator: Current accumulator [1024]
            move: Move being made
            board: Board before move
            
        Returns:
            updated_accumulator: New accumulator [1024]
        """
        # For simplicity, recompute for now
        # Real implementation would track changed features and update incrementally
        board_copy = board.copy()
        board_copy.push(move)
        return self.compute_accumulator(board_copy)
    
    def batch_forward(self, boards: List[chess.Board]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch evaluation for multiple positions
        
        Args:
            boards: List of chess positions
            
        Returns:
            features: Tensor [batch_size, 1024]
            values: Tensor [batch_size]
        """
        features_list = []
        values_list = []
        
        for board in boards:
            feat, val = self.forward(board)
            features_list.append(feat)
            values_list.append(val)
        
        features = torch.stack(features_list)
        values = torch.tensor(values_list, dtype=torch.float32)
        
        return features, values


    def __del__(self):
        """Cleanup Stockfish engine when object is destroyed"""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass


def create_nnue_evaluator(weights_path: str = None, use_stockfish: bool = False) -> NNUEEvaluator:
    """
    Factory function to create NNUE evaluator
    
    Args:
        weights_path: Path to NNUE weights file
        use_stockfish: Whether to use Stockfish engine for evaluations (slower but accurate)
        
    Returns:
        evaluator: NNUEEvaluator instance
    """
    if weights_path is None:
        weights_path = config.STOCKFISH_NNUE_PATH
    
    evaluator = NNUEEvaluator(weights_path, use_stockfish_engine=use_stockfish)
    return evaluator


if __name__ == '__main__':
    # Test NNUE evaluator
    print("Testing NNUE Evaluator...")
    
    evaluator = create_nnue_evaluator()
    
    # Test on starting position
    board = chess.Board()
    features, value = evaluator.forward(board)
    
    print(f"Starting position:")
    print(f"  Features shape: {features.shape}")
    print(f"  Value: {value:.2f} centipawns")
    
    # Test on position after e4
    board.push_san('e4')
    features, value = evaluator.forward(board)
    print(f"\nAfter 1. e4:")
    print(f"  Features shape: {features.shape}")
    print(f"  Value: {value:.2f} centipawns")
    
    print("\nNNUE Evaluator test complete!")
