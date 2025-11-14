"""
Chess Transformer Model - Interface to pre-trained transformer
"""

import torch
import torch.nn as nn
import chess
from typing import Tuple, List
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chess_utils import board_to_tensor
import config


class ChessTransformer(nn.Module):
    """
    Pre-trained transformer model for chess
    Architecture: Position encoding -> Transformer layers -> Policy/Value heads
    
    This is a simplified implementation that will interface with
    the chess-transformers repository
    """
    
    def __init__(self, weights_path: str = None):
        super(ChessTransformer, self).__init__()
        
        self.hidden_dim = config.TRANSFORMER_DIM  # 512
        self.num_moves = config.NUM_MOVES  # 1858
        
        # Position encoder - maps strategic features to transformer input
        self.position_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Transformer layers (simplified - real model would have multiple layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        # Policy head - predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.num_moves)
        )
        
        # Value head - predicts position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Try to load weights if path provided
        if weights_path and os.path.exists(weights_path):
            self.load_transformer_weights(weights_path)
        else:
            print(f"Warning: Transformer weights not found at {weights_path}")
            print("Using randomly initialized weights (placeholder)")
        
        # Freeze parameters - no training for base transformer
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def load_transformer_weights(self, path: str):
        """
        Load pre-trained transformer weights
        
        Args:
            path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)
            
            print(f"Loaded transformer weights from {path}")
        except Exception as e:
            print(f"Error loading transformer weights: {e}")
            print("Using randomly initialized weights")
    
    def forward(self, strategic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer
        
        Args:
            strategic_features: Tensor [batch_size, hidden_dim] or [hidden_dim]
                                (from projection layer)
        
        Returns:
            policy_logits: Tensor [batch_size, num_moves] or [num_moves]
            value: Tensor [batch_size] or scalar - position evaluation
        """
        # Handle single sample
        single_sample = False
        if strategic_features.dim() == 1:
            strategic_features = strategic_features.unsqueeze(0)
            single_sample = True
        
        batch_size = strategic_features.shape[0]
        
        # Add positional encoding
        encoded = self.position_encoder(strategic_features)  # [batch, hidden_dim]
        
        # Add sequence dimension for transformer (treating as seq_len=1)
        encoded = encoded.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer layers
        transformed = self.transformer_layers(encoded)  # [batch, 1, hidden_dim]
        
        # Remove sequence dimension
        transformed = transformed.squeeze(1)  # [batch, hidden_dim]
        
        # Output heads
        policy_logits = self.policy_head(transformed)  # [batch, num_moves]
        value = self.value_head(transformed).squeeze(-1)  # [batch]
        
        # Convert value from [-1, 1] to centipawns
        value = value * 1000.0  # Scale to approximately centipawns
        
        if single_sample:
            policy_logits = policy_logits.squeeze(0)
            value = value.item()
        
        return policy_logits, value
    
    def batch_forward(self, features_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch forward pass
        
        Args:
            features_batch: Tensor [batch_size, hidden_dim]
        
        Returns:
            policy_logits: Tensor [batch_size, num_moves]
            values: Tensor [batch_size]
        """
        return self.forward(features_batch)
    
    def get_policy_value(self, strategic_features: torch.Tensor,
                        legal_mask: torch.Tensor = None) -> Tuple[torch.Tensor, float]:
        """
        Get policy and value with optional legal move masking
        
        Args:
            strategic_features: Tensor [hidden_dim]
            legal_mask: Boolean tensor [num_moves] for legal moves
        
        Returns:
            policy_probs: Probability distribution over moves
            value: Position evaluation
        """
        policy_logits, value = self.forward(strategic_features)
        
        # Apply legal move masking if provided
        if legal_mask is not None:
            from utils.chess_utils import legal_softmax
            policy_probs = legal_softmax(policy_logits, legal_mask, 
                                        temperature=config.TEMPERATURE)
        else:
            # Regular softmax
            policy_probs = torch.softmax(policy_logits, dim=-1)
        
        return policy_probs, value


def create_transformer_model(weights_path: str = None) -> ChessTransformer:
    """
    Factory function to create transformer model
    
    Args:
        weights_path: Path to transformer checkpoint
    
    Returns:
        model: ChessTransformer instance
    """
    if weights_path is None:
        # Use configured weights path
        weights_path = config.TRANSFORMER_WEIGHTS_PATH
        if weights_path and os.path.exists(weights_path):
            print(f"Using configured transformer weights: {weights_path}")
        else:
            # Fallback: Try to find weights in chess-transformers directory
            transformer_dir = config.CHESS_TRANSFORMERS_DIR
            if os.path.exists(transformer_dir):
                checkpoints = os.path.join(transformer_dir, 'checkpoints')
                if os.path.exists(checkpoints):
                    # Look for .pt or .pth files
                    import glob
                    weight_files = glob.glob(os.path.join(checkpoints, '**/*.pt'), recursive=True)
                    weight_files.extend(glob.glob(os.path.join(checkpoints, '**/*.pth'), recursive=True))
                    if weight_files:
                        weights_path = weight_files[0]
                        print(f"Found transformer weights: {weights_path}")
    
    model = ChessTransformer(weights_path)
    return model


if __name__ == '__main__':
    # Test transformer model
    print("Testing Chess Transformer...")
    
    model = create_transformer_model()
    
    # Test with random strategic features
    batch_size = 4
    strategic_features = torch.randn(batch_size, config.TRANSFORMER_DIM)
    
    print(f"Input shape: {strategic_features.shape}")
    
    # Forward pass
    policy_logits, values = model.forward(strategic_features)
    
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Sample value: {values[0]:.2f} centipawns")
    
    # Test single sample
    single_features = torch.randn(config.TRANSFORMER_DIM)
    policy_logits, value = model.forward(single_features)
    
    print(f"\nSingle sample:")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value: {value:.2f} centipawns")
    
    print("\nTransformer model test complete!")
