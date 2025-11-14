"""
Hybrid Evaluator - Combines NNUE, Transformer, Projection, and Selector
"""

import torch
import torch.nn as nn
import chess
import time
from typing import Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nnue_evaluator import NNUEEvaluator
from models.transformer_model import ChessTransformer
from models.projection_layer import ProjectionLayer
from models.selector import SelectionFunction
from utils.chess_utils import get_legal_move_mask, legal_softmax, extract_selection_features
import config


class HybridEvaluator:
    """
    Combines NNUE, projection, transformer, and selection function
    Provides unified interface for position evaluation
    """
    
    def __init__(self, nnue_model: NNUEEvaluator, 
                 transformer_model: ChessTransformer,
                 projection_layer: ProjectionLayer,
                 selector: SelectionFunction,
                 device: str = 'cpu'):
        """
        Initialize hybrid evaluator
        
        Args:
            nnue_model: Pre-trained NNUE evaluator (frozen)
            transformer_model: Pre-trained transformer (frozen)
            projection_layer: Trainable projection layer
            selector: Trainable selection function
            device: Device for computation ('cpu' or 'cuda')
        """
        self.nnue = nnue_model.to(device)
        self.transformer = transformer_model.to(device)
        self.projection = projection_layer.to(device)
        self.selector = selector.to(device)
        self.device = device
        
        # Statistics for monitoring
        self.stats = {
            'nnue_only_evals': 0,
            'hybrid_evals': 0,
            'total_time_nnue': 0.0,
            'total_time_hybrid': 0.0,
            'total_evals': 0
        }
    
    def evaluate(self, board: chess.Board, depth_remaining: int = 10,
                legal_mask: torch.Tensor = None) -> Tuple[torch.Tensor, float, str]:
        """
        Evaluate position using NNUE or hybrid approach
        
        Args:
            board: Chess board state
            depth_remaining: Depth left in search tree
            legal_mask: Boolean tensor [num_moves] indicating legal moves
        
        Returns:
            policy_probs: Probability distribution over moves [num_moves]
            value: Position evaluation score (centipawns)
            eval_method: 'nnue' or 'hybrid'
        """
        start_time = time.time()
        
        # Step 1: Always get NNUE features and value
        with torch.no_grad():
            nnue_features, nnue_value = self.nnue.forward(board)
            nnue_features = nnue_features.to(self.device)
        
        # Step 2: Decide if transformer is needed
        selection_features = extract_selection_features(board, depth_remaining)
        selection_features = selection_features.to(self.device)
        
        use_transformer = self.selector.should_use_transformer(selection_features)
        
        # Get legal mask if not provided
        if legal_mask is None:
            legal_mask = get_legal_move_mask(board)
        legal_mask = legal_mask.to(self.device)
        
        if not use_transformer:
            # Fast path: NNUE only
            self.stats['nnue_only_evals'] += 1
            
            # NNUE doesn't have policy head, so use uniform over legal moves
            policy_probs = legal_mask.float() / (legal_mask.sum() + 1e-8)
            
            elapsed = time.time() - start_time
            self.stats['total_time_nnue'] += elapsed
            self.stats['total_evals'] += 1
            
            return policy_probs, nnue_value, 'nnue'
        
        else:
            # Slow path: NNUE + Transformer
            self.stats['hybrid_evals'] += 1
            
            with torch.no_grad():
                # Project NNUE features to transformer space
                strategic_features = self.projection(nnue_features)
                
                # Get transformer policy and value
                policy_logits, transformer_value = self.transformer.forward(strategic_features)
                
                # Apply legal move masking and softmax
                policy_probs = legal_softmax(
                    policy_logits, 
                    legal_mask, 
                    temperature=config.TEMPERATURE
                )
                
                # Blend NNUE and transformer values
                blended_value = (
                    config.NNUE_VALUE_WEIGHT * nnue_value +
                    config.TRANSFORMER_VALUE_WEIGHT * transformer_value
                )
            
            elapsed = time.time() - start_time
            self.stats['total_time_hybrid'] += elapsed
            self.stats['total_evals'] += 1
            
            return policy_probs, blended_value, 'hybrid'
    
    def evaluate_batch(self, boards: list, depth_remaining: int = 10) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Batch evaluation of multiple positions
        
        Args:
            boards: List of chess boards
            depth_remaining: Depth left in search
        
        Returns:
            policy_probs_batch: Tensor [batch_size, num_moves]
            values_batch: Tensor [batch_size]
            methods: List of eval methods for each position
        """
        batch_size = len(boards)
        
        # Get NNUE features for all positions
        with torch.no_grad():
            nnue_features_list = []
            nnue_values_list = []
            for board in boards:
                feat, val = self.nnue.forward(board)
                nnue_features_list.append(feat)
                nnue_values_list.append(val)
            
            nnue_features = torch.stack(nnue_features_list).to(self.device)
            nnue_values = torch.tensor(nnue_values_list).to(self.device)
        
        # Get selection features
        selection_features_list = []
        legal_masks_list = []
        for board in boards:
            sel_feat = extract_selection_features(board, depth_remaining)
            selection_features_list.append(sel_feat)
            legal_masks_list.append(get_legal_move_mask(board))
        
        selection_features = torch.stack(selection_features_list).to(self.device)
        legal_masks = torch.stack(legal_masks_list).to(self.device)
        
        # Batch selection decisions
        use_transformer_batch = self.selector.should_use_transformer_batch(selection_features)
        
        # Initialize outputs
        policy_probs_batch = torch.zeros(batch_size, config.NUM_MOVES).to(self.device)
        values_batch = torch.zeros(batch_size).to(self.device)
        methods = []
        
        # Process NNUE-only positions
        nnue_only_mask = ~use_transformer_batch
        if nnue_only_mask.any():
            nnue_only_indices = nnue_only_mask.nonzero(as_tuple=True)[0]
            for idx in nnue_only_indices:
                policy_probs_batch[idx] = legal_masks[idx].float() / (legal_masks[idx].sum() + 1e-8)
                values_batch[idx] = nnue_values[idx]
                methods.append('nnue')
                self.stats['nnue_only_evals'] += 1
        
        # Process hybrid positions
        if use_transformer_batch.any():
            hybrid_indices = use_transformer_batch.nonzero(as_tuple=True)[0]
            
            with torch.no_grad():
                # Project NNUE features
                hybrid_nnue_features = nnue_features[hybrid_indices]
                strategic_features = self.projection(hybrid_nnue_features)
                
                # Get transformer outputs
                policy_logits, transformer_values = self.transformer.forward(strategic_features)
                
                # Process each hybrid position
                for i, idx in enumerate(hybrid_indices):
                    policy_probs_batch[idx] = legal_softmax(
                        policy_logits[i],
                        legal_masks[idx],
                        temperature=config.TEMPERATURE
                    )
                    
                    blended_value = (
                        config.NNUE_VALUE_WEIGHT * nnue_values[idx] +
                        config.TRANSFORMER_VALUE_WEIGHT * transformer_values[i]
                    )
                    values_batch[idx] = blended_value
                    methods.append('hybrid')
                    self.stats['hybrid_evals'] += 1
        
        self.stats['total_evals'] += batch_size
        
        return policy_probs_batch, values_batch, methods
    
    def print_stats(self):
        """Print evaluation statistics"""
        total = self.stats['nnue_only_evals'] + self.stats['hybrid_evals']
        
        if total == 0:
            print("No evaluations performed yet")
            return
        
        pct_hybrid = 100 * self.stats['hybrid_evals'] / total
        
        avg_time_nnue = (self.stats['total_time_nnue'] / self.stats['nnue_only_evals'] * 1000 
                        if self.stats['nnue_only_evals'] > 0 else 0)
        avg_time_hybrid = (self.stats['total_time_hybrid'] / self.stats['hybrid_evals'] * 1000
                          if self.stats['hybrid_evals'] > 0 else 0)
        
        print(f"\n=== Hybrid Evaluator Statistics ===")
        print(f"Total evaluations: {total:,}")
        print(f"  NNUE only: {self.stats['nnue_only_evals']:,} ({100-pct_hybrid:.1f}%)")
        print(f"  Hybrid: {self.stats['hybrid_evals']:,} ({pct_hybrid:.1f}%)")
        print(f"\nAverage time per evaluation:")
        print(f"  NNUE only: {avg_time_nnue:.2f} ms")
        print(f"  Hybrid: {avg_time_hybrid:.2f} ms")
        print(f"  Overall: {(self.stats['total_time_nnue'] + self.stats['total_time_hybrid']) / total * 1000:.2f} ms")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'nnue_only_evals': 0,
            'hybrid_evals': 0,
            'total_time_nnue': 0.0,
            'total_time_hybrid': 0.0,
            'total_evals': 0
        }
    
    def save(self, path: str):
        """Save trainable components"""
        torch.save({
            'projection_state_dict': self.projection.state_dict(),
            'selector_state_dict': self.selector.state_dict(),
            'stats': self.stats
        }, path)
        print(f"Saved hybrid evaluator to {path}")
    
    def load_trainable_components(self, path: str):
        """Load trainable components"""
        checkpoint = torch.load(path, map_location=self.device)
        self.projection.load_state_dict(checkpoint['projection_state_dict'])
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        print(f"Loaded trainable components from {path}")


def create_hybrid_evaluator(nnue_path: str = None, transformer_path: str = None,
                           checkpoint_path: str = None, device: str = None) -> HybridEvaluator:
    """
    Factory function to create hybrid evaluator
    
    Args:
        nnue_path: Path to NNUE weights
        transformer_path: Path to transformer weights
        checkpoint_path: Path to trained projection+selector checkpoint
        device: Computation device
    
    Returns:
        evaluator: HybridEvaluator instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    from models.nnue_evaluator import create_nnue_evaluator
    from models.transformer_model import create_transformer_model
    from models.projection_layer import create_projection_layer
    from models.selector import create_selector
    
    nnue = create_nnue_evaluator(nnue_path)
    transformer = create_transformer_model(transformer_path)
    projection = create_projection_layer()
    selector = create_selector()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        projection.load_state_dict(checkpoint['projection_state_dict'])
        selector.load_state_dict(checkpoint['selector_state_dict'])
        print(f"Loaded trained components from {checkpoint_path}")
    
    evaluator = HybridEvaluator(nnue, transformer, projection, selector, device)
    return evaluator


if __name__ == '__main__':
    # Test hybrid evaluator
    print("Testing Hybrid Evaluator...")
    
    evaluator = create_hybrid_evaluator()
    
    # Test on starting position
    board = chess.Board()
    policy, value, method = evaluator.evaluate(board)
    
    print(f"\nStarting position:")
    print(f"  Method: {method}")
    print(f"  Value: {value:.2f} cp")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Top move probability: {policy.max().item():.4f}")
    
    # Test on several positions
    print("\nEvaluating multiple positions...")
    boards = [chess.Board()]
    
    # Add some positions
    for moves in [['e4'], ['e4', 'e5'], ['e4', 'e5', 'Nf3'], ['e4', 'e5', 'Nf3', 'Nc6']]:
        b = chess.Board()
        for m in moves:
            b.push_san(m)
        boards.append(b)
    
    for i, b in enumerate(boards):
        policy, value, method = evaluator.evaluate(b)
        print(f"Position {i+1}: {method}, value={value:.2f}")
    
    # Print statistics
    evaluator.print_stats()
    
    print("\nHybrid evaluator test complete!")
