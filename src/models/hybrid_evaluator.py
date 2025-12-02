"""
Hybrid Evaluator - Combines NNUE, Transformer, Projection, and Selector
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import nullcontext
try:
    # CUDA autocast is available in recent torch
    from torch.cuda.amp import autocast
except Exception:
    # Fallback to a no-op context if autocast isn't available
    autocast = nullcontext
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
from utils.chess_utils import (
    get_legal_move_mask,
    legal_softmax,
    extract_selection_features,
    move_to_index,
)
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
                 device: str = 'cpu',
                 compatibility_scale: float = None,
                 evaluation_mode: str = 'auto'):
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
        self._piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 320,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }

        # Compatibility scale: if older checkpoints were trained with a
        # different numeric target scaling (e.g. 100x), set compatibility_scale=100
        # to automatically rescale evaluations at inference time. If None,
        # the evaluator will auto-detect implausibly large values and rescale once.
        self.compatibility_scale = compatibility_scale
        self._compat_warn_shown = False
        # Small LRU cache for recent evaluations to avoid repeated expensive
        # transformer forwards for repeated positions (keyed by FEN + depth)
        self._cache_size = getattr(config, 'EVAL_CACHE_SIZE', 1024)
        self._eval_cache = OrderedDict()
        self._eval_mode = self._normalize_mode(evaluation_mode)

        # Statistics for monitoring
        self.stats = self._init_stats()
        # Internal flag to mark whether we've seen the first transformer forward
        self._seen_first_transformer_call = False

    def _init_stats(self) -> Dict:
        """Initialize evaluation statistics dictionary."""
        return {
            'nnue_only_evals': 0,
            'hybrid_evals': 0,
            'total_time_nnue': 0.0,
            'total_time_hybrid': 0.0,
            'total_evals': 0,
            # Cache instrumentation
            'cache_hits': 0,
            'cache_misses': 0,
            # Transformer timing
            'transformer_calls': 0,
            'total_time_transformer': 0.0,
            # If the first transformer call includes compile time, it will be recorded here
            'transformer_first_call_time': None,
            'evaluation_mode': self._eval_mode
        }

    def reset_statistics(self):
        """Alias for reset_stats for backwards compatibility."""
        self.reset_stats()

    def _normalize_mode(self, mode: str) -> str:
        """Normalize evaluation mode string."""
        if not mode:
            return 'auto'
        normalized = mode.strip().lower()
        if normalized in ('auto', 'hybrid', 'default'):
            return 'auto'
        if normalized in ('nnue', 'nnue_only', 'stockfish'):
            return 'nnue'
        if normalized in ('transformer', 'transformer_only', 'chessformer'):
            return 'transformer'
        raise ValueError(f"Unknown evaluation_mode '{mode}'. Expected 'auto', 'nnue', or 'transformer'.")

    def set_evaluation_mode(self, mode: str):
        """
        Override selector decisions and force evaluation mode.
        
        Args:
            mode: 'auto', 'nnue', or 'transformer'
        """
        self._eval_mode = self._normalize_mode(mode)
        # Persist mode info in stats for downstream logging
        if hasattr(self, 'stats'):
            self.stats['evaluation_mode'] = self._eval_mode

    @property
    def evaluation_mode(self) -> str:
        """Current evaluation mode."""
        return self._eval_mode

    def _cache_mode_token(self) -> str:
        """Token used to differentiate cache entries per evaluation mode."""
        return self._eval_mode
    
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

        # Cache key: FEN + depth_remaining (simple and robust); avoids
        # repeated transformer evaluations for identical states during search.
        cache_key = (board.fen(), int(depth_remaining), self._cache_mode_token())
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            # Cache hit
            self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            self._eval_cache.move_to_end(cache_key)
            policy_cpu, value_cpu, method = cached
            # Reconstruct tensors on the requested device
            policy = torch.tensor(policy_cpu, device=self.device)
            value = torch.tensor(value_cpu, device=self.device)
            return policy, value, method
        else:
            # Record cache miss for instrumentation
            self.stats['cache_misses'] += 1
        
        # Step 1: Always get NNUE features and value
        # Use inference_mode for fastest no-grad execution
        with torch.inference_mode():
            nnue_features, nnue_value = self.nnue.forward(board)
            nnue_features = nnue_features.to(self.device)
        
        # Step 2: Decide if transformer is needed
        selection_features = extract_selection_features(board, depth_remaining)
        selection_features = selection_features.to(self.device)
        
        if self._eval_mode == 'nnue':
            use_transformer = False
        elif self._eval_mode == 'transformer':
            use_transformer = True
        else:
            use_transformer = self.selector.should_use_transformer(selection_features)
        
        # Get legal mask if not provided
        if legal_mask is None:
            legal_mask = get_legal_move_mask(board)
        legal_mask = legal_mask.to(self.device)
        
        if not use_transformer:
            # Fast path: NNUE only
            self.stats['nnue_only_evals'] += 1

            # Use a fast MVV-LVA style heuristic policy so alpha-beta has
            # sensible move ordering instead of a uniform distribution.
            policy_probs = self._heuristic_policy(board, legal_mask)
            
            elapsed = time.time() - start_time
            self.stats['total_time_nnue'] += elapsed
            self.stats['total_evals'] += 1
            # Apply compatibility scaling detection/rescaling to NNUE-only values
            nnue_value = self._maybe_rescale_value(nnue_value)

            # Cache NNUE-only result (store CPU tensors)
            try:
                if self._cache_size > 0:
                    self._eval_cache[cache_key] = (policy_probs.cpu().numpy(), float(nnue_value.detach().cpu().item()), 'nnue')
                    self._eval_cache.move_to_end(cache_key)
                    if len(self._eval_cache) > self._cache_size:
                        self._eval_cache.popitem(last=False)
            except Exception:
                # Ignore caching errors
                pass

            return policy_probs, nnue_value, 'nnue'
        
        else:
            # Slow path: NNUE + Transformer
            self.stats['hybrid_evals'] += 1

            # Use inference_mode + autocast (on CUDA) to speed up transformer
            amp_ctx = autocast if (self.device.startswith('cuda') and hasattr(torch, 'cuda')) else nullcontext
            with torch.inference_mode():
                # Project NNUE features to transformer space
                strategic_features = self.projection(nnue_features)

                # Time the transformer forward (first call may include compile time)
                t0 = time.time()
                with amp_ctx():
                    # Get transformer policy and value
                    policy_logits, transformer_value = self.transformer.forward(strategic_features)
                t1 = time.time()
                dt = t1 - t0
                # Update transformer timing statistics
                self.stats['total_time_transformer'] += dt
                # Count this as one transformer forward call (batch counts as one)
                self.stats['transformer_calls'] += 1
                if not self._seen_first_transformer_call:
                    # Record first call time (useful to see compile+warmup cost)
                    self.stats['transformer_first_call_time'] = dt
                    self._seen_first_transformer_call = True

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
                # Apply compatibility scaling detection/rescaling
                blended_value = self._maybe_rescale_value(blended_value)

            elapsed = time.time() - start_time
            self.stats['total_time_hybrid'] += elapsed
            self.stats['total_evals'] += 1

            # Cache hybrid result
            try:
                if self._cache_size > 0:
                    self._eval_cache[cache_key] = (policy_probs.cpu().numpy(), float(blended_value.detach().cpu().item()), 'hybrid')
                    self._eval_cache.move_to_end(cache_key)
                    if len(self._eval_cache) > self._cache_size:
                        self._eval_cache.popitem(last=False)
            except Exception:
                pass

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
        with torch.inference_mode():
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
            
            # Use inference_mode + autocast for transformer forwards
            amp_ctx = autocast if (self.device.startswith('cuda') and hasattr(torch, 'cuda')) else nullcontext
            with torch.inference_mode():
                # Project NNUE features
                hybrid_nnue_features = nnue_features[hybrid_indices]
                strategic_features = self.projection(hybrid_nnue_features)

                # Time the batched transformer forward
                t0 = time.time()
                with amp_ctx():
                    # Get transformer outputs
                    policy_logits, transformer_values = self.transformer.forward(strategic_features)
                t1 = time.time()
                dt = t1 - t0
                self.stats['total_time_transformer'] += dt
                self.stats['transformer_calls'] += 1
                if not self._seen_first_transformer_call:
                    self.stats['transformer_first_call_time'] = dt
                    self._seen_first_transformer_call = True
                
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
        # Cache stats
        print(f"\nCache statistics:")
        print(f"  Hits: {self.stats.get('cache_hits', 0):,}")
        print(f"  Misses: {self.stats.get('cache_misses', 0):,}")
        # Transformer timings
        if self.stats.get('transformer_calls', 0) > 0:
            avg_transformer = self.stats['total_time_transformer'] / self.stats['transformer_calls'] * 1000
            first_t = self.stats.get('transformer_first_call_time')
            first_str = f"{first_t:.3f}s" if first_t is not None else "n/a"
            print(f"\nTransformer timings:")
            print(f"  Calls: {self.stats['transformer_calls']:,}")
            print(f"  Avg per forward: {avg_transformer:.2f} ms")
            print(f"  First forward (compile+warmup) time: {first_str}")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = self._init_stats()
        self._seen_first_transformer_call = False

    def get_stats(self) -> Dict:
        """Return a shallow copy of the current stats dictionary."""
        return dict(self.stats)

    def _heuristic_policy(self, board: chess.Board, legal_mask: torch.Tensor) -> torch.Tensor:
        """
        Produce a lightweight policy distribution using basic tactical heuristics.
        """
        scores = torch.zeros(config.NUM_MOVES, device=self.device)
        legal_moves = list(board.legal_moves)

        for move in legal_moves:
            idx = move_to_index(move)
            score = 1.0

            piece = board.piece_at(move.from_square)
            captured = board.piece_at(move.to_square)

            if captured is not None:
                captured_value = self._piece_values.get(captured.piece_type, 0)
                attacker_value = self._piece_values.get(piece.piece_type, 0) if piece else 0
                score += 1.0 + (captured_value - attacker_value) / 100.0

            if move.promotion is not None:
                score += 1.5 + self._piece_values.get(move.promotion, 0) / 200.0

            board.push(move)
            if board.is_check():
                score += 0.5
            board.pop()

            scores[idx] = score

        scores = scores * legal_mask.float()
        total = scores.sum()
        if total <= 0:
            # Fallback to uniform over legal moves
            return legal_mask.float() / (legal_mask.sum() + 1e-8)
        return scores / total

    def _maybe_rescale_value(self, value):
        """
        Detect and rescale implausibly large evaluation outputs produced by
        legacy checkpoints that used a different numeric target scaling.

        If `self.compatibility_scale` is provided (>1.0) this scale is applied
        deterministically. Otherwise a heuristic is used: if the absolute
        magnitude of the value tensor is > 10, assume legacy 100x scaling and
        rescale by 100.
        """
        try:
            # Tensor branch
            if isinstance(value, torch.Tensor):
                # compute max absolute magnitude
                if value.numel() == 0:
                    return value
                max_abs = float(value.abs().max().detach().cpu().item())

                if self.compatibility_scale and self.compatibility_scale > 1.0:
                    if not self._compat_warn_shown:
                        print(f"[WARNING] Applying compatibility rescale by {self.compatibility_scale} to evaluator outputs.")
                        self._compat_warn_shown = True
                    return value / float(self.compatibility_scale)

                if max_abs > 10.0:
                    if not self._compat_warn_shown:
                        print("[WARNING] Detected large evaluator outputs (|value| > 10). Assuming legacy 100x scaling. Rescaling outputs by /100 for inference.")
                        self._compat_warn_shown = True
                    return value / 100.0

                return value

            # Numeric branch
            else:
                abs_val = abs(value)
                if self.compatibility_scale and self.compatibility_scale > 1.0:
                    if not self._compat_warn_shown:
                        print(f"[WARNING] Applying compatibility rescale by {self.compatibility_scale} to evaluator outputs.")
                        self._compat_warn_shown = True
                    return value / float(self.compatibility_scale)

                if abs_val > 10.0:
                    if not self._compat_warn_shown:
                        print("[WARNING] Detected large evaluator outputs (|value| > 10). Assuming legacy 100x scaling. Rescaling outputs by /100 for inference.")
                        self._compat_warn_shown = True
                    return value / 100.0

                return value
        except Exception:
            # Fall back to returning original value on any unexpected error
            return value
    
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
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
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
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
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
