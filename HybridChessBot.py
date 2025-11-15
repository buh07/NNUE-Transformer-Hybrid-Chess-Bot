"""
Hybrid Chess Bot - Uses trained NNUE-Transformer hybrid architecture
Compatible with ChessGame.py requirements
"""

import chess
import torch
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.nnue_evaluator import create_nnue_evaluator
from models.transformer_model import create_transformer_model
from models.projection_layer import create_projection_layer
from models.selector import create_selector
from models.hybrid_evaluator import HybridEvaluator
from search import AlphaBetaSearch
from time_manager import TimeManager
import config


class HybridChessBot:
    """
    Chess bot using trained hybrid NNUE-Transformer architecture.
    
    Features:
    - Adaptive evaluation (NNUE for tactics, Transformer for strategy)
    - Alpha-beta search with transposition table
    - Move ordering and quiescence search
    - 97.56% accurate position type classification
    
    Usage:
        bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=5)
        move = bot.choose_move(board)
    """
    
    def __init__(self, checkpoint: str = 'checkpoints/best_phase2.pt', 
                 depth: int = 5, time_limit: float = 5.0,
                 device: str = None, verbose: bool = False,
                 use_time_management: bool = False,
                 total_game_time: float = None):
        """
        Initialize the hybrid chess bot.
        
        Args:
            checkpoint: Path to trained model checkpoint (relative to project root)
            depth: Maximum search depth (default: 5)
            time_limit: Time limit per move in seconds (default: 5.0)
            device: 'cuda' or 'cpu' (auto-detect if None)
            verbose: Print search statistics
            use_time_management: Enable dynamic time allocation (default: False)
            total_game_time: Total time for game in seconds (required if use_time_management=True)
        """
        self.verbose = verbose
        self.depth = depth
        self.time_limit = time_limit
        self.use_time_management = use_time_management
        
        # Initialize time manager if requested
        if use_time_management:
            if total_game_time is None:
                raise ValueError("total_game_time must be specified when use_time_management=True")
            self.time_manager = TimeManager(total_time=total_game_time)
        else:
            self.time_manager = None
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if self.verbose:
            print(f"[HybridChessBot] Initializing on device: {self.device}")
        
        # Convert checkpoint path to absolute if relative
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(os.path.dirname(__file__), checkpoint)
        
        # Load models
        self._load_models(checkpoint)
        
        # Initialize search engine
        self.search_engine = AlphaBetaSearch(
            hybrid_evaluator=self.hybrid_evaluator,
            max_depth=self.depth,
            tt_size=1000000,
            use_quiescence=True
        )
        
        if self.verbose:
            print(f"[HybridChessBot] Ready! Max depth: {self.depth}")
            print(f"[HybridChessBot] Projection params: {self._count_params(self.projection):,}")
            print(f"[HybridChessBot] Selector params: {self._count_params(self.selector):,}")
    
    def _load_models(self, checkpoint_path: str):
        """Load all model components and trained weights."""
        if self.verbose:
            print(f"[HybridChessBot] Loading checkpoint: {checkpoint_path}")
        
        # Create base models (frozen, pre-trained)
        self.nnue = create_nnue_evaluator()
        self.transformer = create_transformer_model()
        
        # Create trainable components (use config defaults)
        self.projection = create_projection_layer()
        self.selector = create_selector()
        
        # Load trained weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load projection and selector weights
            if 'projection_state_dict' in checkpoint:
                self.projection.load_state_dict(checkpoint['projection_state_dict'])
                if self.verbose:
                    print(f"[HybridChessBot] Loaded projection weights")
            
            if 'selector_state_dict' in checkpoint:
                self.selector.load_state_dict(checkpoint['selector_state_dict'])
                if self.verbose:
                    print(f"[HybridChessBot] Loaded selector weights")
            
            # Print training info if available
            if self.verbose and 'epoch' in checkpoint:
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', 'unknown')
                selector_acc = checkpoint.get('selector_accuracy', 'unknown')
                print(f"[HybridChessBot] Checkpoint: epoch {epoch}")
                print(f"[HybridChessBot] Validation loss: {val_loss}")
                print(f"[HybridChessBot] Selector accuracy: {selector_acc}%")
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            print(f"[WARNING] Using untrained weights!")
        
        # Create hybrid evaluator
        self.hybrid_evaluator = HybridEvaluator(
            nnue_model=self.nnue,
            transformer_model=self.transformer,
            projection_layer=self.projection,
            selector=self.selector,
            device=self.device
        )
        
        # Set to evaluation mode
        self.projection.eval()
        self.selector.eval()
    
    def _count_params(self, model) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Choose the best move for the current position.
        
        This is the required interface method for ChessGame compatibility.
        
        Args:
            board: chess.Board object representing current game state
            
        Returns:
            chess.Move object representing the chosen move
        """
        # Check if game is over (shouldn't happen, but safe)
        if board.is_game_over():
            # Return any legal move (shouldn't be called)
            return list(board.legal_moves)[0] if board.legal_moves else None
        
        # Determine time allocation
        if self.use_time_management and self.time_manager is not None:
            import time
            move_start_time = time.time()
            
            # Get dynamic time allocation based on position complexity
            time_alloc = self.time_manager.allocate_time(
                board=board,
                selector_model=self.selector,
                depth_remaining=self.depth
            )
            
            time_for_move = time_alloc['allocated_time']
            
            if self.verbose:
                print(f"\n[Time Management]")
                print(f"  Allocated time: {time_for_move:.2f}s")
                print(f"  Base time: {time_alloc['base_time']:.2f}s")
                print(f"  Complexity: {time_alloc['complexity_score']:.2f}")
                print(f"  Selector confidence: {time_alloc['selector_confidence']:.2f}")
                print(f"  Selector wants transformer: {time_alloc['selector_probability']:.2f}")
                print(f"  Time remaining: {self.time_manager.total_time:.1f}s")
                print(f"  Est. moves remaining: {time_alloc['estimated_moves_remaining']}")
        else:
            time_for_move = self.time_limit
            move_start_time = None
        
        # Run iterative deepening search with time limit
        best_move, score = self.search_engine.iterative_deepening(
            board=board,
            max_depth=self.depth,
            time_limit=time_for_move
        )
        
        # Update time manager
        if self.use_time_management and move_start_time is not None:
            import time
            time_used = time.time() - move_start_time
            self.time_manager.update_after_move(time_used)
        
        # Print statistics if verbose
        if self.verbose:
            stats = self.search_engine.get_statistics()
            eval_stats = self.hybrid_evaluator.stats
            
            print(f"\n[Search Statistics]")
            print(f"  Move: {best_move.uci()} (score: {score:.2f})")
            print(f"  Nodes: {stats['nodes_searched']:,}")
            print(f"  Time: {stats['time_elapsed']:.2f}s")
            print(f"  NPS: {stats['nps']:,.0f}")
            print(f"  TT hit rate: {stats['tt_hit_rate']*100:.1f}%")
            print(f"  NNUE evals: {eval_stats['nnue_only_evals']:,}")
            print(f"  Hybrid evals: {eval_stats['hybrid_evals']:,}")
            if eval_stats['total_evals'] > 0:
                transformer_pct = 100 * eval_stats['hybrid_evals'] / eval_stats['total_evals']
                print(f"  Transformer usage: {transformer_pct:.1f}%")
        
        return best_move
    
    def get_statistics(self) -> dict:
        """
        Get detailed statistics about the bot's performance.
        
        Returns:
            Dictionary with search and evaluation statistics
        """
        search_stats = self.search_engine.get_statistics()
        eval_stats = self.hybrid_evaluator.stats.copy()
        
        stats = {
            'search': search_stats,
            'evaluation': eval_stats,
            'config': {
                'max_depth': self.depth,
                'time_limit': self.time_limit,
                'device': self.device,
                'use_time_management': self.use_time_management
            }
        }
        
        # Add time management statistics if enabled
        if self.use_time_management and self.time_manager is not None:
            stats['time_management'] = self.time_manager.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.search_engine.reset_statistics()
        self.hybrid_evaluator.stats = {
            'nnue_only_evals': 0,
            'hybrid_evals': 0,
            'total_time_nnue': 0.0,
            'total_time_hybrid': 0.0,
            'total_evals': 0
        }
    
    def analyze_position(self, board: chess.Board, depth: int = None) -> dict:
        """
        Analyze a position in detail.
        
        Args:
            board: chess.Board to analyze
            depth: Search depth (uses default if None)
            
        Returns:
            Dictionary with analysis results
        """
        depth = depth or self.depth
        
        # Get best move and evaluation
        best_move, score = self.search_engine.iterative_deepening(
            board=board,
            max_depth=depth,
            time_limit=self.time_limit
        )
        
        # Get top moves
        stats = self.search_engine.get_statistics()
        
        return {
            'best_move': best_move.uci(),
            'score': score,
            'depth': depth,
            'nodes': stats['nodes_searched'],
            'time': stats['time_elapsed'],
            'nps': stats['nps'],
            'pv': [best_move.uci()]  # Principal variation (just best move for now)
        }


def create_hybrid_bot(checkpoint: str = 'checkpoints/best_phase2.pt',
                     depth: int = 5, 
                     time_limit: float = 5.0,
                     verbose: bool = False) -> HybridChessBot:
    """
    Convenience function to create a HybridChessBot.
    
    Args:
        checkpoint: Path to trained checkpoint
        depth: Maximum search depth
        time_limit: Time limit per move
        verbose: Print statistics
        
    Returns:
        Initialized HybridChessBot
    """
    return HybridChessBot(
        checkpoint=checkpoint,
        depth=depth,
        time_limit=time_limit,
        verbose=verbose
    )


# Example usage
if __name__ == '__main__':
    """Test the bot with a simple position."""
    
    print("=" * 60)
    print("Hybrid NNUE-Transformer Chess Bot")
    print("=" * 60)
    
    # Create bot
    bot = HybridChessBot(
        checkpoint='checkpoints/best_phase2.pt',
        depth=5,
        verbose=True
    )
    
    # Test position
    board = chess.Board()
    
    print("\nStarting position:")
    print(board)
    print(f"\nWhite to move")
    
    # Get move
    print("\nThinking...")
    move = bot.choose_move(board)
    
    print(f"\nChosen move: {move.uci()}")
    
    # Make move
    board.push(move)
    print("\nPosition after move:")
    print(board)
    
    print("\n" + "=" * 60)
    print("Test complete!")
