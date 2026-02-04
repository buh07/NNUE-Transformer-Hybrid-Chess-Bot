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
from utils.chess_utils import index_to_move, get_legal_move_mask
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
        bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=None)
        move = bot.choose_move(board)
    """
    
    def __init__(self, checkpoint: str = None, 
                 depth: int = None, time_limit: float = 5.0,
                 device: str = None, verbose: bool = False,
                 use_time_management: bool = False,
                 total_game_time: float = None,
                 evaluation_mode: str = 'auto',
                 engine_backend: str = None):
        """
        Initialize the hybrid chess bot with ALL downloaded weights.
        
        Args:
            checkpoint: Path to trained model checkpoint (default: config.HYBRID_CHECKPOINT_PATH)
            depth: Optional maximum search depth. Set to None or <=0 to rely fully on time controls.
            time_limit: Time limit per move in seconds (default: 5.0)
            device: 'cuda' or 'cpu' (auto-detect if None)
            verbose: Print search statistics
            use_time_management: Enable dynamic time allocation (default: False)
            total_game_time: Total time for game in seconds (required if use_time_management=True)
            evaluation_mode: 'auto' (selector decides), 'nnue', or 'transformer'
        """
        self.verbose = verbose
        self.max_search_depth = depth if (depth is not None and depth > 0) else None
        self.time_limit = time_limit
        self.use_time_management = use_time_management
        self.evaluation_mode = evaluation_mode
        
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
        
        # Use default checkpoint if not specified
        if checkpoint is None:
            checkpoint = config.HYBRID_CHECKPOINT_PATH
            if self.verbose:
                print(f"[HybridChessBot] Using default checkpoint: {checkpoint}")
        
        # Convert checkpoint path to absolute if relative
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(os.path.dirname(__file__), checkpoint)
        
        # Load models (with ALL downloaded weights)
        self._load_models(checkpoint, evaluation_mode=self.evaluation_mode, engine_backend=engine_backend)
        
        # Initialize search engine
        engine_default_depth = self.max_search_depth or getattr(config, 'MAX_SEARCH_DEPTH', 18)
        self.search_engine = AlphaBetaSearch(
            hybrid_evaluator=self.hybrid_evaluator,
            max_depth=engine_default_depth,
            tt_size=getattr(config, 'TT_SIZE', 2000000),
            use_quiescence=True
        )
        
        if self.verbose:
            depth_msg = self.max_search_depth if self.max_search_depth else f"time-based (<= {engine_default_depth})"
            print(f"[HybridChessBot] Ready! Max depth: {depth_msg}")
            print(f"[HybridChessBot] Projection params: {self._count_params(self.projection):,}")
            print(f"[HybridChessBot] Selector params: {self._count_params(self.selector):,}")
    
    def _load_models(self, checkpoint_path: str, evaluation_mode: str = 'auto', engine_backend: str = None):
        """Load all model components and trained weights."""
        if self.verbose:
            print(f"\n[HybridChessBot] Loading ALL available weights:")
            print(f"  1. Stockfish NNUE: {config.STOCKFISH_NNUE_PATH}")
            print(f"  2. Stockfish Engine: {config.STOCKFISH_BINARY_PATH}")
            print(f"  3. ChessTransformer: {config.TRANSFORMER_WEIGHTS_PATH}")
            print(f"  4. Hybrid checkpoint: {checkpoint_path}")
        
        # Create base models (frozen, pre-trained)
        # Use Stockfish engine for NNUE evaluations (uses downloaded nn-49c1193b131c.nnue)
        additional_kwargs = {}
        if engine_backend:
            additional_kwargs["preferred_backend"] = engine_backend

        self.nnue = create_nnue_evaluator(
            weights_path=config.STOCKFISH_NNUE_PATH,
            use_stockfish=True,
            **additional_kwargs,
        )
        
        # Load pre-trained transformer (uses CT-EFT-85.pt)
        self.transformer = create_transformer_model(
            weights_path=config.TRANSFORMER_WEIGHTS_PATH
        )
        # Try to compile the transformer for faster inference when supported.
        # Compilation can invoke Triton/ptxas which may not be present or may
        # fail when the path contains spaces. Disable Triton and make Dynamo
        # fall back to eager on compilation errors to keep runtime robust.
        try:
            # Prefer disabling Triton (which needs ptxas) to avoid subprocess
            # errors when paths contain spaces or ptxas is unavailable.
            try:
                os.environ.setdefault('TRITON_DISABLE', '1')
            except Exception:
                pass

            # Make Dynamo suppress errors and fallback to eager if compile fails.
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.suppress_errors = True
            except Exception:
                # Older torch may not have _dynamo exposed; ignore.
                pass

            if hasattr(torch, 'compile'):
                # Only attempt to compile if explicitly enabled via env var.
                # Compilation can trigger Triton/ptxas which may not be present
                # or can fail if paths contain spaces. Default is to skip
                # compilation to avoid hard failures; enable explicitly by
                # setting HYBRID_ENABLE_TORCH_COMPILE=1 in the environment.
                if os.environ.get('HYBRID_ENABLE_TORCH_COMPILE', '0') == '1':
                    try:
                        self.transformer = torch.compile(self.transformer)
                        if self.verbose:
                            print("  ✓ torch.compile() applied to transformer (triton disabled)")
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠ torch.compile() failed (fallback to eager): {e}")
                else:
                    if self.verbose:
                        print("  ⚑ torch.compile() disabled by HYBRID_ENABLE_TORCH_COMPILE env var (default) — running eager mode")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Transformer compile guard failed: {e}")
        
        # Create trainable components (use config defaults)
        self.projection = create_projection_layer()
        self.selector = create_selector()
        
        # Load trained projection and selector weights from hybrid training
        if os.path.exists(checkpoint_path):
            # Use weights_only=True when available to avoid unpickling arbitrary objects
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                # Older torch versions may not support weights_only kwarg
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load projection and selector weights
            if 'projection_state_dict' in checkpoint:
                self.projection.load_state_dict(checkpoint['projection_state_dict'])
                if self.verbose:
                    print(f"  ✓ Loaded trained projection weights")
            else:
                if self.verbose:
                    print(f"  ⚠ No projection weights in checkpoint")
            
            if 'selector_state_dict' in checkpoint:
                self.selector.load_state_dict(checkpoint['selector_state_dict'])
                if self.verbose:
                    print(f"  ✓ Loaded trained selector weights")
            else:
                if self.verbose:
                    print(f"  ⚠ No selector weights in checkpoint")
            
            # Print training info if available and detect legacy scaling
            history = checkpoint.get('history', {})
            compat_scale = None
            if history:
                last_val_loss = history.get('val_loss', [None])[-1] if 'val_loss' in history else None
                last_selector_acc = history.get('selector_accuracy', [None])[-1] if 'selector_accuracy' in history else None
                if self.verbose and last_val_loss:
                    try:
                        print(f"  ✓ Final validation loss: {last_val_loss:.4f}")
                    except Exception:
                        print(f"  ✓ Final validation loss: {last_val_loss}")
                if self.verbose and last_selector_acc:
                    try:
                        print(f"  ✓ Final selector accuracy: {last_selector_acc:.2f}%")
                    except Exception:
                        print(f"  ✓ Final selector accuracy: {last_selector_acc}")

                # Heuristic: very large validation loss indicates legacy 100x scaling
                try:
                    if last_val_loss is not None and float(last_val_loss) > 1000.0:
                        compat_scale = 100.0
                        print("\n[WARNING] Checkpoint appears to have been trained with legacy 100x value scaling.")
                        print("[WARNING] Inference will automatically rescale evaluator outputs by /100 to maintain numeric compatibility.")
                except Exception:
                    pass
        else:
            print(f"\n[WARNING] Checkpoint not found: {checkpoint_path}")
            print(f"[WARNING] Using untrained projection/selector weights!")
            print(f"[WARNING] NNUE and Transformer will still use their pre-trained weights.")
        
        # Create hybrid evaluator (pass compatibility scaling if detected)
        self.hybrid_evaluator = HybridEvaluator(
            nnue_model=self.nnue,
            transformer_model=self.transformer,
            projection_layer=self.projection,
            selector=self.selector,
            device=self.device,
            compatibility_scale=compat_scale,
            evaluation_mode=evaluation_mode
        )
        self.evaluation_mode = self.hybrid_evaluator.evaluation_mode
        
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
            depth_hint = self.max_search_depth or getattr(config, 'MAX_SEARCH_DEPTH', 18)
            time_alloc = self.time_manager.allocate_time(
                board=board,
                selector_model=self.selector,
                depth_remaining=depth_hint
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
        prepare_fn = getattr(self.hybrid_evaluator, 'prepare_search', None)
        if callable(prepare_fn):
            try:
                prepare_fn(board)
            except Exception:
                pass
        best_move, score = self.search_engine.iterative_deepening(
            board=board,
            max_depth=self.max_search_depth,
            time_limit=time_for_move
        )
        
        # Update time manager
        if self.use_time_management and move_start_time is not None:
            import time
            time_used = time.time() - move_start_time
            self.time_manager.update_after_move(time_used)
        
        # Defensive fallback: if search returned None, pick a sensible legal move
        if best_move is None:
            if self.verbose:
                print("[WARNING] Search returned None for best_move — applying fallback policy-based move selection.")

            # Try to obtain policy distribution from evaluator and map to a legal move
            try:
                legal_mask = get_legal_move_mask(board).to(self.device)
                depth_remaining = self.max_search_depth or getattr(config, 'MAX_SEARCH_DEPTH', 18)
                policy_probs, _value, _method = self.hybrid_evaluator.evaluate(board, depth_remaining=depth_remaining, legal_mask=legal_mask)

                # Mask illegal moves and pick highest-probability legal move
                masked_probs = policy_probs * legal_mask.float()
                best_idx = int(masked_probs.argmax().item())
                fallback_move = index_to_move(best_idx, board)
                if fallback_move is None:
                    # As a last resort, choose a random legal move
                    fallback_move = next(iter(board.legal_moves), None)

                best_move = fallback_move
                if self.verbose:
                    print(f"[Fallback] Chosen move from policy: {best_move}")
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Fallback policy selection failed: {e}. Choosing first legal move.")
                best_move = next(iter(board.legal_moves), None)

        # Print statistics if verbose
        if self.verbose:
            stats = self.search_engine.get_statistics()
            eval_stats = self.hybrid_evaluator.stats
            move_str = best_move.uci() if best_move is not None else 'None'
            print(f"\n[Search Statistics]")
            print(f"  Move: {move_str} (score: {score:.2f})")
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
                'max_depth': self.max_search_depth,
                'time_based_search': self.max_search_depth is None,
                'time_limit': self.time_limit,
                'device': self.device,
                'use_time_management': self.use_time_management,
                'evaluation_mode': self.evaluation_mode
            }
        }
        
        # Add time management statistics if enabled
        if self.use_time_management and self.time_manager is not None:
            stats['time_management'] = self.time_manager.get_statistics()
        
        return stats

    def set_evaluation_mode(self, mode: str):
        """
        Force evaluation mode (auto/nnue/transformer).
        """
        self.hybrid_evaluator.set_evaluation_mode(mode)
        self.evaluation_mode = self.hybrid_evaluator.evaluation_mode
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.search_engine.reset_statistics()
        self.hybrid_evaluator.reset_stats()
    
    def analyze_position(self, board: chess.Board, depth: int = None) -> dict:
        """
        Analyze a position in detail.
        
        Args:
            board: chess.Board to analyze
            depth: Search depth (uses default if None)
            
        Returns:
            Dictionary with analysis results
        """
        if depth is None or depth <= 0:
            depth = self.max_search_depth or getattr(config, 'MAX_SEARCH_DEPTH', 18)
        
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
                     depth: int = None, 
                     time_limit: float = 5.0,
                     verbose: bool = False) -> HybridChessBot:
    """
    Convenience function to create a HybridChessBot.
    
    Args:
        checkpoint: Path to trained checkpoint
        depth: Optional depth cap (None = time-based)
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
        depth=None,
        time_limit=3.0,
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
