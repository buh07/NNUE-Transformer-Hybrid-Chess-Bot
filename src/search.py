"""
Alpha-Beta Search with Transposition Table and Move Ordering
Uses the hybrid evaluator for position evaluation.
"""

import chess
import torch
import time
from typing import Tuple, Optional, Dict
from collections import OrderedDict


class TranspositionTable:
    """
    Transposition table for caching position evaluations.
    Uses LRU eviction when table gets too large.
    """
    
    # Entry types
    EXACT = 0
    LOWER_BOUND = 1  # Alpha cutoff
    UPPER_BOUND = 2  # Beta cutoff
    
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.table = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, zobrist_hash: int) -> Optional[Tuple[float, int, int, chess.Move]]:
        """
        Returns (score, depth, entry_type, best_move) if found, else None.
        """
        if zobrist_hash in self.table:
            self.hits += 1
            # Move to end (most recently used)
            entry = self.table.pop(zobrist_hash)
            self.table[zobrist_hash] = entry
            return entry
        self.misses += 1
        return None
    
    def store(self, zobrist_hash: int, score: float, depth: int, 
              entry_type: int, best_move: Optional[chess.Move]):
        """
        Store evaluation in transposition table.
        """
        # If table is full, remove oldest entry (LRU)
        if len(self.table) >= self.max_size:
            self.table.popitem(last=False)
        
        self.table[zobrist_hash] = (score, depth, entry_type, best_move)
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Return statistics about table usage."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.table),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class AlphaBetaSearch:
    """
    Alpha-beta search engine using the hybrid NNUE-Transformer evaluator.
    Features:
    - Alpha-beta pruning
    - Transposition table
    - Move ordering (captures first, then checks)
    - Iterative deepening
    - Quiescence search for tactical positions
    """
    
    def __init__(self, hybrid_evaluator, max_depth: int = 5, 
                 tt_size: int = 1000000, use_quiescence: bool = True):
        """
        Args:
            hybrid_evaluator: HybridEvaluator instance
            max_depth: Maximum search depth
            tt_size: Transposition table size
            use_quiescence: Whether to use quiescence search
        """
        self.evaluator = hybrid_evaluator
        self.max_depth = max_depth
        self.tt = TranspositionTable(max_size=tt_size)
        self.use_quiescence = use_quiescence
        
        # Search statistics
        self.nodes_searched = 0
        self.quiescence_nodes = 0
        self.start_time = 0
        self.time_limit = None
    
    def order_moves(self, board: chess.Board, moves: list) -> list:
        """
        Order moves for better alpha-beta pruning.
        Priority: Captures > Checks > Other moves
        """
        scored_moves = []
        for move in moves:
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    score += 10 * captured.piece_type - attacker.piece_type
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5
            board.pop()
            
            # Prioritize promotions
            if move.promotion:
                score += 8
            
            scored_moves.append((score, move))
        
        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, 
                          max_depth: int = 4) -> float:
        """
        Quiescence search to avoid horizon effect.
        Only searches capture moves to reach a "quiet" position.
        """
        if max_depth == 0:
            return self.evaluate_position(board)
        
        self.quiescence_nodes += 1
        
        # Stand-pat evaluation
        stand_pat = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only search captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        captures = self.order_moves(board, captures)
        
        for move in captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a position using the hybrid evaluator.
        Returns score from current player's perspective.
        """
        # Check for terminal positions
        if board.is_checkmate():
            return -10000  # Loss for current player
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if board.is_repetition(3):
            return 0
        
        # Use hybrid evaluator
        with torch.no_grad():
            policy_logits, value, use_transformer = self.evaluator.evaluate(board)
        
        # Value is from white's perspective, convert to current player's perspective
        score = value.item()
        if not board.turn:  # Black to move
            score = -score
        
        return score
    
    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, 
                   beta: float, is_maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        """
        Alpha-beta search with transposition table.
        
        Returns:
            (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # Check time limit
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            return self.evaluate_position(board), None
        
        # Check transposition table
        zobrist = board._transposition_key()
        tt_entry = self.tt.get(zobrist)
        if tt_entry is not None:
            tt_score, tt_depth, tt_type, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_type == TranspositionTable.EXACT:
                    return tt_score, tt_move
                elif tt_type == TranspositionTable.LOWER_BOUND:
                    alpha = max(alpha, tt_score)
                elif tt_type == TranspositionTable.UPPER_BOUND:
                    beta = min(beta, tt_score)
                
                if alpha >= beta:
                    return tt_score, tt_move
        
        # Terminal depth or game over
        if depth == 0 or board.is_game_over():
            if self.use_quiescence and depth == 0:
                score = self.quiescence_search(board, alpha, beta)
            else:
                score = self.evaluate_position(board)
            return score, None
        
        # Get and order legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(board), None
        
        # Use transposition table move for move ordering
        if tt_entry and tt_entry[3]:
            tt_move = tt_entry[3]
            if tt_move in legal_moves:
                legal_moves.remove(tt_move)
                legal_moves.insert(0, tt_move)
        
        legal_moves = self.order_moves(board, legal_moves)
        
        best_move = legal_moves[0]
        best_score = float('-inf') if is_maximizing else float('inf')
        
        for move in legal_moves:
            board.push(move)
            score, _ = self.alpha_beta(board, depth - 1, alpha, beta, not is_maximizing)
            board.pop()
            
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break
        
        # Store in transposition table
        if best_score <= alpha:
            entry_type = TranspositionTable.UPPER_BOUND
        elif best_score >= beta:
            entry_type = TranspositionTable.LOWER_BOUND
        else:
            entry_type = TranspositionTable.EXACT
        
        self.tt.store(zobrist, best_score, depth, entry_type, best_move)
        
        return best_score, best_move
    
    def search_iterative_deepening(self, board: chess.Board, max_depth: Optional[int] = None,
                                   time_limit: Optional[float] = None) -> Tuple[chess.Move, float, Dict]:
        """
        Iterative deepening search with time management.
        
        Args:
            board: Current board position
            max_depth: Maximum depth to search (if None, uses self.max_depth)
            time_limit: Time limit in seconds (if None, no limit)
        
        Returns:
            (best_move, score, stats) tuple
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes_searched = 0
        self.quiescence_nodes = 0
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            try:
                score, move = self.alpha_beta(board, depth, float('-inf'), float('inf'), True)
                
                # Check if search was interrupted by time limit
                if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                    break
                
                best_move = move
                best_score = score
                
                elapsed = time.time() - self.start_time
                nps = self.nodes_searched / elapsed if elapsed > 0 else 0
                
                print(f"Depth {depth}: score={score:.3f}, move={move}, "
                      f"nodes={self.nodes_searched}, time={elapsed:.2f}s, nps={nps:.0f}")
                
            except Exception as e:
                print(f"Search interrupted at depth {depth}: {e}")
                break
        
        # Compile statistics
        elapsed = time.time() - self.start_time
        tt_stats = self.tt.get_stats()
        
        stats = {
            'depth': depth,
            'nodes': self.nodes_searched,
            'quiescence_nodes': self.quiescence_nodes,
            'time': elapsed,
            'nps': self.nodes_searched / elapsed if elapsed > 0 else 0,
            'tt_size': tt_stats['size'],
            'tt_hit_rate': tt_stats['hit_rate']
        }
        
        return best_move, best_score, stats
    
    def get_best_move(self, board: chess.Board, depth: Optional[int] = None,
                      time_limit: Optional[float] = None) -> chess.Move:
        """
        Get the best move for the current position.
        
        Args:
            board: Current board position
            depth: Search depth (if None, uses self.max_depth)
            time_limit: Time limit in seconds
        
        Returns:
            Best move found
        """
        best_move, score, stats = self.search_iterative_deepening(board, depth, time_limit)
        
        print(f"\nBest move: {best_move} (score: {score:.3f})")
        print(f"Statistics: {stats}")
        
        if best_move is None:
            # Fallback: return first legal move
            best_move = next(board.legal_moves)
        
        return best_move
    
    def clear_tt(self):
        """Clear the transposition table."""
        self.tt.clear()
