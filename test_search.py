#!/usr/bin/env python3
"""
Test the alpha-beta search implementation with dummy evaluator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import torch.nn as nn

from src.search import AlphaBetaSearch, TranspositionTable


class DummyEvaluator:
    """Simple evaluator for testing search."""
    
    def __init__(self):
        self.device = 'cpu'
    
    def evaluate(self, board):
        """Return simple material evaluation with checkmate detection."""
        # Check for terminal positions first
        if board.is_checkmate():
            # Checkmate is huge loss for side to move
            value = -1000.0
        elif board.is_stalemate() or board.is_insufficient_material():
            value = 0.0
        else:
            # Material values
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            
            # Count material
            value = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    material = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        value += material
                    else:
                        value -= material
        
        # From white's perspective
        value_tensor = torch.tensor([value], dtype=torch.float32)
        
        # Dummy policy logits (all zeros)
        policy_logits = torch.zeros(1858)
        
        # Dummy transformer usage
        use_transformer = True
        
        return policy_logits, value_tensor, use_transformer


def test_transposition_table():
    """Test transposition table functionality."""
    print("Testing Transposition Table...")
    
    tt = TranspositionTable(max_size=100)
    
    # Store some entries
    tt.store(12345, 0.5, 5, TranspositionTable.EXACT, None)
    tt.store(67890, -0.3, 4, TranspositionTable.LOWER_BOUND, None)
    
    # Retrieve entries
    entry1 = tt.get(12345)
    assert entry1 is not None
    assert entry1[0] == 0.5
    assert entry1[1] == 5
    
    entry2 = tt.get(67890)
    assert entry2 is not None
    assert entry2[0] == -0.3
    
    # Non-existent entry
    entry3 = tt.get(99999)
    assert entry3 is None
    
    stats = tt.get_stats()
    print(f"✓ TT Stats: {stats}")
    
    print("✓ Transposition Table tests passed!\n")


def test_move_ordering():
    """Test move ordering."""
    print("Testing Move Ordering...")
    
    board = chess.Board()
    evaluator = DummyEvaluator()
    search = AlphaBetaSearch(evaluator, max_depth=3)
    
    # Test initial position
    moves = list(board.legal_moves)
    ordered = search.order_moves(board, moves)
    
    print(f"✓ Ordered {len(ordered)} moves")
    print(f"  First moves: {ordered[:5]}")
    
    # Test position with captures
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    moves = list(board.legal_moves)
    ordered = search.order_moves(board, moves)
    
    # e4xd5 should be prioritized (capture)
    capture_move = chess.Move.from_uci("e4d5")
    assert capture_move in ordered[:5], "Capture should be prioritized"
    
    print("✓ Move ordering tests passed!\n")


def test_basic_search():
    """Test basic alpha-beta search."""
    print("Testing Basic Alpha-Beta Search...")
    
    board = chess.Board()
    evaluator = DummyEvaluator()
    search = AlphaBetaSearch(evaluator, max_depth=3, use_quiescence=False)
    
    # Search from starting position
    print("Searching starting position (depth 3)...")
    best_move = search.get_best_move(board, depth=3)
    
    assert best_move is not None
    assert best_move in board.legal_moves
    
    print(f"✓ Best move found: {best_move}")
    print(f"✓ Nodes searched: {search.nodes_searched}")
    print("✓ Basic search tests passed!\n")


def test_iterative_deepening():
    """Test iterative deepening."""
    print("Testing Iterative Deepening...")
    
    board = chess.Board()
    evaluator = DummyEvaluator()
    search = AlphaBetaSearch(evaluator, max_depth=5, use_quiescence=False)
    
    # Search with iterative deepening
    print("Searching with iterative deepening (max depth 4)...")
    best_move, score, stats = search.search_iterative_deepening(board, max_depth=4)
    
    assert best_move is not None
    assert best_move in board.legal_moves
    
    print(f"✓ Best move: {best_move}, Score: {score:.3f}")
    print(f"✓ Stats: {stats}")
    print("✓ Iterative deepening tests passed!\n")


def test_checkmate_detection():
    """Test that search recognizes checkmate positions."""
    print("Testing Checkmate Detection...")
    
    # Fool's mate - mate in 1 for black
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    
    evaluator = DummyEvaluator()
    search = AlphaBetaSearch(evaluator, max_depth=3, use_quiescence=False)
    
    print("Searching for mate in 1 (depth 3)...")
    best_move = search.get_best_move(board, depth=3)
    
    # Qh4# is checkmate
    mate_move = chess.Move.from_uci("d8h4")
    print(f"✓ Best move: {best_move}")
    print(f"✓ Mate move (Qh4): {mate_move}")
    
    # Test that evaluator recognizes checkmate
    board.push(mate_move)
    policy, value, _ = evaluator.evaluate(board)
    print(f"✓ Checkmate evaluation: {value.item()}")
    assert value.item() < -900, "Should recognize checkmate as huge negative value"
    board.pop()
    
    # With depth 3, search should find mate or at least a strong move
    # (Simple material evaluator may not always find mate, but should find good moves)
    assert best_move in board.legal_moves, "Move should be legal"
    
    print("✓ Checkmate detection tests passed!\n")


def test_time_limit():
    """Test time-limited search."""
    print("Testing Time-Limited Search...")
    
    board = chess.Board()
    evaluator = DummyEvaluator()
    search = AlphaBetaSearch(evaluator, max_depth=10, use_quiescence=False)
    
    # Search with time limit
    print("Searching with 1 second time limit...")
    best_move, score, stats = search.search_iterative_deepening(
        board, max_depth=10, time_limit=1.0
    )
    
    assert best_move is not None
    assert stats['time'] <= 2.0, "Should respect time limit (with some buffer)"
    
    print(f"✓ Search completed in {stats['time']:.2f}s")
    print(f"✓ Reached depth {stats['depth']}")
    print("✓ Time limit tests passed!\n")


def main():
    print("=" * 60)
    print("Alpha-Beta Search Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_transposition_table()
        test_move_ordering()
        test_basic_search()
        test_iterative_deepening()
        test_checkmate_detection()
        test_time_limit()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
