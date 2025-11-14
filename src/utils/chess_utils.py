"""
Utility functions for the Hybrid NNUE-Transformer Chess Engine
"""

import torch
import chess
import numpy as np
from typing import List, Tuple
import config


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to index in [0, 1857]
    
    Encoding scheme:
    - Normal moves: from_square * 64 + to_square
    - Promotions: Add offset based on promotion piece type
    
    Args:
        move: chess.Move object
        
    Returns:
        index: Integer in range [0, 1857]
    """
    from_sq = move.from_square
    to_sq = move.to_square
    
    # Basic move index
    index = from_sq * 64 + to_sq
    
    # Handle promotions
    if move.promotion:
        # Add offset for promotion moves
        base_moves = 64 * 64  # 4096 normal moves
        promotion_offset = {
            chess.KNIGHT: 0,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 3
        }
        
        # Promotions use separate index space
        prom_idx = promotion_offset.get(move.promotion, 0)
        index = base_moves + (from_sq * 64 + to_sq) % 500 + prom_idx * 500
    
    return index % config.NUM_MOVES


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert move index back to chess.Move
    
    Args:
        index: Move index [0, 1857]
        board: Current chess board (for validation)
        
    Returns:
        move: chess.Move object (if legal), else None
    """
    if index >= 4096:
        # Promotion move
        adjusted_idx = index - 4096
        prom_type = adjusted_idx // 500
        move_idx = adjusted_idx % 500
        
        promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        promotion = promotion_pieces[prom_type]
        
        # Approximate from/to squares
        from_sq = move_idx // 64
        to_sq = move_idx % 64
        
        move = chess.Move(from_sq, to_sq, promotion=promotion)
    else:
        # Normal move
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)
    
    # Validate move is legal
    if move in board.legal_moves:
        return move
    
    return None


def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """
    Create binary mask for legal moves
    
    Args:
        board: Chess board
        
    Returns:
        mask: Boolean tensor [1858] where mask[i] = True if move i is legal
    """
    mask = torch.zeros(config.NUM_MOVES, dtype=torch.bool)
    
    for move in board.legal_moves:
        move_index = move_to_index(move)
        mask[move_index] = True
    
    return mask


def legal_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, 
                  temperature: float = 1.0) -> torch.Tensor:
    """
    Compute softmax over legal moves only using log-sum-exp trick
    
    Args:
        logits: Raw network outputs, shape [num_moves]
        legal_mask: Binary mask, shape [num_moves], 1=legal, 0=illegal
        temperature: Temperature for exploration control
        
    Returns:
        probs: Probability distribution over moves, shape [num_moves]
    """
    # Scale by temperature
    scaled_logits = logits / temperature
    
    # Mask illegal moves with large negative value
    NINF = -1e9
    masked_logits = torch.where(legal_mask, scaled_logits, torch.tensor(NINF))
    
    # Log-sum-exp trick for numerical stability
    if legal_mask.any():
        max_logit = masked_logits[legal_mask].max()
        shifted_logits = masked_logits - max_logit
        
        # Exponentiate and mask
        exp_logits = torch.exp(shifted_logits) * legal_mask.float()
        
        # Normalize
        sum_exp = exp_logits.sum()
        probs = exp_logits / (sum_exp + 1e-8)
    else:
        # No legal moves - return uniform (shouldn't happen)
        probs = torch.zeros_like(logits)
    
    return probs


def extract_selection_features(board: chess.Board, depth_remaining: int) -> torch.Tensor:
    """
    Extract fast features to decide if transformer is needed
    Must be very fast (<1% of NNUE evaluation time)
    
    Args:
        board: Chess board
        depth_remaining: Depth left in search tree
        
    Returns:
        features: Tensor of shape [20]
    """
    features = []
    
    # Feature 1-2: Move counts
    legal_moves = list(board.legal_moves)
    features.append(len(legal_moves) / 50.0)  # Normalize
    capture_moves = [m for m in legal_moves if board.is_capture(m)]
    features.append(len(capture_moves) / 20.0)
    
    # Feature 3: Material balance (pawns)
    material = (
        len(board.pieces(chess.PAWN, chess.WHITE)) - 
        len(board.pieces(chess.PAWN, chess.BLACK))
    )
    features.append(material / 8.0)
    
    # Feature 4-5: Total pieces (game phase)
    total_pieces = len(board.piece_map())
    features.append(total_pieces / 32.0)
    features.append(1.0 if total_pieces <= 10 else 0.0)  # Endgame flag
    
    # Feature 6-7: King safety
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square is not None and black_king_square is not None:
        features.append(len(board.attackers(chess.BLACK, white_king_square)) / 8.0)
        features.append(len(board.attackers(chess.WHITE, black_king_square)) / 8.0)
    else:
        features.extend([0.0, 0.0])
    
    # Feature 8: Checks
    features.append(1.0 if board.is_check() else 0.0)
    
    # Feature 9-10: Pawn structure
    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    features.append(white_pawns / 8.0)
    features.append(black_pawns / 8.0)
    
    # Feature 11: Castling rights
    castling_count = (
        board.has_kingside_castling_rights(chess.WHITE) +
        board.has_queenside_castling_rights(chess.WHITE) +
        board.has_kingside_castling_rights(chess.BLACK) +
        board.has_queenside_castling_rights(chess.BLACK)
    )
    features.append(castling_count / 4.0)
    
    # Feature 12-13: Center control (e4, e5, d4, d5)
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    white_center = sum(
        1 for sq in center_squares 
        if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
    )
    black_center = sum(
        1 for sq in center_squares 
        if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK
    )
    features.append(white_center / 4.0)
    features.append(black_center / 4.0)
    
    # Feature 14: Search depth (importance)
    features.append(depth_remaining / 20.0)
    
    # Feature 15: Halfmove clock (draw proximity)
    features.append(board.halfmove_clock / 50.0)
    
    # Feature 16: Forcing move ratio (tactical indicator)
    # High ratio = tactical, low ratio = strategic
    forcing_moves = sum(1 for m in legal_moves if 
                       board.is_capture(m) or 
                       board.gives_check(m) or
                       (m.promotion is not None))
    forcing_ratio = forcing_moves / max(len(legal_moves), 1)
    features.append(forcing_ratio)
    
    # Feature 17: Pawn chain length (closed structure indicator)
    # Count pawns that have supporting pawns behind them
    pawn_chains = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            rank, file = divmod(sq, 8)
            # Check for supporting pawns on adjacent files
            if color == chess.WHITE and rank > 0:
                support_sqs = [sq - 9, sq - 7] if file > 0 and file < 7 else []
                if file == 0: support_sqs = [sq - 7]
                if file == 7: support_sqs = [sq - 9]
            elif color == chess.BLACK and rank < 7:
                support_sqs = [sq + 7, sq + 9] if file > 0 and file < 7 else []
                if file == 0: support_sqs = [sq + 9]
                if file == 7: support_sqs = [sq + 7]
            else:
                support_sqs = []
            
            for support in support_sqs:
                if 0 <= support < 64 and support in pawns:
                    pawn_chains += 1
                    break
    features.append(pawn_chains / 16.0)  # Normalize (max ~16 pawns)
    
    # Feature 18: Piece coordination (strategic indicator)
    # Simple measure: pieces defending each other
    piece_defense = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            defenders = board.attackers(piece.color, square)
            if len(defenders) > 0:
                piece_defense += 1
    features.append(piece_defense / 16.0)  # Normalize
    
    # Feature 19: Material imbalance (tactical indicator)
    # Large imbalances need precise tactical evaluation
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                   chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(piece_values.get(piece.piece_type, 0) 
                        for piece in board.piece_map().values() 
                        if piece.color == chess.WHITE and piece.piece_type != chess.KING)
    black_material = sum(piece_values.get(piece.piece_type, 0)
                        for piece in board.piece_map().values()
                        if piece.color == chess.BLACK and piece.piece_type != chess.KING)
    material_imbalance = abs(white_material - black_material) / 39.0  # Max ~39 points
    features.append(material_imbalance)
    
    # Feature 20: Mobility ratio (open vs closed)
    # Higher mobility = more open position (tactical)
    mobility = len(legal_moves)
    features.append(mobility / 50.0)
    
    return torch.tensor(features, dtype=torch.float32)


def position_hash(board: chess.Board) -> int:
    """
    Hash chess position for transposition table
    
    Args:
        board: Chess board
        
    Returns:
        hash: Integer hash of position
    """
    return hash(board.fen())


def extract_halfka_features(board: chess.Board, color: chess.Color) -> List[int]:
    """
    Extract HalfKAv2 features for NNUE
    Returns sparse binary feature vector indices
    
    Args:
        board: Chess board
        color: Color perspective (WHITE or BLACK)
        
    Returns:
        features: List of active feature indices
    """
    features = []
    king_square = board.king(color)
    
    if king_square is None:
        return features
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Feature index based on piece type, color, square, and king square
            piece_idx = piece.piece_type - 1  # 0-5 for pawn-king
            color_idx = 1 if piece.color == color else 0
            
            # Simplified HalfKA encoding
            feature_idx = (
                piece_idx * 64 * 64 * 2 +
                color_idx * 64 * 64 +
                king_square * 64 +
                square
            )
            features.append(feature_idx)
    
    return features


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert chess board to tensor representation (8x8x12 channels)
    
    Args:
        board: Chess board
        
    Returns:
        tensor: Shape [12, 8, 8] representing piece positions
    """
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = piece_to_channel.get((piece.piece_type, piece.color))
            if channel is not None:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                tensor[channel, rank, file] = 1.0
    
    return tensor
