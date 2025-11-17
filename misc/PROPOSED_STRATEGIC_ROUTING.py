"""
PROPOSED CHANGES TO IMPLEMENT STRATEGIC ROUTING
================================================

Option 1: Invert the selector training labels
----------------------------------------------
This makes the selector learn to use Transformer for strategic positions
where NNUE struggles, rather than tactical positions.

In src/train.py, line ~162:
"""

# CURRENT (Use transformer when it has lower error):
improvement = transformer_error < nnue_error
selector_labels = improvement.float().unsqueeze(1)

# PROPOSED (Use transformer when position is STRATEGIC):
# Define strategic positions as those where:
# 1. Both evaluations are similar (no clear tactical advantage)
# 2. Position has strategic characteristics
with torch.no_grad():
    nnue_error = torch.abs(nnue_values - target_values_scaled)
    transformer_error = torch.abs(transformer_values - target_values_scaled)
    
    # Transformer is strategic expert when:
    # 1. Errors are similar (not purely tactical)
    error_diff = torch.abs(nnue_error - transformer_error)
    similar_evaluation = error_diff < 50.0  # Within 50cp
    
    # 2. AND position has strategic features (computed from selection_features)
    # Extract strategic indicators from features
    few_captures = selection_features[:, 1] < 0.2  # Feature 2: few captures
    not_in_check = selection_features[:, 7] < 0.5  # Feature 8: no check
    endgame = selection_features[:, 4] > 0.5      # Feature 5: endgame flag
    low_mobility = selection_features[:, 0] < 0.4  # Feature 1: few moves (closed)
    
    is_strategic = (few_captures & not_in_check) | endgame
    
    # Use transformer for strategic positions where it might excel
    use_transformer = similar_evaluation & is_strategic
    selector_labels = use_transformer.float().unsqueeze(1)

"""
Option 2: Add new strategic features
-------------------------------------
Enhance extract_selection_features() to better detect strategic positions
"""

def extract_selection_features(board: chess.Board, depth_remaining: int) -> torch.Tensor:
    # ... existing features ...
    
    # NEW STRATEGIC FEATURES:
    
    # Feature: Pawn chain length (closed structures)
    pawn_chains = count_pawn_chains(board)
    features.append(pawn_chains / 8.0)
    
    # Feature: Piece coordination (strategic vs tactical)
    coordination = measure_piece_coordination(board)
    features.append(coordination)
    
    # Feature: Pawn structure tension
    pawn_tension = count_pawn_tensions(board)
    features.append(pawn_tension / 10.0)
    
    # Feature: Outpost squares
    outposts = count_outpost_squares(board)
    features.append(outposts / 8.0)
    
    # Feature: Forcing move ratio (low = strategic)
    forcing_moves = sum(1 for m in legal_moves if 
                       board.is_capture(m) or 
                       board.gives_check(m) or
                       m.promotion)
    forcing_ratio = forcing_moves / max(len(legal_moves), 1)
    features.append(forcing_ratio)
    
    # Feature: Space advantage
    space_white = count_controlled_squares(board, chess.WHITE)
    space_black = count_controlled_squares(board, chess.BLACK)
    space_advantage = abs(space_white - space_black) / 64.0
    features.append(space_advantage)
    
    return torch.tensor(features, dtype=torch.float32)

"""
Option 3: Pre-classify positions
---------------------------------
Create explicit rules for strategic vs tactical routing
"""

def is_strategic_position(board: chess.Board) -> bool:
    """Determine if position requires strategic understanding"""
    
    # Endgame with few pieces
    piece_count = len(board.piece_map())
    if 6 <= piece_count <= 12:
        return True
    
    # Closed position (few pawn breaks possible)
    legal_moves = list(board.legal_moves)
    pawn_breaks = sum(1 for m in legal_moves if 
                     board.piece_at(m.from_square).piece_type == chess.PAWN and
                     board.is_capture(m))
    if pawn_breaks < 2:
        return True
    
    # Low immediate tactics
    forcing_moves = sum(1 for m in legal_moves if
                       board.is_capture(m) or 
                       board.gives_check(m))
    forcing_ratio = forcing_moves / len(legal_moves)
    if forcing_ratio < 0.15:  # Less than 15% forcing moves
        return True
    
    # Maneuvering phase (no immediate threats)
    if not board.is_check():
        hanging_pieces = count_hanging_pieces(board)
        if hanging_pieces == 0:
            return True
    
    return False

# Then in training:
selector_labels = torch.tensor([is_strategic_position(b) for b in boards]).unsqueeze(1)
