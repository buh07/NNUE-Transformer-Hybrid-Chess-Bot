# Current vs Proposed Routing Strategy

## 🔴 CURRENT IMPLEMENTATION: "Transformer for Tactics"

### Assumption
- **NNUE**: Fast but tactically limited
- **Transformer**: Slow but tactically strong
- **Use Transformer when**: Position has high tactical complexity

### Training Label
```python
# Use transformer when it reduces value error
improvement = transformer_error < nnue_error
selector_labels = improvement
```

### Features Favor Transformer For:
- ✓ Many legal moves (open positions)
- ✓ Many captures available
- ✓ King under attack
- ✓ Checks on the board
- ✓ High search depth (important nodes)

### Result from Testing:
- Pure NNUE (1.0/0.0) gave BEST value loss (8214.84)
- Transformer value head didn't improve evaluations
- **This suggests NNUE is already excellent at tactical evaluation!**

---

## 🟢 PROPOSED STRATEGY: "Transformer for Strategy"

### Revised Assumption
- **NNUE**: Tactically strong (trained on billions of Stockfish positions)
- **Transformer**: Strategically understanding (trained on human games)
- **Use Transformer when**: Position needs strategic/positional understanding

### Routing Rules

**→ Route to ChessFormer (Transformer):**
- Low tactical complexity (few hanging pieces, no forcing sequences)
- Closed pawn structures (strategic planning > calculation)
- Endgames (6-12 pieces, positional understanding critical)
- Positional deadlocks (maneuvering without immediate threats)

**→ Route to NNUE:**
- Open positions (many tactical possibilities)
- Material imbalances (precise evaluation needed)
- Forcing sequences (checks, captures, threats)
- Time pressure (speed matters)

---

## 🔧 IMPLEMENTATION CHANGES NEEDED

### 1. **Modify Selector Training Labels** (src/train.py)

**Current:**
```python
# Line ~162
improvement = transformer_error < nnue_error
selector_labels = improvement.float()
```

**Proposed:**
```python
# Label positions as strategic rather than where transformer wins
with torch.no_grad():
    # Strategic indicators from features
    few_captures = selection_features[:, 1] < 0.2
    not_in_check = selection_features[:, 7] < 0.5
    endgame = selection_features[:, 4] > 0.5
    closed = selection_features[:, 0] < 0.4  # Few legal moves
    
    # Strategic = (closed OR endgame) AND not tactical
    is_strategic = ((closed | endgame) & few_captures & not_in_check)
    
    selector_labels = is_strategic.float().unsqueeze(1)
```

### 2. **Enhance Strategic Features** (src/utils/chess_utils.py)

Add new features in `extract_selection_features()`:

```python
# Strategic features to add:

# Pawn structure complexity
def count_pawn_chains(board):
    """Count connected pawns (closed structure indicator)"""
    chains = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            # Check for supporting pawns
            rank, file = divmod(sq, 8)
            if color == chess.WHITE:
                support_sqs = [sq - 9, sq - 7]  # Behind diagonals
            else:
                support_sqs = [sq + 7, sq + 9]
            for support in support_sqs:
                if 0 <= support < 64 and support in pawns:
                    chains += 1
    return chains

# Tactical tension
def count_forcing_moves(board):
    """Count checks, captures, threats"""
    forcing = 0
    for move in board.legal_moves:
        if (board.is_capture(move) or 
            board.gives_check(move) or 
            move.promotion):
            forcing += 1
    return forcing

# Strategic space
def measure_space_control(board):
    """Count squares controlled by each side"""
    white_control = len(board.attacks(chess.WHITE))
    black_control = len(board.attacks(chess.BLACK))
    return abs(white_control - black_control)

# Add these to feature vector
features.append(count_pawn_chains(board) / 8.0)
features.append(count_forcing_moves(board) / 20.0)
features.append(measure_space_control(board) / 64.0)
```

### 3. **Invert Selector Logic** (if using explicit rules)

Create `is_strategic_position()` helper:

```python
def is_strategic_position(board: chess.Board) -> bool:
    """Explicit rules for strategic positions"""
    
    # Endgame
    if 6 <= len(board.piece_map()) <= 12:
        return True
    
    # Closed (few pawn breaks)
    legal_moves = list(board.legal_moves)
    forcing = sum(1 for m in legal_moves if 
                  board.is_capture(m) or board.gives_check(m))
    if forcing / len(legal_moves) < 0.15:  # <15% tactical
        return True
    
    # No hanging pieces = strategic maneuvering
    if not board.is_check():
        # Count undefended pieces
        hanging = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                if len(attackers) > len(defenders):
                    hanging += 1
        if hanging == 0:
            return True
    
    return False
```

---

## 🎯 EXPECTED IMPROVEMENTS

### Why This Makes Sense:

1. **Your Test Results Show NNUE is Tactically Strong**
   - Pure NNUE (1.0/0.0) gave best value loss
   - NNUE trained on Stockfish evaluations = billions of tactical positions
   - Transformer value head didn't add tactical improvement

2. **Transformer's Real Strength: Pattern Recognition**
   - Trained on human games with strategic themes
   - Better at long-term planning vs. tactical calculation
   - Policy head still valuable for move selection

3. **Optimal Division of Labor**
   - NNUE: Fast, tactical, precise evaluation
   - Transformer: Strategic patterns, positional understanding
   - Selector: Route based on position type, not just "complexity"

### Performance Prediction:

- **Better value accuracy** in endgames and closed positions
- **Faster evaluation** in tactical positions (fewer transformer calls)
- **More selective transformer usage** (maybe 1-3% vs current ~2-5%)
- **Improved playing strength** in positional games

---

## ⚡ QUICK START: Minimal Changes

If you want to test this quickly, just change the selector training:

```python
# In src/train.py, replace lines ~162-165:

# OLD:
improvement = transformer_error < nnue_error
selector_labels = improvement.float().unsqueeze(1)

# NEW:
# Use transformer for endgames and closed positions
endgame = selection_features[:, 4] > 0.5          # Feature 5
few_captures = selection_features[:, 1] < 0.2     # Feature 2
not_check = selection_features[:, 7] < 0.5        # Feature 8
closed = selection_features[:, 0] < 0.4           # Feature 1

is_strategic = (endgame | closed) & few_captures & not_check
selector_labels = is_strategic.float().unsqueeze(1)
```

Then retrain Phase 2 (joint training) with this new selector objective.
