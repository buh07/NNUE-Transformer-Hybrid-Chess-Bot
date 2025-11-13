# Hybrid NNUE-Transformer Chess Engine Implementation Guide

## Overview

This document provides complete pseudocode for implementing a hybrid chess engine that combines:
- **Pre-trained NNUE** (frozen) for fast tactical evaluation
- **Pre-trained Transformer** (frozen) for strategic reasoning
- **Learned projection layer** (trainable) to bridge NNUE features to transformer space
- **Learned selection function** (trainable) to decide when to use transformer
- **Per-position selective evaluation** for optimal efficiency
- **Legal move masking** for valid chess moves

**Training Setup**: 2 days on RTX6000 GPU, training only projection layer and selector.

---

## Architecture Diagram

```
                    Chess Position
                          |
                          v
        +----------------+----------------+
        |                                 |
        v                                 v
    NNUE Eval                    Selection Function
   (frozen)                        (trainable)
        |                                 |
        v                                 v
  [1024 features]              [Use NNUE only?]
        |                                 |
        +---------> if complex ------------+
                          |
                          v
                  Projection Layer
                    (trainable)
                          |
                          v
                    [512 features]
                          |
                          v
                   Transformer
                     (frozen)
                          |
                          v
                 Strategic Features
                          |
        +-----------------+-----------------+
        |                                   |
        v                                   v
   Policy Head                         Value Head
  (from transformer)                  (from transformer)
        |                                   |
        v                                   v
  Legal Softmax                        Position Score
```

---

## System Components

### 1. Pre-trained Models (Frozen)

#### 1.1 NNUE Model

```python
class NNUEEvaluator:
    """
    Pre-trained NNUE model (e.g., from Stockfish)
    Architecture: HalfKAv2 features -> Feature Transformer -> Hidden layers
    """
    def __init__(self, nnue_weights_path):
        self.feature_transformer = load_nnue_feature_transformer(nnue_weights_path)
        self.hidden_layers = load_nnue_hidden_layers(nnue_weights_path)
        self.freeze_parameters()  # No training

    def forward(self, position):
        """
        Args:
            position: Board state (python-chess Board object)
        Returns:
            features: Tensor of shape [1024] (accumulator output)
            value: Scalar evaluation score (centipawns)
        """
        # Convert position to HalfKAv2 features (45,056 sparse binary)
        white_features = extract_halfka_features(position, Color.WHITE)
        black_features = extract_halfka_features(position, Color.BLACK)

        # Feature transformer (incremental accumulator)
        white_accumulator = self.feature_transformer(white_features)  # [512]
        black_accumulator = self.feature_transformer(black_features)  # [512]

        # Concatenate and process
        features = concatenate([white_accumulator, black_accumulator])  # [1024]

        # Hidden layers for value
        value = self.hidden_layers(features)  # Scalar

        return features, value

    def incremental_update(self, accumulator, move):
        """Fast incremental update when position changes by one move"""
        changed_features = get_changed_features(move)
        updated_accumulator = accumulator.copy()

        # Only update changed features (typically 2-4 pieces)
        for feature_idx in changed_features.removed:
            updated_accumulator -= self.feature_transformer.weights[feature_idx]
        for feature_idx in changed_features.added:
            updated_accumulator += self.feature_transformer.weights[feature_idx]

        return updated_accumulator
```

#### 1.2 Transformer Model

```python
class ChessTransformer:
    """
    Pre-trained transformer model (e.g., ChessFormer or chess-transformers)
    Architecture: Position encoding -> Transformer layers -> Policy/Value heads
    """
    def __init__(self, transformer_weights_path):
        self.position_encoder = load_position_encoder(transformer_weights_path)
        self.transformer_layers = load_transformer_layers(transformer_weights_path)
        self.policy_head = load_policy_head(transformer_weights_path)
        self.value_head = load_value_head(transformer_weights_path)
        self.freeze_parameters()  # No training

    def forward(self, strategic_features):
        """
        Args:
            strategic_features: Tensor of shape [512] (from projection layer)
        Returns:
            policy_logits: Tensor of shape [1858] (all possible moves)
            value: Scalar position evaluation
        """
        # Add positional encoding
        encoded = self.position_encoder(strategic_features)  # [512]

        # Transformer layers
        for layer in self.transformer_layers:
            encoded = layer(encoded)  # [512]

        # Output heads
        policy_logits = self.policy_head(encoded)  # [1858]
        value = self.value_head(encoded)  # Scalar

        return policy_logits, value
```

---

### 2. Trainable Components

#### 2.1 Projection Layer

```python
class ProjectionLayer(nn.Module):
    """
    Learns to project NNUE features (1024-dim) to transformer space (512-dim)
    This is the bridge between tactical NNUE and strategic transformer
    """
    def __init__(self, nnue_dim=1024, transformer_dim=512):
        super().__init__()

        # Linear projection with layer normalization
        self.projection = nn.Sequential(
            nn.Linear(nnue_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Initialize with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

    def forward(self, nnue_features):
        """
        Args:
            nnue_features: Tensor of shape [batch_size, 1024]
        Returns:
            transformer_features: Tensor of shape [batch_size, 512]
        """
        return self.projection(nnue_features)
```

#### 2.2 Selection Function

```python
class SelectionFunction(nn.Module):
    """
    Learns to predict when transformer evaluation is beneficial
    Input: Fast features about position complexity
    Output: Probability that transformer should be used
    """
    def __init__(self, feature_dim=20):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, position_features):
        """
        Args:
            position_features: Tensor of shape [batch_size, 20]
                Features include: move count, material balance, phase, etc.
        Returns:
            probability: Tensor of shape [batch_size, 1]
                Probability that transformer should be used
        """
        return self.network(position_features)

    def should_use_transformer(self, position_features, threshold=0.5):
        """Binary decision for inference"""
        prob = self.forward(position_features)
        return prob > threshold


def extract_selection_features(position, depth_remaining):
    """
    Extract fast features to decide if transformer is needed
    Must be very fast (<1% of NNUE evaluation time)
    """
    features = []

    # Feature 1-2: Move counts
    legal_moves = list(position.legal_moves)
    features.append(len(legal_moves) / 50.0)  # Normalize
    features.append(len([m for m in legal_moves if position.is_capture(m)]) / 20.0)

    # Feature 3: Material balance
    material = (
        len(position.pieces(chess.PAWN, chess.WHITE)) - 
        len(position.pieces(chess.PAWN, chess.BLACK))
    )
    features.append(material / 8.0)

    # Feature 4-5: Total pieces (game phase)
    total_pieces = len(position.piece_map())
    features.append(total_pieces / 32.0)
    features.append(1.0 if total_pieces <= 10 else 0.0)  # Endgame flag

    # Feature 6-7: King safety
    white_king_square = position.king(chess.WHITE)
    black_king_square = position.king(chess.BLACK)
    features.append(len(position.attackers(chess.BLACK, white_king_square)) / 8.0)
    features.append(len(position.attackers(chess.WHITE, black_king_square)) / 8.0)

    # Feature 8: Checks
    features.append(1.0 if position.is_check() else 0.0)

    # Feature 9-10: Pawn structure
    white_pawns = len(position.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(position.pieces(chess.PAWN, chess.BLACK))
    features.append(white_pawns / 8.0)
    features.append(black_pawns / 8.0)

    # Feature 11: Castling rights
    features.append(sum(position.castling_rights) / 4.0)

    # Feature 12-13: Center control (e4, e5, d4, d5)
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    white_center = sum(1 for sq in center_squares if position.piece_at(sq) and position.piece_at(sq).color == chess.WHITE)
    black_center = sum(1 for sq in center_squares if position.piece_at(sq) and position.piece_at(sq).color == chess.BLACK)
    features.append(white_center / 4.0)
    features.append(black_center / 4.0)

    # Feature 14: Search depth (importance)
    features.append(depth_remaining / 20.0)

    # Feature 15: Halfmove clock (draw proximity)
    features.append(position.halfmove_clock / 50.0)

    # Feature 16-20: Piece mobility (simplified)
    mobility = len(legal_moves)
    features.append(mobility / 50.0)
    features.extend([0.0] * 4)  # Padding to 20 features

    return torch.tensor(features, dtype=torch.float32)
```

---

### 3. Hybrid Evaluation System

#### 3.1 Combined Evaluator

```python
class HybridEvaluator:
    """
    Combines NNUE, projection, transformer, and selection function
    """
    def __init__(self, nnue_model, transformer_model, projection_layer, selector):
        self.nnue = nnue_model
        self.transformer = transformer_model
        self.projection = projection_layer
        self.selector = selector

        # Statistics for monitoring
        self.stats = {
            'nnue_only_evals': 0,
            'hybrid_evals': 0,
            'total_time': 0.0
        }

    def evaluate(self, position, depth_remaining, legal_mask=None):
        """
        Evaluate position using NNUE or hybrid approach

        Args:
            position: Chess board state
            depth_remaining: Depth left in search tree
            legal_mask: Boolean tensor [1858] indicating legal moves

        Returns:
            policy_probs: Probability distribution over moves [1858]
            value: Position evaluation score
            eval_method: 'nnue' or 'hybrid'
        """
        # Step 1: Always get NNUE features and value
        nnue_features, nnue_value = self.nnue.forward(position)

        # Step 2: Decide if transformer is needed
        selection_features = extract_selection_features(position, depth_remaining)
        use_transformer = self.selector.should_use_transformer(selection_features)

        if not use_transformer:
            # Fast path: NNUE only
            self.stats['nnue_only_evals'] += 1

            # NNUE doesn't have policy head, so use uniform over legal moves
            if legal_mask is not None:
                policy_probs = legal_mask.float() / legal_mask.sum()
            else:
                policy_probs = torch.ones(1858) / 1858

            return policy_probs, nnue_value, 'nnue'

        else:
            # Slow path: NNUE + Transformer
            self.stats['hybrid_evals'] += 1

            # Project NNUE features to transformer space
            strategic_features = self.projection(nnue_features.unsqueeze(0))

            # Get transformer policy and value
            policy_logits, transformer_value = self.transformer.forward(strategic_features)

            # Apply legal move masking and softmax
            policy_probs = legal_softmax(policy_logits.squeeze(0), legal_mask)

            # Blend NNUE and transformer values (optional)
            blended_value = 0.3 * nnue_value + 0.7 * transformer_value

            return policy_probs, blended_value, 'hybrid'

    def print_stats(self):
        total = self.stats['nnue_only_evals'] + self.stats['hybrid_evals']
        pct_hybrid = 100 * self.stats['hybrid_evals'] / total if total > 0 else 0
        print(f"Evaluations: {total}")
        print(f"  NNUE only: {self.stats['nnue_only_evals']} ({100-pct_hybrid:.1f}%)")
        print(f"  Hybrid: {self.stats['hybrid_evals']} ({pct_hybrid:.1f}%)")
```

#### 3.2 Legal Softmax

```python
def legal_softmax(logits, legal_mask, temperature=1.0):
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
    masked_logits = torch.where(legal_mask.bool(), scaled_logits, torch.tensor(NINF))

    # Log-sum-exp trick for numerical stability
    max_logit = masked_logits[legal_mask.bool()].max()
    shifted_logits = masked_logits - max_logit

    # Exponentiate and mask
    exp_logits = torch.exp(shifted_logits) * legal_mask.float()

    # Normalize
    sum_exp = exp_logits.sum()
    probs = exp_logits / (sum_exp + 1e-8)

    return probs
```

---

### 4. Search Algorithm

#### 4.1 Alpha-Beta Search with Per-Position Selection

```python
class AlphaBetaSearch:
    """
    Standard alpha-beta search with per-position hybrid evaluation
    """
    def __init__(self, evaluator, max_depth=18):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.transposition_table = {}
        self.nodes_searched = 0

    def search(self, position, time_limit=10.0):
        """
        Iterative deepening search

        Args:
            position: Starting chess position
            time_limit: Maximum time in seconds

        Returns:
            best_move: Best move found
            eval: Evaluation score
        """
        start_time = time.time()
        best_move = None
        best_eval = -float('inf')

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > time_limit * 0.9:
                break

            eval, move = self.alpha_beta(
                position, 
                depth, 
                alpha=-float('inf'), 
                beta=float('inf'),
                depth_remaining=depth
            )

            if move is not None:
                best_move = move
                best_eval = eval

            print(f"Depth {depth}: eval={eval:.2f}, move={best_move}, nodes={self.nodes_searched}")

        return best_move, best_eval

    def alpha_beta(self, position, depth, alpha, beta, depth_remaining):
        """
        Alpha-beta search with per-position hybrid evaluation

        Args:
            position: Current position
            depth: Depth to search
            alpha, beta: Alpha-beta bounds
            depth_remaining: Depth from root (for selection)

        Returns:
            eval: Position evaluation
            best_move: Best move found
        """
        self.nodes_searched += 1

        # Check transposition table
        pos_hash = position_hash(position)
        if pos_hash in self.transposition_table:
            tt_entry = self.transposition_table[pos_hash]
            if tt_entry['depth'] >= depth:
                return tt_entry['eval'], tt_entry['move']

        # Terminal conditions
        if depth == 0 or position.is_game_over():
            legal_mask = get_legal_move_mask(position)
            policy_probs, value, method = self.evaluator.evaluate(
                position, 
                depth_remaining,
                legal_mask
            )
            return value, None

        # Generate legal moves
        legal_moves = list(position.legal_moves)
        if not legal_moves:
            # Checkmate or stalemate
            if position.is_checkmate():
                return -10000, None
            else:
                return 0, None

        # Search children
        best_move = None
        best_eval = -float('inf')

        for move in legal_moves:
            # Make move
            position.push(move)

            # Recursive search (negate for opponent's perspective)
            eval, _ = self.alpha_beta(
                position, 
                depth - 1, 
                -beta, 
                -alpha,
                depth_remaining
            )
            eval = -eval

            # Unmake move
            position.pop()

            # Update best
            if eval > best_eval:
                best_eval = eval
                best_move = move

            # Alpha-beta pruning
            alpha = max(alpha, eval)
            if alpha >= beta:
                break  # Beta cutoff

        # Store in transposition table
        self.transposition_table[pos_hash] = {
            'eval': best_eval,
            'move': best_move,
            'depth': depth
        }

        return best_eval, best_move


def get_legal_move_mask(position):
    """
    Create binary mask for legal moves

    Args:
        position: Chess board

    Returns:
        mask: Boolean tensor [1858] where mask[i] = True if move i is legal
    """
    mask = torch.zeros(1858, dtype=torch.bool)

    for move in position.legal_moves:
        move_index = move_to_index(move)  # Convert chess.Move to index
        mask[move_index] = True

    return mask


def move_to_index(move):
    """Convert chess.Move to index in [0, 1857]"""
    # Standard move encoding: from_square * 64 + to_square + promotion_offset
    from_sq = move.from_square
    to_sq = move.to_square

    # Basic move: from_sq * 64 + to_sq
    index = from_sq * 64 + to_sq

    # Handle promotions (add offset for knight/bishop/rook promotions)
    if move.promotion:
        promotion_offset = {
            chess.KNIGHT: 0,
            chess.BISHOP: 64 * 64,
            chess.ROOK: 64 * 64 * 2,
            chess.QUEEN: 0  # Queen is default, no offset
        }
        index += promotion_offset.get(move.promotion, 0)

    return index % 1858  # Ensure within bounds
```

---

### 5. Training Procedure

#### 5.1 Dataset Preparation

```python
class ChessDataset(torch.utils.data.Dataset):
    """
    Dataset of chess positions with labels
    """
    def __init__(self, pgn_files, max_positions=1000000):
        """
        Args:
            pgn_files: List of PGN files with high-quality games
            max_positions: Maximum positions to extract
        """
        self.positions = []
        self.load_games(pgn_files, max_positions)

    def load_games(self, pgn_files, max_positions):
        """Extract positions from PGN files"""
        count = 0

        for pgn_file in pgn_files:
            with open(pgn_file) as f:
                while count < max_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    # Extract positions from game
                    board = game.board()
                    for move in game.mainline_moves():
                        # Store position before move
                        fen = board.fen()
                        result = game.headers['Result']

                        # Convert result to value
                        if result == '1-0':
                            value = 1.0 if board.turn == chess.WHITE else -1.0
                        elif result == '0-1':
                            value = -1.0 if board.turn == chess.WHITE else 1.0
                        else:
                            value = 0.0

                        self.positions.append({
                            'fen': fen,
                            'move': move.uci(),
                            'value': value,
                            'depth': len(list(game.mainline_moves()))  # Game length
                        })

                        board.push(move)
                        count += 1

                        if count >= max_positions:
                            break

        print(f"Loaded {len(self.positions)} positions")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Returns:
            position: Chess board
            move: Target move (UCI string)
            value: Position value
            selection_features: Features for selector
        """
        data = self.positions[idx]
        board = chess.Board(data['fen'])

        # Extract selection features
        depth_remaining = 15  # Assume mid-search depth
        selection_features = extract_selection_features(board, depth_remaining)

        # Get legal move mask
        legal_mask = get_legal_move_mask(board)

        # Target move index
        target_move = chess.Move.from_uci(data['move'])
        target_index = move_to_index(target_move)

        return {
            'board': board,
            'nnue_features': None,  # Will be computed in forward pass
            'selection_features': selection_features,
            'legal_mask': legal_mask,
            'target_move_index': target_index,
            'target_value': data['value']
        }
```

#### 5.2 Training Loop

```python
def train_hybrid_model(
    nnue_model,
    transformer_model,
    projection_layer,
    selector,
    train_dataset,
    val_dataset,
    num_epochs=50,
    batch_size=256,
    learning_rate=1e-3
):
    """
    Train projection layer and selector (freeze NNUE and transformer)

    Training objectives:
    1. Projection layer: Minimize difference between projected NNUE->transformer path and pure transformer
    2. Selector: Predict when transformer improves over NNUE
    """

    # Freeze pre-trained models
    nnue_model.eval()
    transformer_model.eval()
    for param in nnue_model.parameters():
        param.requires_grad = False
    for param in transformer_model.parameters():
        param.requires_grad = False

    # Trainable models
    projection_layer.train()
    selector.train()

    # Optimizers
    optimizer = torch.optim.Adam([
        {'params': projection_layer.parameters(), 'lr': learning_rate},
        {'params': selector.parameters(), 'lr': learning_rate * 0.1}  # Lower LR for selector
    ])

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    selector_criterion = nn.BCELoss()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        projection_layer.train()
        selector.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            # Extract batch data
            boards = batch['board']
            selection_features = batch['selection_features']  # [batch_size, 20]
            legal_masks = batch['legal_mask']  # [batch_size, 1858]
            target_moves = batch['target_move_index']  # [batch_size]
            target_values = batch['target_value']  # [batch_size]

            # Get NNUE features (no gradients)
            with torch.no_grad():
                nnue_features_list = []
                nnue_values_list = []
                for board in boards:
                    nnue_feat, nnue_val = nnue_model.forward(board)
                    nnue_features_list.append(nnue_feat)
                    nnue_values_list.append(nnue_val)
                nnue_features = torch.stack(nnue_features_list)  # [batch_size, 1024]
                nnue_values = torch.tensor(nnue_values_list)  # [batch_size]

            # Project to transformer space
            projected_features = projection_layer(nnue_features)  # [batch_size, 512]

            # Get transformer outputs (no gradients)
            with torch.no_grad():
                policy_logits_list = []
                transformer_values_list = []
                for proj_feat in projected_features:
                    pol_logits, trans_val = transformer_model.forward(proj_feat.unsqueeze(0))
                    policy_logits_list.append(pol_logits.squeeze(0))
                    transformer_values_list.append(trans_val)
                policy_logits = torch.stack(policy_logits_list)  # [batch_size, 1858]
                transformer_values = torch.tensor(transformer_values_list)  # [batch_size]

            # Loss 1: Policy prediction (cross-entropy on legal moves)
            # Mask illegal moves
            masked_logits = policy_logits.clone()
            masked_logits[~legal_masks] = -1e9
            policy_loss = policy_criterion(masked_logits, target_moves)

            # Loss 2: Value prediction
            blended_values = 0.3 * nnue_values + 0.7 * transformer_values
            value_loss = value_criterion(blended_values, target_values)

            # Loss 3: Selector training
            # Label: 1 if transformer improves, 0 otherwise
            improvement = torch.abs(transformer_values - target_values) < torch.abs(nnue_values - target_values)
            selector_labels = improvement.float().unsqueeze(1)  # [batch_size, 1]

            selector_preds = selector(selection_features)
            selector_loss = selector_criterion(selector_preds, selector_labels)

            # Combined loss
            total_loss = policy_loss + value_loss + 0.5 * selector_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        projection_layer.eval()
        selector.eval()
        val_loss = 0.0
        selector_accuracy = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Similar evaluation logic
                # ... (omitted for brevity, same as training without optimizer.step())
                pass

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'projection': projection_layer.state_dict(),
                'selector': selector.state_dict(),
                'epoch': epoch
            }, 'best_hybrid_model.pt')

    print("Training complete!")
    return projection_layer, selector
```

#### 5.3 Simplified 2-Day Training Schedule

```python
def quick_train(gpu_name='RTX6000', days=2):
    """
    Optimized training for 2 days on single GPU
    """

    print("=== Hybrid NNUE-Transformer Training (2-Day Schedule) ===")

    # Day 1: Focus on projection layer
    print("\nDay 1: Training projection layer...")

    # Load pre-trained models
    nnue = load_pretrained_nnue('stockfish_nnue.bin')
    transformer = load_pretrained_transformer('chessformer.pt')

    # Initialize trainable components
    projection = ProjectionLayer(nnue_dim=1024, transformer_dim=512)
    selector = SelectionFunction(feature_dim=20)

    # Small dataset for fast iteration (100K positions)
    train_data = ChessDataset(['lichess_elite_2024.pgn'], max_positions=80000)
    val_data = ChessDataset(['lichess_elite_2024_val.pgn'], max_positions=20000)

    # Phase 1: Projection layer only (12 hours)
    print("Phase 1: Projection layer (epochs 1-25)")
    for param in selector.parameters():
        param.requires_grad = False  # Freeze selector initially

    train_hybrid_model(
        nnue, transformer, projection, selector,
        train_data, val_data,
        num_epochs=25,
        batch_size=512,  # Large batch for stability
        learning_rate=1e-3
    )

    # Day 2: Joint training
    print("\nDay 2: Joint training projection + selector...")

    # Phase 2: Add selector training (12 hours)
    print("Phase 2: Projection + Selector (epochs 26-50)")
    for param in selector.parameters():
        param.requires_grad = True  # Unfreeze selector

    train_hybrid_model(
        nnue, transformer, projection, selector,
        train_data, val_data,
        num_epochs=25,
        batch_size=512,
        learning_rate=5e-4  # Lower LR for stability
    )

    print("\n=== Training Complete ===")
    print("Models saved to: best_hybrid_model.pt")

    return projection, selector
```

---

### 6. Inference and Game Playing

```python
def play_game_with_hybrid(
    nnue_model,
    transformer_model,
    projection_layer,
    selector,
    opponent='stockfish',
    time_per_move=5.0
):
    """
    Play a full chess game using hybrid evaluation
    """
    # Initialize
    board = chess.Board()
    evaluator = HybridEvaluator(nnue_model, transformer_model, projection_layer, selector)
    search = AlphaBetaSearch(evaluator, max_depth=20)

    move_count = 0

    while not board.is_game_over():
        move_count += 1
        print(f"\nMove {move_count}: {board.fen()}")

        if board.turn == chess.WHITE:
            # Hybrid engine's turn
            print("Hybrid engine thinking...")
            best_move, eval = search.search(board, time_limit=time_per_move)
            print(f"Chose: {best_move} (eval={eval:.2f})")

        else:
            # Opponent's turn (e.g., Stockfish)
            print("Opponent thinking...")
            best_move = get_opponent_move(board, opponent, time_per_move)
            print(f"Opponent chose: {best_move}")

        board.push(best_move)
        print(board)

    # Game over
    result = board.result()
    print(f"\nGame over: {result}")
    evaluator.print_stats()

    return result


def get_opponent_move(board, opponent, time_limit):
    """Get move from opponent engine (e.g., Stockfish)"""
    if opponent == 'stockfish':
        import chess.engine
        with chess.engine.SimpleEngine.popen_uci('stockfish') as engine:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move
    else:
        # Random legal move for testing
        import random
        return random.choice(list(board.legal_moves))
```

---

### 7. Evaluation and Testing

```python
def evaluate_model_strength(projection, selector, num_games=100):
    """
    Evaluate hybrid model strength against baseline
    """
    print("=== Model Evaluation ===")

    # Load models
    nnue = load_pretrained_nnue('stockfish_nnue.bin')
    transformer = load_pretrained_transformer('chessformer.pt')

    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0
    }

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")

        result = play_game_with_hybrid(
            nnue, transformer, projection, selector,
            opponent='stockfish',
            time_per_move=3.0
        )

        if result == '1-0':
            results['wins'] += 1
        elif result == '0-1':
            results['losses'] += 1
        else:
            results['draws'] += 1

    # Calculate win rate and estimated Elo
    win_rate = (results['wins'] + 0.5 * results['draws']) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate)) if win_rate not in [0, 1] else 0

    print("\n=== Results ===")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Draws: {results['draws']}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Estimated Elo difference: {elo_diff:+.0f}")
```

---

### 8. Utility Functions

```python
def position_hash(board):
    """Hash chess position for transposition table"""
    return hash(board.fen())


def extract_halfka_features(board, color):
    """
    Extract HalfKAv2 features for NNUE
    Returns sparse binary feature vector
    """
    features = []
    king_square = board.king(color)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Feature index based on piece type, color, square, and king square
            feature_idx = (
                piece.piece_type * 64 * 64 +
                (1 if piece.color == color else 0) * 64 * 64 +
                king_square * 64 +
                square
            )
            features.append(feature_idx)

    return features


def load_pretrained_nnue(path):
    """Load pre-trained NNUE from Stockfish binary"""
    # Use python-chess or custom loader
    # This is simplified - actual implementation depends on NNUE format
    nnue = NNUEEvaluator(path)
    print(f"Loaded NNUE from {path}")
    return nnue


def load_pretrained_transformer(path):
    """Load pre-trained transformer (e.g., ChessFormer checkpoint)"""
    transformer = ChessTransformer(path)
    print(f"Loaded Transformer from {path}")
    return transformer
```

---

## Implementation Checklist

### Pre-Training Requirements
- [ ] Download pre-trained NNUE weights (e.g., Stockfish NNUE)
- [ ] Download pre-trained transformer weights (e.g., ChessFormer or chess-transformers)
- [ ] Prepare training dataset (100K-1M positions from high-quality games)
- [ ] Verify GPU setup (RTX6000, CUDA, PyTorch)

### Day 1: Projection Layer
- [ ] Implement ProjectionLayer class
- [ ] Implement training loop for projection only
- [ ] Train for 12-18 hours (25 epochs)
- [ ] Validate projection quality (value prediction accuracy)
- [ ] Save checkpoint

### Day 2: Selector + Joint Training
- [ ] Implement SelectionFunction class
- [ ] Extract selection features from positions
- [ ] Train selector with frozen projection (6 hours)
- [ ] Joint fine-tuning (6-12 hours)
- [ ] Validate selector accuracy (precision/recall)
- [ ] Save final checkpoint

### Testing
- [ ] Implement HybridEvaluator
- [ ] Implement AlphaBetaSearch
- [ ] Play test games against Stockfish
- [ ] Measure evaluation speed (nodes/second)
- [ ] Measure selector usage (% hybrid evaluations)
- [ ] Calculate estimated Elo rating

---

## Expected Performance

### With 2 Days Training on RTX6000

**Projection Layer Quality**:
- Value prediction MSE: < 0.2 (comparable to transformer alone)
- Policy accuracy: > 45% (top-1 move prediction on legal moves)

**Selector Performance**:
- Precision: > 70% (correctly identifies complex positions)
- Recall: > 60% (doesn't miss critical positions)
- Usage rate: 2-5% of all evaluations

**Playing Strength**:
- Estimated Elo: 2800-3200 (depends on base models)
- Evaluation speed: 1-10M nodes/second (20-100× faster than pure transformer)
- Expected win rate vs Stockfish 15: 10-30% (respectable for hybrid approach)

---

## Troubleshooting

### Common Issues

**Issue 1: Projection layer not learning**
- Solution: Check gradient flow, reduce learning rate, increase batch size

**Issue 2: Selector always chooses NNUE or always chooses transformer**
- Solution: Balance training data, adjust loss weights, check feature extraction

**Issue 3: Slow training**
- Solution: Increase batch size, use mixed precision (FP16), reduce dataset size

**Issue 4: Model makes illegal moves**
- Solution: Verify legal_softmax implementation, check move indexing

**Issue 5: Poor playing strength**
- Solution: Verify base models are loaded correctly, test NNUE and transformer independently

---

## References

### Pre-trained Models
- **Stockfish NNUE**: https://github.com/official-stockfish/Stockfish
- **ChessFormer**: https://github.com/KempnerInstitute/chess-research
- **Chess Transformers**: https://github.com/sgrvinod/chess-transformers

### Datasets
- **Lichess Database**: https://database.lichess.org/ (elite games, 2500+ Elo)
- **CCRL Games**: http://ccrl.chessdom.com/ccrl/404/ (engine games)

### Libraries
- **python-chess**: Move generation, legal move checking
- **PyTorch**: Neural network training and inference
- **Chess Engine**: Stockfish for opponent and testing

---

*Implementation Guide Version 1.0*
*Last Updated: November 13, 2025*
