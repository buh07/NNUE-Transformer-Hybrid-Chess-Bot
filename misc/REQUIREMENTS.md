# Chess Bot Requirements

## Overview
This document describes the requirements for creating a chess bot that can play against the existing chess game implementation.

## Mandatory Requirements

### 1. Core Interface
Your chess bot **must** implement a `choose_move()` method with the following signature:

```python
def choose_move(self, board):
    """
    Choose a move given the current board state.
    
    Args:
        board: A chess.Board object representing the current game state
        
    Returns:
        A chess.Move object representing the chosen move
    """
    # Your implementation here
    return move
```

### 2. Dependencies
- **Required Library**: `python-chess`
  - Install with: `pip3 install python-chess`
  - Import with: `import chess`

### 3. Input/Output
- **Input**: Receives a `chess.Board` object as the `board` parameter
- **Output**: Must return a `chess.Move` object
- **Validity**: The returned move must be in `board.legal_moves`

### 4. Integration
Your bot must be compatible with the `ChessGame` class:
```python
from ChessGame import ChessGame
from YourBot import YourChessBot

player1 = YourChessBot()
player2 = SomeOtherBot()
game = ChessGame(player1, player2)
```

## Available Board Methods

### Accessing Game State
```python
board.legal_moves          # Generator of legal moves
board.turn                 # True if White's turn, False if Black's turn
board.is_game_over()       # Check if game has ended
board.is_checkmate()       # Check if current position is checkmate
board.is_stalemate()       # Check if current position is stalemate
board.is_check()           # Check if current player is in check
```

### Move Manipulation (for lookahead/simulation)
```python
board.push(move)           # Make a move (modifies board)
board.pop()                # Undo last move (modifies board)
board.copy()               # Create a copy of the board for simulation
```

### Move Creation
```python
chess.Move.from_uci("e2e4")        # Create move from UCI notation
list(board.legal_moves)            # Get list of all legal moves
```

## Optional Features

### Constructor Parameters
You may add constructor parameters for configuration:
```python
class MyChessBot:
    def __init__(self, depth=3, time_limit=5):
        self.max_depth = depth
        self.time_limit = time_limit
```

### Performance Tracking
You may track statistics for analysis:
```python
class MyChessBot:
    def __init__(self):
        self.node_count = 0
        self.node_count_by_depth = {}
```

### State Management
You may maintain internal state:
```python
class MyChessBot:
    def __init__(self):
        self.best_move = None
        self.last_value = 0
```

## Example Implementations

### Minimal Valid Bot
```python
import chess
import random

class MinimalBot:
    def __init__(self):
        pass
    
    def choose_move(self, board):
        moves = list(board.legal_moves)
        return random.choice(moves)
```

### Bot with Depth Parameter
```python
import chess

class DepthBot:
    def __init__(self, depth=3):
        self.max_depth = depth
    
    def choose_move(self, board):
        # Your search algorithm here
        moves = list(board.legal_moves)
        # ... implement your logic ...
        return moves[0]  # Return best move
```

## Reference Implementations

The workspace includes these reference implementations:
- `RandomAI.py` - Simple random move selection
- `HumanPlayer.py` - Human input interface
- `MinimaxAI.py` - Minimax with iterative deepening
- `AlphaBetaAI.py` - Alpha-beta pruning optimization

## Testing Your Bot

1. Create your bot file (e.g., `MyBot.py`)
2. Edit `test_chess.py`:
```python
from MyBot import MyBot
from MinimaxAI import MinimaxAI

player1 = MyBot()
player2 = MinimaxAI(3)

game = ChessGame(player1, player2)

while not game.is_game_over():
    print(game)
    game.make_move()
```
3. Run: `python test_chess.py`

## Common Patterns

### Evaluating Board State
```python
def evaluate(self, board):
    if board.is_checkmate():
        return float('inf') if board.turn != self.color else float('-inf')
    # Add your heuristic evaluation
    return score
```

### Minimax/Alpha-Beta Search
```python
def choose_move(self, board):
    best_move = None
    best_value = float('-inf')
    
    for move in board.legal_moves:
        board.push(move)
        value = self.minimax(board, depth - 1, False)
        board.pop()
        
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_move
```

## Notes
- The `ChessGame` class automatically alternates turns between player1 (White) and player2 (Black)
- Your bot will not know in advance whether it's playing White or Black - check `board.turn` each time
- The game handles move validation and game-over detection automatically
