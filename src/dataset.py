"""
Dataset for training the Hybrid NNUE-Transformer Chess Engine
"""

import torch
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.chess_utils import (
    extract_selection_features,
    get_legal_move_mask,
    move_to_index
)
import config


class ChessDataset(Dataset):
    """
    Dataset of chess positions with labels for training
    """
    
    def __init__(self, pgn_files: List[str], max_positions: int = 100000, 
                 use_stockfish_targets: bool = False, stockfish_depth: int = 10):
        """
        Args:
            pgn_files: List of PGN files with high-quality games
            max_positions: Maximum positions to extract
            use_stockfish_targets: If True, use Stockfish evaluations as target values
            stockfish_depth: Depth for Stockfish evaluation (default 10)
        """
        self.positions = []
        self.max_positions = max_positions
        self.use_stockfish_targets = use_stockfish_targets
        self.stockfish_depth = stockfish_depth
        self.stockfish_evaluator = None
        
        if pgn_files:
            self.load_games(pgn_files, max_positions)
        else:
            print("Warning: No PGN files provided")
    
    def load_games(self, pgn_files: List[str], max_positions: int):
        """
        Extract positions from PGN files
        
        Args:
            pgn_files: List of PGN file paths
            max_positions: Maximum number of positions to extract
        """
        count = 0
        
        for pgn_file in pgn_files:
            if not os.path.exists(pgn_file):
                print(f"Warning: PGN file not found: {pgn_file}")
                continue
            
            print(f"Loading positions from {pgn_file}...")
            
            with open(pgn_file) as f:
                while count < max_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Extract positions from game
                    board = game.board()
                    moves = list(game.mainline_moves())
                    
                    for move_idx, move in enumerate(moves):
                        # Store position before move
                        fen = board.fen()
                        result = game.headers.get('Result', '*')
                        
                        # Store position info (will compute value on-the-fly if using Stockfish)
                        if not self.use_stockfish_targets:
                            # Convert game result to value
                            if result == '1-0':
                                value = 1.0 if board.turn == chess.WHITE else -1.0
                            elif result == '0-1':
                                value = -1.0 if board.turn == chess.WHITE else 1.0
                            else:
                                value = 0.0
                        else:
                            # Will compute with Stockfish during __getitem__
                            value = None
                        
                        self.positions.append({
                            'fen': fen,
                            'move': move.uci(),
                            'value': value,
                            'depth': len(moves) - move_idx  # Remaining moves
                        })
                        
                        board.push(move)
                        count += 1
                        
                        if count >= max_positions:
                            break
        
        print(f"Loaded {len(self.positions)} positions from {len(pgn_files)} file(s)")
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample
        
        Returns:
            dict with keys:
                - board: Chess board object
                - selection_features: Features for selector [20]
                - legal_mask: Legal move mask [num_moves]
                - target_move_index: Index of target move
                - target_value: Position value [-1, 1]
        """
        data = self.positions[idx]
        board = chess.Board(data['fen'])
        
        # Extract selection features
        depth_remaining = min(data['depth'], 15)  # Assume mid-search depth
        selection_features = extract_selection_features(board, depth_remaining)
        
        # Get legal move mask
        legal_mask = get_legal_move_mask(board)
        
        # Target move index
        target_move = chess.Move.from_uci(data['move'])
        target_index = move_to_index(target_move)
        
        # Get target value
        if self.use_stockfish_targets:
            # Lazy initialization of Stockfish evaluator
            if self.stockfish_evaluator is None:
                from utils.stockfish_eval import StockfishEvaluator
                self.stockfish_evaluator = StockfishEvaluator(depth=self.stockfish_depth)
            target_value = self.stockfish_evaluator.evaluate_position(board)
        else:
            target_value = data['value']
        
        return {
            'board': board,
            'selection_features': selection_features,
            'legal_mask': legal_mask,
            'target_move_index': target_index,
            'target_value': target_value
        }
    
    def __del__(self):
        """Cleanup Stockfish evaluator if it exists"""
        if self.stockfish_evaluator is not None:
            self.stockfish_evaluator.close()


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader
    Handles chess.Board objects which can't be batched automatically
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    boards = [item['board'] for item in batch]
    selection_features = torch.stack([item['selection_features'] for item in batch])
    legal_masks = torch.stack([item['legal_mask'] for item in batch])
    target_move_indices = torch.tensor([item['target_move_index'] for item in batch], dtype=torch.long)
    target_values = torch.tensor([item['target_value'] for item in batch], dtype=torch.float32)
    
    return {
        'boards': boards,
        'selection_features': selection_features,
        'legal_masks': legal_masks,
        'target_move_indices': target_move_indices,
        'target_values': target_values
    }


def create_dataloaders(train_pgn_files: List[str], val_pgn_files: List[str],
                      batch_size: int = None, num_workers: int = 0,
                      use_stockfish_targets: bool = False, stockfish_depth: int = 10) -> tuple:
    """
    Create training and validation dataloaders
    
    Args:
        train_pgn_files: List of training PGN files
        val_pgn_files: List of validation PGN files (can be same as train)
        batch_size: Batch size (default from config)
        num_workers: Number of worker processes for data loading
        use_stockfish_targets: If True, use Stockfish evaluations as target values
        stockfish_depth: Depth for Stockfish evaluation
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Create combined dataset first
    total_positions = config.MAX_TRAIN_POSITIONS + config.MAX_VAL_POSITIONS
    full_dataset = ChessDataset(train_pgn_files, max_positions=total_positions,
                                use_stockfish_targets=use_stockfish_targets,
                                stockfish_depth=stockfish_depth)
    
    # Split into train/val
    train_size = min(config.MAX_TRAIN_POSITIONS, len(full_dataset))
    val_size = min(config.MAX_VAL_POSITIONS, len(full_dataset) - train_size)
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def create_dummy_dataset(num_positions: int = 1000) -> ChessDataset:
    """
    Create dummy dataset for testing (generates random positions)
    
    Args:
        num_positions: Number of positions to generate
    
    Returns:
        dataset: ChessDataset with random positions
    """
    dataset = ChessDataset([], max_positions=0)
    
    import random
    
    for i in range(num_positions):
        # Start from initial position
        board = chess.Board()
        
        # Make random moves
        num_moves = random.randint(5, 20)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        # Get a random legal move as target
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            continue
        
        target_move = random.choice(legal_moves)
        value = random.uniform(-1.0, 1.0)
        
        dataset.positions.append({
            'fen': board.fen(),
            'move': target_move.uci(),
            'value': value,
            'depth': random.randint(5, 15)
        })
    
    print(f"Created dummy dataset with {len(dataset.positions)} positions")
    return dataset


if __name__ == '__main__':
    # Test dataset
    print("Testing ChessDataset...")
    
    # Create dummy dataset for testing
    dataset = create_dummy_dataset(num_positions=100)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Board: {sample['board'].fen()}")
    print(f"Selection features shape: {sample['selection_features'].shape}")
    print(f"Legal mask shape: {sample['legal_mask'].shape}")
    print(f"Target move index: {sample['target_move_index']}")
    print(f"Target value: {sample['target_value']:.2f}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    print(f"Batch boards: {len(batch['boards'])}")
    print(f"Batch selection features shape: {batch['selection_features'].shape}")
    print(f"Batch legal masks shape: {batch['legal_masks'].shape}")
    print(f"Batch target indices shape: {batch['target_move_indices'].shape}")
    print(f"Batch target values shape: {batch['target_values'].shape}")
    
    print("\nDataset test complete!")
