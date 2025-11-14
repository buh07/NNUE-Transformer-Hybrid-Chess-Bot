"""
Test the new strategic routing features and selector logic
"""

import chess
import sys
import os
sys.path.append('src')

from utils.chess_utils import extract_selection_features

# Test positions with expected routing decisions

test_cases = [
    {
        'name': 'Starting Position',
        'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'expected_routing': 'Strategic? (closed, low tactics)',
        'reason': 'Closed pawn structure, no forcing moves'
    },
    {
        'name': 'Open Sicilian (Tactical)',
        'fen': 'r1bqkb1r/pp2pppp/2np1n2/2p5/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6',
        'expected_routing': 'NNUE (open, tactical)',
        'reason': 'Open position, many captures possible'
    },
    {
        'name': 'King and Pawn Endgame',
        'fen': '8/5pk1/6p1/7p/6PP/5PK1/8/8 w - - 0 1',
        'expected_routing': 'Strategic (endgame)',
        'reason': '6 pieces, positional understanding critical'
    },
    {
        'name': 'Closed French Defense',
        'fen': 'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3',
        'expected_routing': 'Strategic? (closed center)',
        'reason': 'Locked pawn center, strategic planning needed'
    },
    {
        'name': 'Tactical Position with Check',
        'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
        'expected_routing': 'NNUE (tactical)',
        'reason': 'Many pieces, tactical threats, captures available'
    },
    {
        'name': 'Rook Endgame',
        'fen': '6k1/5ppp/8/3R4/8/8/5PPP/6K1 w - - 0 1',
        'expected_routing': 'Strategic (endgame)',
        'reason': '8 pieces, positional play important'
    },
    {
        'name': 'Material Imbalance (Queen vs Rooks)',
        'fen': '4r1k1/5ppp/8/8/3Q4/8/5PPP/6K1 w - - 0 1',
        'expected_routing': 'NNUE (material imbalance)',
        'reason': 'Material imbalance needs precise evaluation'
    },
    {
        'name': 'Complex Middlegame',
        'fen': 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8',
        'expected_routing': 'NNUE (many pieces, tactical)',
        'reason': 'Many pieces, tactical possibilities'
    }
]

print("=" * 80)
print("STRATEGIC ROUTING FEATURE EXTRACTION TEST")
print("=" * 80)

for test in test_cases:
    print(f"\n{test['name']}")
    print(f"FEN: {test['fen']}")
    print(f"Expected: {test['expected_routing']}")
    print(f"Reason: {test['reason']}")
    print("-" * 80)
    
    board = chess.Board(test['fen'])
    features = extract_selection_features(board, depth_remaining=10)
    
    # Key features for strategic routing
    print(f"Feature Analysis:")
    print(f"  [0] Legal moves ratio:     {features[0].item():.3f} {'(LOW=closed)' if features[0] < 0.4 else ''}")
    print(f"  [1] Captures ratio:        {features[1].item():.3f} {'(LOW=strategic)' if features[1] < 0.2 else ''}")
    print(f"  [4] Endgame flag:          {features[4].item():.3f} {'(ENDGAME)' if features[4] > 0.5 else ''}")
    print(f"  [7] Check flag:            {features[7].item():.3f} {'(CHECK!)' if features[7] > 0.5 else ''}")
    print(f"  [16] Forcing move ratio:   {features[16].item():.3f} {'(HIGH=tactical)' if features[16] > 0.3 else '(LOW=strategic)'}")
    print(f"  [17] Pawn chains:          {features[17].item():.3f} {'(HIGH=closed)' if features[17] > 0.3 else ''}")
    print(f"  [18] Piece coordination:   {features[18].item():.3f}")
    print(f"  [19] Material imbalance:   {features[19].item():.3f} {'(IMBALANCED!)' if features[19] > 0.15 else ''}")
    
    # Apply strategic routing logic (from train.py)
    endgame = features[4] > 0.5
    few_captures = features[1] < 0.2
    not_in_check = features[7] < 0.5
    closed_position = features[0] < 0.4
    
    is_strategic = ((endgame or closed_position) and few_captures and not_in_check)
    
    routing = "TRANSFORMER (Strategic)" if is_strategic else "NNUE (Tactical)"
    print(f"\n  → ROUTING DECISION: {routing}")
    print(f"     Logic: endgame={endgame}, closed={closed_position}, few_captures={few_captures}, not_check={not_in_check}")

import torch

print("\n" + "=" * 80)
print("Feature vector shape:", features.shape)
print("All features valid:", all(not torch.isnan(features).item() for _ in [0]) and all(not torch.isinf(features).item() for _ in [0]))
print("=" * 80)
print("\n✓ Strategic routing feature extraction test completed!")
