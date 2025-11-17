#!/usr/bin/env python3
"""
Quick test to verify all model weights are properly configured and loadable
"""

import os
import sys
sys.path.insert(0, '.')
import config

print('=' * 70)
print('Testing Weight Availability and Paths')
print('=' * 70)

# Check all weight paths
checks = [
    ('Stockfish Binary', config.STOCKFISH_BINARY_PATH),
    ('Stockfish NNUE', config.STOCKFISH_NNUE_PATH),
    ('ChessTransformer', config.TRANSFORMER_WEIGHTS_PATH),
    ('Hybrid Checkpoint', config.HYBRID_CHECKPOINT_PATH),
]

all_good = True
for name, path in checks:
    exists = os.path.exists(path) if path else False
    status = '✓' if exists else '✗'
    size = ''
    if exists and os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        size = f' ({size_mb:.1f} MB)'
    
    print(f'{status} {name:20s}: {path}{size}')
    if not exists:
        all_good = False

print('=' * 70)

if all_good:
    print('SUCCESS: All weight files found!')
    print('\nNow testing quick bot initialization (without running games)...')
    print('=' * 70)
    
    # Quick initialization test without playing
    from HybridChessBot import HybridChessBot
    import chess
    
    print('\nInitializing bot with verbose output...\n')
    bot = HybridChessBot(verbose=True, depth=3)
    
    print('\n' + '=' * 70)
    print('Bot Statistics:')
    print(f'  Device: {bot.device}')
    print(f'  Max depth: {bot.depth}')
    print(f'  Projection params: {bot._count_params(bot.projection):,}')
    print(f'  Selector params: {bot._count_params(bot.selector):,}')
    print(f'  NNUE using Stockfish: {bot.nnue.use_stockfish_engine}')
    print('=' * 70)
    print('\nSUCCESS: Bot ready with all weights loaded!')
    
else:
    print('ERROR: Some weight files are missing!')
    print('Please download all required model weights.')
    sys.exit(1)
