import time
import chess
import traceback
import torch

# Directly instantiate HybridChessBot and run safe eval timings without transformer
from HybridChessBot import HybridChessBot


def time_eval_nnue_only(bot, board, runs=5, cache_size=None):
    he = bot.hybrid_evaluator
    if cache_size is not None:
        he._cache_size = int(cache_size)

    # Force selector to never use transformer
    he.selector.should_use_transformer = lambda feats: False
    he.selector.should_use_transformer_batch = lambda feats: torch.zeros(len(feats), dtype=torch.bool)

    times = []
    for i in range(runs):
        t0 = time.time()
        res = he.evaluate(board, depth_remaining=1)
        t1 = time.time()
        times.append(t1 - t0)
    return times


if __name__ == '__main__':
    start = time.time()
    try:
        print('=== Minimal Safe NNUE-only Bench ===')
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('CUDA device count:', torch.cuda.device_count())
            try:
                print('Torch current device:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
            except Exception:
                pass

        print('\nInstantiating HybridChessBot (loads models; may take a few seconds)...')
        bot = HybridChessBot(checkpoint='checkpoints/best_phase2.pt', depth=1, time_limit=0.1, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
        load_done = time.time()
        print(f'Loaded bot in {load_done - start:.3f}s')

        board = chess.Board()

        # NNUE-only timings
        print('\nNNUE-only (forced)')
        bot.hybrid_evaluator.reset_stats()
        times = time_eval_nnue_only(bot, board, runs=5, cache_size=bot.hybrid_evaluator._cache_size)
        print('Times (s):', ['{:.6f}'.format(t) for t in times])

        total_done = time.time()
        print(f'\nTotal script time: {total_done - start:.3f}s')

    except Exception as e:
        print('ERROR DURING SAFE MINIMAL EVAL:', e)
        traceback.print_exc()
        raise
