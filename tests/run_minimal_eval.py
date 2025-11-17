import time
import chess
import traceback
import torch

# Directly instantiate HybridChessBot and run repeatable eval timings
from HybridChessBot import HybridChessBot


def time_eval(bot, board, runs=3, force_hybrid=None, cache_size=None):
    he = bot.hybrid_evaluator
    # Optionally set cache size
    if cache_size is not None:
        he._cache_size = int(cache_size)

    # Monkeypatch selector decisions if requested
    if force_hybrid is True:
        he.selector.should_use_transformer = lambda feats: True
        he.selector.should_use_transformer_batch = lambda feats: torch.ones(len(feats), dtype=torch.bool)
    elif force_hybrid is False:
        he.selector.should_use_transformer = lambda feats: False
        he.selector.should_use_transformer_batch = lambda feats: torch.zeros(len(feats), dtype=torch.bool)

    times = []
    methods = []
    for i in range(runs):
        t0 = time.time()
        try:
            res = he.evaluate(board, depth_remaining=1)
            # normalize return (policy, value, method)
            if len(res) == 3:
                policy, value, method = res
            else:
                policy, value = res
                method = 'unknown'
        except Exception as e:
            print('Eval failed:', e)
            raise
        t1 = time.time()
        times.append(t1 - t0)
        methods.append(method)
    return times, methods


if __name__ == '__main__':
    start = time.time()
    try:
        print('=== Minimal Hybrid Evaluator Bench ===')
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

        # Case 1: Hybrid evaluations with cache enabled (default)
        print('\nCase 1: Hybrid (force transformer) with cache enabled (default)')
        bot.hybrid_evaluator.reset_stats()
        bot.hybrid_evaluator._cache_size = getattr(bot.hybrid_evaluator, '_cache_size', 1024)
        times1, methods1 = time_eval(bot, board, runs=3, force_hybrid=True, cache_size=bot.hybrid_evaluator._cache_size)
        print('Times (s):', ['{:.6f}'.format(t) for t in times1])
        print('Methods:', methods1)

        # Case 2: Hybrid evaluations with cache disabled
        print('\nCase 2: Hybrid (force transformer) with cache DISABLED')
        bot.hybrid_evaluator.reset_stats()
        times2, methods2 = time_eval(bot, board, runs=3, force_hybrid=True, cache_size=0)
        print('Times (s):', ['{:.6f}'.format(t) for t in times2])
        print('Methods:', methods2)

        # Case 3: NNUE-only evaluations
        print('\nCase 3: NNUE-only (force no transformer)')
        bot.hybrid_evaluator.reset_stats()
        times3, methods3 = time_eval(bot, board, runs=3, force_hybrid=False, cache_size=0)
        print('Times (s):', ['{:.6f}'.format(t) for t in times3])
        print('Methods:', methods3)

        total_done = time.time()
        print(f'\nTotal script time: {total_done - start:.3f}s')

    except Exception as e:
        print('ERROR DURING MINIMAL EVAL:', e)
        traceback.print_exc()
        raise
