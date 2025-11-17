import time
import chess
import traceback

from HybridChessBot import create_hybrid_bot

if __name__ == '__main__':
    start = time.time()
    try:
        bot = create_hybrid_bot(checkpoint='checkpoints/best_phase2.pt', depth=2, time_limit=2.0, verbose=True)
        init_done = time.time()
        board = chess.Board()
        move = bot.choose_move(board)
        end = time.time()
        print('MOVE:', move)
        print('INIT_TIME:', init_done - start)
        print('CHOOSE_TIME:', end - init_done)
        print('TOTAL_TIME:', end - start)
    except Exception as e:
        print('ERROR DURING INSTANTIATE/CHOOSE:', e)
        traceback.print_exc()
        raise
