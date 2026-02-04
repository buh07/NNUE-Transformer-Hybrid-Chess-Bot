import chess
import torch

from search import AlphaBetaSearch, TranspositionTable


class DummyEvaluator:
    """Lightweight evaluator that returns deterministic scores."""

    def __init__(self, fn):
        self.fn = fn
        self.stats = {}

    def evaluate(self, board: chess.Board, depth_remaining: int = 0, legal_mask=None):
        value = float(self.fn(board))
        return torch.zeros(1), torch.tensor(value, dtype=torch.float32), "dummy"


def make_search(value_fn):
    evaluator = DummyEvaluator(value_fn)
    return AlphaBetaSearch(hybrid_evaluator=evaluator, max_depth=3, tt_size=256, use_quiescence=False)


def test_alpha_beta_returns_white_perspective_scores():
    search = make_search(lambda board: 100.0 if board.turn == chess.WHITE else -100.0)
    board = chess.Board()

    score_white, _ = search.alpha_beta(board, depth=0, alpha=-1000, beta=1000, is_maximizing=True)
    assert score_white == 100.0

    board.push(chess.Move.from_uci("e2e4"))
    score_black, _ = search.alpha_beta(board, depth=0, alpha=-1000, beta=1000, is_maximizing=False)
    assert score_black == -100.0


def test_transposition_table_exact_entries():
    search = make_search(lambda board: 42.0)
    board = chess.Board()

    score, _ = search.alpha_beta(board, depth=1, alpha=-1000, beta=1000, is_maximizing=True)
    assert score == 42.0

    zobrist = board._transposition_key()
    entry = search.tt.table[zobrist]
    assert entry[2] == TranspositionTable.EXACT


def test_transposition_table_upper_bound_entries():
    search = make_search(lambda board: -75.0)
    board = chess.Board()

    score, _ = search.alpha_beta(board, depth=1, alpha=0.0, beta=1000.0, is_maximizing=True)
    assert score == -75.0

    zobrist = board._transposition_key()
    entry = search.tt.table[zobrist]
    assert entry[2] == TranspositionTable.UPPER_BOUND
