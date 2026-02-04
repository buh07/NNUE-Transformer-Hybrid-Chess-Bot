# Hybrid NNUE–Transformer Chess Bot

This repository contains a playable chess engine that couples Stockfish’s NNUE evaluator with a transformer policy/value head and a selector/projection bridge. The included scripts let you train the projection/selector, run quick evaluations, and play games via `python-chess`.

## Repository Layout

```
HybridChessBot.py          # Main bot class (alpha-beta + hybrid evaluator)
ChessGame.py               # Simple game harness compatible with python-chess
play_vs_stockfish.py       # CLI for playing against the bot or Stockfish
config.py                  # Paths + hyper-parameters (edit for your system)
src/                       # All model components (NNUE bridge, transformer wrapper, search, utils)
chess-transformers/        # CT-EFT-85 transformer code + checkpoints
Stockfish/                 # Stockfish binary + NNUE networks
checkpoints/               # Projection + selector weights (e.g., best_phase1.pt, best_phase2.pt, final_model.pt)
tests/                     # Training scripts, benchmarks, diagnostics
chess_env/                 # (Optional) local virtual environment used for development
logs/                      # Latest training logs (trimmed to current session)
misc/                      # Archived docs, experiment outputs, and historic logs
```

The files in `misc/` are not required for runtime; everything else is used directly or by the training/evaluation scripts.

## Requirements

* Python 3.10+.
* `python-chess`, PyTorch 2.x, and the other packages listed in `requirements.txt`.
* To use the new embedded Stockfish binding, ensure `pybind11` is installed and run:
  ```bash
  HYBRID_STOCKFISH_ARCH=x86-64-avx2 \
  HYBRID_STOCKFISH_JOBS=32 \
  bindings/pybind11/build_binding.sh
  export PYTHONPATH="$(pwd)/bindings/pybind11:${PYTHONPATH}"
  ```
  (`HYBRID_STOCKFISH_ARCH`/`HYBRID_STOCKFISH_JOBS` default to `x86-64` and `nproc`; override them for CI or cross-compiles.)
* The legacy embedded binding (`stockfish_binding.cpython-*.so`) is disabled by default. To opt in (for regression testing only) set `HYBRID_ENABLE_LEGACY_BINDING=1` before launching the bot.
* Stockfish binary (already included under `Stockfish/src/stockfish`) with the NNUE nets referenced by `config.py`.
* Transformer weights (`chess-transformers/checkpoints/CT-EFT-85/CT-EFT-85.pt`).

### Quick Setup

```bash
python -m venv chess_env
source chess_env/bin/activate       # On Windows: chess_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x Stockfish/src/stockfish    # Ensure the engine is executable
```

Verify the paths in `config.py` match your local directory layout (especially `Stockfish`, `chess-transformers`, and `checkpoints`).

## Using the Bot with `python-chess`

### 1. Load the Bot Programmatically

```python
import chess
from HybridChessBot import HybridChessBot

bot = HybridChessBot(
    checkpoint="checkpoints/final_model.pt",  # or best_phase*.pt
    depth=4,
    time_limit=1.0,
    verbose=False
)

board = chess.Board()
while not board.is_game_over():
    if board.turn:
        move = bot.choose_move(board)
        board.push(move)
    else:
        # Example opponent: Stockfish via python-chess
        import chess.engine
        with chess.engine.SimpleEngine.popen_uci("Stockfish/src/stockfish") as sf:
            sf_move = sf.play(board, chess.engine.Limit(time=0.1)).move
        board.push(sf_move)

print(board.result())
```

The bot exposes `choose_move(board)` and has optional helper methods (`analyze_position`, `get_statistics`) for integration with other `python-chess` tooling.

### 2. Play from the CLI

```bash
source chess_env/bin/activate
export PYTHONPATH="$(pwd)/bindings/pybind11:${PYTHONPATH}"
python play_vs_stockfish.py --time 1.0 --depth 4 --color white --engine-backend pybind
```

This script spins up the hybrid evaluator and plays against Stockfish (or human input) via the console.

### 3. Minimal Evaluation / Diagnostics

* `python tests/run_minimal_eval.py` – quick functional test (loads weights, runs a few evals).
* `python tests/run_quick_instantiate.py` – sanity check that all weights load without errors.
* `python tests/run_benchmark.py` – rough throughput benchmark.

## Training (Projection + Selector)

Only the projection layer and selector are trained here; Stockfish’s NNUE network and the transformer weights remain frozen.

1. Prepare PGNs and set their paths in `config.PGN_FILES`. Adjust `MAX_TRAIN_POSITIONS` and `MAX_VAL_POSITIONS` for the amount of data you want to stream.
2. Run `python tests/run_minimal_eval_safe.py` to ensure Stockfish/NNUE integration works.
3. Launch training (GPU recommended):

```bash
source chess_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python tests/run_quick_instantiate.py  # optional warmup
CUDA_VISIBLE_DEVICES=0 python src/train.py
```

The trainer will run the two-phase schedule configured in `config.py`. Checkpoints are written to `checkpoints/` (e.g., `best_phase1.pt`, `best_phase2.pt`, `final_model.pt`). Use the latest checkpoint with `HybridChessBot(checkpoint=...)` when distributing the model.

## Packaging for Distribution

When zipping the repo for someone else:

* Include `HybridChessBot.py`, `ChessGame.py`, `play_vs_stockfish.py`, `config.py`.
* Include `src/`, `chess-transformers/` (with the CT-EFT-85 weights), `Stockfish/`, `checkpoints/`, and `requirements.txt`.
* Optionally include `logs/retrain_gpu4_20251116_230009.log` (or the latest log) for reproducibility notes.

Recipients can then extract the archive, install requirements, and run `python play_vs_stockfish.py` or embed `HybridChessBot` in their own `python-chess` applications.

## Support / Troubleshooting

* Ensure Stockfish’s binary path in `config.py` is valid and executable.
* If transformer weights fail to load, re-run `setup_pretrained_models.sh` or `download_best_models.sh`.
* For CUDA issues, set `DEVICE='cpu'` in config or run with `CUDA_VISIBLE_DEVICES=""`.

Feel free to open issues or contact the maintainer if you encounter unexpected behavior. Happy hacking!
