# Stockfish Hybrid pybind11 Binding

This directory hosts the new pybind11 extension that gives Python direct access to the
`Stockfish_hybrid` fork. It is intended to replace the legacy
`stockfish_binding.cpython-*.so` by embedding the engine in-process.

## Prerequisites

1. Ensure `pybind11` and `python-chess` are installed in the active environment
   (already listed in `requirements.txt`).
2. The helper script below will rebuild `libstockfish.{a,so}` automatically, but you can
   still run `./build_stockfish_hybrid.sh` manually if you prefer.

## Building the module

```bash
HYBRID_STOCKFISH_ARCH=x86-64-avx2 \
HYBRID_STOCKFISH_JOBS=32 \
bindings/pybind11/build_binding.sh
```

This produces `stockfish_hybrid_binding.*.so` alongside the sources.

*The build helper automatically falls back to `nproc`/`sysctl` to detect CPU cores when `HYBRID_STOCKFISH_JOBS` is not set. Set `PYTHON_BIN` to point at a specific interpreter if needed.*

### CI-friendly build & smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
HYBRID_STOCKFISH_ARCH=x86-64-avx2 HYBRID_STOCKFISH_JOBS=8 bindings/pybind11/build_binding.sh
# Smoke test
python bindings/pybind11/smoke_test.py
```

The smoke test imports the module, constructs `StockfishHybridEngine`, runs `set_fen("startpos")`, and calls `evaluate`. Any failure in CI should stop the pipeline immediately.

## Usage example

```python
from stockfish_hybrid_binding import StockfishHybridEngine

engine = StockfishHybridEngine(binary_dir="Stockfish_hybrid/src")
engine.set_fen("startpos", moves=["e2e4", "e7e5"])
print(engine.evaluate())               # centipawns from side-to-move POV
print(engine.evaluate(white_pov=True)) # convert to White POV
```

Available methods:

* `load_networks(big_net, small_net)` – reload NNUE weights (paths relative to `binary_dir`
  unless absolute).
* `set_fen(fen="startpos", chess960=False, moves=[])` – reset the position and optionally
  apply a move list.
* `push_move("g1f3")` / `pop_move()` – incremental move management.
* `evaluate(white_pov=False)` – return the NNUE score in centipawns.
* `fen()` – retrieve the current FEN string.

Further functionality (policy access, incremental evaluation hooks, etc.) can be layered on
top of this scaffold as the hybrid bot’s needs grow.
