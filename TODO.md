# Hybrid Stockfish Bot – Roadmap

## Guiding principles
- Keep the transformer selector, dataset tooling, and experimentation-friendly diagnostics in Python so we can iterate quickly.
- Push deterministic, performance-critical search routines (alpha-beta, q-search, TT management, NNUE delta updates) into `Stockfish_hybrid` when profiling proves they are the bottleneck.
- Maintain a clean separation: Python orchestrates game/ML logic, C++ exposes fast primitives via the pybind11 binding.
- Always profile before porting. Measure node throughput, TT hit rate, and move ordering quality so optimizations target the right layer.

## Recently completed
- [x] Cloned upstream Stockfish into `Stockfish_hybrid/` and ensured a pristine upstream `Stockfish/` remains untouched for comparisons.
- [x] Added `build_stockfish_hybrid.sh` plus `bindings/pybind11/build_binding.sh` to compile `libstockfish.{a,so}` and the `stockfish_hybrid_binding` module in one step.
- [x] Updated `NNUEEvaluator` to prefer the pybind11 `StockfishHybridEngine`, fall back to the legacy embedded `.so`, and finally to the subprocess path.
- [x] Logged two Stockfish-vs-Hybrid match suites (depth-limited and equal-time) under `match_results_*.{txt,json}` to baseline current strength.

---

## Step 1 – Profile the Python search stack (in progress)
- [x] Instrument `src/search.py` (alpha-beta, quiescence, TT lookup/store, move ordering) to emit per-depth statistics: nodes, branching factor, TT hit/miss, pruning counts, cache latency.
- [x] Add lightweight timers inside `HybridEvaluator` to measure NNUE calls (pybind, embedded, subprocess) and transformer forward passes.
- [x] Create `tools/profile_hybrid_search.py` that runs a mini gauntlet (e.g., 32 tactical positions) and dumps a JSON/CSV profile for later comparison.
- [x] Summarize the hot spots (e.g., TT misses due to shallow table, move ordering regressions) and decide which ones demand C++ acceleration.
  - Latest profile (`logs/profiles/profile_20251205_220600_gpu1.json`) shows ~234 nodes/pos in 0.56 s (≈412 NPS), TT hit rate only 1.3 %, q-search consuming ~277 nodes/pos, and NNUE forwards dominated by the embedded binding (10 073 calls = 5.6 s total) with zero transformer usage. Focus optimizations on TT effectiveness, reducing quiescence explosions, and restoring transformer routing.

## Step 2 – Decide Python vs C++ ownership boundaries
- [x] From the profiling report, classify each subsystem as **keep in Python**, **lift to C++**, or **hybrid**:
  - **Lift to C++ soon**: transposition-table storage/lookup (Python dict + LRU too slow, 1.3 % hit rate); quiescence search (dominates nodes, better handled alongside move generator); move ordering heuristics (captures/checks only) should leverage Stockfish move pickers.
  - **Hybrid approach**: Alpha-beta driver stays Python-side for experimentation but should delegate leaf evaluations, TT probes, and move lists via the binding. Time manager and transformer selector stay Python but need hooks to inject priorities/time budgets into the C++ search.
  - **Keep in Python**: training/selector logic, diagnostics, profiling harnesses; transformer feature extraction. NNUE accumulator updates must remain accessible via Python callbacks even if the binding exposes incremental state.
- [x] Draft API sketches for logic migrating to C++:
  - `StockfishHybridEngine` should expose: `prepare_fen(fen)`, `generate_moves(board_state) -> list[move handles]`, `probe_tt(zobrist, depth, alpha, beta) -> optional result`, `store_tt(zobrist, depth, score, flag, move)`, `quiescence(board_state, alpha, beta, max_depth)`; and a `search_with_callbacks(board_state, depth, time_ms, hooks)` entry point that lets Python inject transformer adjustments or stop early.
  - Incremental NNUE state export via `get_accumulator()` or a streaming delta feed so Python selectors/transformers can reuse features without re-evaluating.
- [x] Document transformer data flow when logic migrates:
  - Keep selection features (`extract_selection_features`) in Python; if search moves to C++, expose a lightweight callback every time a node is evaluated so Python can request transformer scores or adjust move ordering.
  - Maintain a shared tensor ring buffer (torch tensor pinned in Python) where the binding writes NNUE accumulator slices; transformer/projection layers read from that buffer without recomputing.
  - Provide a C++ hook `request_transformer_eval(node_info) -> float` that blocks until Python returns a value, ensuring the transformer path still runs on GPU while core search stays native.

## Step 3 – Integrate bindings with packaging & CI
- [x] Added `bindings/pybind11/build_binding.sh` to rebuild Stockfish_hybrid and the pybind11 extension together.
- [x] Detect CPU feature sets (SSE2, AVX2, VNNI) and GPU toggles via env vars (`HYBRID_STOCKFISH_ARCH`, `HYBRID_STOCKFISH_JOBS`), falling back gracefully at runtime.
- [x] Extend `bindings/pybind11/README.md` (and repo root README) with CI-friendly build instructions, including how to cache NNUE weights and how to verify the `.so` signature.
- [x] Add a smoke-test script (invoked by CI) that `pip install`s the wheel, imports `stockfish_hybrid_binding`, and runs `engine.eval_fen(startpos)`.

## Step 4 – Update the Hybrid bot to consume the new module
- [x] Replace any remaining direct imports of `stockfish_binding.cpython-*.so` with a helper that prefers the pybind11 module and surfaces clear warnings when falling back.
- [x] Extend diagnostics/benchmarks so every gauntlet records which backend (pybind, legacy embedded, subprocess) served each evaluation.
- [x] Update `play_vs_stockfish.py` and training scripts to accept `--engine-backend` so matches/tests can compare the new binding versus legacy paths automatically.
- [x] After backend unification, retire the legacy `.so` from default installs (keep an opt-in flag for regression testing only).

## Step 5 – Improve search throughput (Python + C++)
- [x] Fix outstanding bugs already identified (quiescence perspective, TT entry types, etc.) and add regression tests in `tests/test_search_perspective.py`.
- [ ] Experiment with delegating move ordering & TT store/load directly to the binding to cut Python overhead.
- [ ] Consider exposing Stockfish's internal search stack via pybind (pass feature deltas + transformer adjustments in callbacks) if Python remains too slow after profiling.
- [ ] Track node-per-second improvements after each search change using the profiling harness from Step 1.

## Step 6 – Diagnostics & benchmarking
- [ ] Expand the existing match harness to support identical time controls, reproducible seeds, and automatic log summaries (Elo, SPRT signals).
- [ ] Maintain side-by-side benchmarks (with/without transformer augmentation, with/without pybind binding) to ensure each code path is monitored.
- [ ] Add dashboards/notebooks that visualize TT hit rate, time allocation per move, and evaluation drift versus pure Stockfish.

## Step 7 – Transformer & activation-function work
- [ ] Retrain or fine-tune the transformer activation/selector using the latest NNUE data; log performance impacts separately from search tweaks.
- [ ] Ensure the binding exposes the incremental NNUE state the selector needs (or cache it inside Python) once Step 2 decisions land.
- [ ] Evaluate whether activation decisions should move C++-side (e.g., scoring hooks) or stay Python-side via fast callbacks.

---

### Notes / references
- Pybind binding sources live in `bindings/pybind11/`.
- Match logs: `match_results_20251205_*.{txt,json}`.
- Profiling scripts and future notebooks should live under `tools/` to keep the repo tidy.
