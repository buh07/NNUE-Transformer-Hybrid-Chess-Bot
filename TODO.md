# TODO

## 0. Search Stability Hotfixes (blocker for Stockfish parity)

- [ ] **Repair quiescence perspective sign flip**  
  1. Audit `AlphaBetaSearch.evaluate_position` (`src/search.py`) to confirm it always returns a White-centric score (it does).  
  2. Refactor `quiescence_search` so it stays consistent with that convention instead of using the negamax `-score` trick. A minimal fix: pass an `is_maximizing` flag (same as the main search) into quiescence, remove the unconditional negation, and compare `stand_pat`/child scores directly just as `alpha_beta` does.  
  3. After the change, add/extend a micro-test (e.g., new helper in `test_search.py` or `test_black_search_debug.py`) that constructs a simple capture-only line where Black is winning and ensures quiescence still returns a negative number.  
  4. Run `python test_search_perspective.py` and a short Stockfish match to confirm no more “false winning eval” spikes occur.

- [ ] **Store correct TT bounds**  
  1. In `AlphaBetaSearch.alpha_beta`, save `alpha_orig, beta_orig = alpha, beta` before the move loop.  
  2. Use those originals when deciding the entry type after the loop (`if best_score <= alpha_orig: UPPER`, `elif best_score >= beta_orig: LOWER`, else `EXACT`). This keeps TT semantics aligned with standard alpha-beta theory.  
  3. Add a regression check in `tests/test_search.py` that mocks a cached node and verifies we record at least one `EXACT` hit when the stored depth equals the search depth.  
  4. Re-run `python tests/test_search.py` and a quick Stockfish match to verify the TT hit rate climbs and Elo no longer collapses.

## 1. Verify Chessformer Value vs Baseline Stockfish (and retrain selector if needed)

- [ ] **Create reproducible ablations** – add an option to `HybridChessBot`/`HybridEvaluator` (e.g., `force_nnue_only`, `force_transformer`) so `play_vs_stockfish.py` and `tools/diagnose_transformer.py` can run matches where the transformer is never consulted. This makes it easy to compare NNUE-only, hybrid, and transformer-only variants with identical search settings.
- [ ] **Record match metrics side-by-side** – extend the logging in `play_vs_stockfish.py` to emit JSON/CSV summaries (`logs/chessformer_vs_stockfish/*.json`) that capture `HybridChessBot.get_statistics()` plus Stockfish match outcomes. Target test grid: SF Elo 1350/1500/1650/1800 with at least 6 games per level. Track transformer usage % (`HybridEvaluator.stats`), average eval difference, and win rate deltas.
- [ ] **Automate measurement harness** – refactor `tools/diagnose_transformer.py` so it can batch-run the above ablations, capturing:
  * selector probability histograms per phase (`opening/middlegame/endgame` from `extract_selection_features`),
  * value deltas between NNUE and transformer for every evaluated node,
  * top-k move agreement with Stockfish targets (can reuse `src/train.py` loss computation by saving tensors).
  Store raw samples under `logs/chessformer_eval/` to drive later analysis notebooks.
- [ ] **Label whether transformer truly helps** – build a script (e.g., `tools/build_selector_dataset.py`) that:
  * streams positions from `data/lichess_elite_*.pgn`,
  * runs NNUE-only and hybrid evaluations to get policy/value deltas,
  * labels positions where transformer improves Stockfish-aligned outcomes beyond a configurable threshold (centipawn swing, policy KL, or move match),
  * writes the dataset to Parquet/NPZ for selector training.
- [ ] **Retrain or redesign selector activation when hybrid is not a net gain** – if match logs show no Elo lift:
  * experiment with richer selector architectures (`src/models/selector.py`) that use `nn.SiLU`/`nn.GELU`, residual connections, or calibration layers,
  * augment `src/train.py` to support new loss terms (ROC-AUC, focal loss weighting, temperature scaling for `config.SELECTOR_THRESHOLD`),
  * add evaluation hooks (confusion matrix, ROC curve) that save to `logs/selector_eval.json`,
  * rerun `tests/test_strategic_routing.py` and a `play_vs_stockfish.py --diagnose-selector` mode to confirm the activation is now deciding correctly.
- [ ] **Document the findings** – summarize the ablation results (win rates, selector accuracy, transformer usage) in `README.md`/`misc/STOCKFISH_TESTING.md` so future training runs know whether Chessformer adds value and what activation settings were used.

## 2. Integrate Opening Book + 5-Piece Endgame Database

- [ ] **Opening book preparation**
  * Use the curated PGNs already under `data/` to build a Polyglot (or custom JSON) book via a new utility (`tools/build_opening_book.py`). Support filters such as minimum game count, move depth, and Elo thresholds.
  * Add `config.OPENING_BOOK_PATH`, `config.OPENING_BOOK_MAX_PLIES`, and `config.OPENING_BOOK_MIN_WEIGHT` so the engine can be configured without code edits.
- [ ] **Opening book runtime integration**
  * Implement `src/opening_book.py` with a small wrapper around `python-chess`’s `polyglot` reader (with caching and thread safety).
  * Extend `HybridChessBot` to consult the book inside `choose_move`: as long as the current line exists in the book and the chosen book move matches the game (`board.fen()` still in book), play the book suggestion immediately and skip alpha-beta.
  * Track when the bot has “deviated” (missing entry, low weight, or depth limit) and fall back to search. Record book usage stats in `get_statistics()` so they show up in match logs.
  * Update CLI scripts so users can disable book usage or choose alternative book files when testing.
- [ ] **Endgame tablebase setup**
  * Download Syzygy 5-piece tablebases and add their paths (e.g., `config.SYZYGY_PATHS` plus a helper `SYZYGY_MAX_PIECES = 5`).
  * Write `src/endgame_tablebase.py` that wraps `chess.syzygy.Tablebase` for probing WDL/DTZ, with graceful fallbacks when files are missing.
- [ ] **Endgame integration**
  * Modify `HybridEvaluator.evaluate` (or `AlphaBetaSearch.evaluate_position`) to detect when `len(board.piece_map()) <= 5` and call the tablebase before the hybrid logic. Return exact WDL/DTZ values (and optionally suggested move) so search can treat them as terminal nodes.
  * Allow `HybridChessBot` to shortcut directly to the tablebase-supplied move if it exists, bypassing search to avoid wasting time.
  * Cache tablebase probes for repeated endgame positions, similar to the existing evaluation cache.
- [ ] **Testing + regression guards**
  * Add unit tests under `tests/` for both systems (e.g., verifying the book keeps playing known theory lines until a deviation, verifying tablebase moves for KPK vs KP endgames).
  * Update documentation (`README.md`, new `docs/BOOK_AND_TABLEBASE.md`?) describing how to build/enable the databases and what fallback behavior to expect.

## 3. Make Time Control Fully Operational

- [ ] **Expose configuration knobs** – add CLI flags (`play_vs_stockfish.py`, `ChessGame.py`) for total time, increment, emergency reserve, and `moves_to_go`. Surface the same options in `HybridChessBot` factory helpers so scripts/tests don’t need manual edits.
- [ ] **Wire allocation into search** – currently `HybridChessBot.choose_move` computes `time_for_move`, but `AlphaBetaSearch` merely uses a flat `time_limit`. Thread the full `TimeManager.allocate_time` output through to search so:
  * iterative deepening can stop earlier using `TimeManager.should_stop_search(...)`,
  * quiescence/search nodes can respect `min_time` vs `allocated_time` (pass both down or poll a callback).
  * statistics about `allocated_time` vs `time_used` are logged per move.
- [ ] **Implement increment + book/tablebase awareness** – when an opening-book move or tablebase move is played instantly, feed the saved time back into `TimeManager` (so the reserve grows). Add logic so `allocate_time` treats deterministic book/tablebase moves as near-zero-cost.
- [ ] **Add diagnostics/tests** – create `tests/test_time_manager_integration.py` that runs a few plies with `use_time_management=True` and asserts:
  * total time decreases appropriately,
  * selector confidence history is populated,
  * emergency reserve is respected under forced low-time scenarios.
  * compare results with/without increment to ensure the reserve logic behaves.
  Mirror the examples already described in `misc/TIME_MANAGEMENT.md` but under automated tests.
- [ ] **Document usage** – extend `README.md`/`TIME_MANAGEMENT.md` with command-line examples (`--total-time 300 --increment 2`) and explain how the bot splits time between openings, complex middlegames, and simple endings. Mention that the upcoming opening book/tablebase integrations feed extra context to the time manager.

## 4. Run (and save) Regression Tests vs Baseline Stockfish

- [ ] **Smoke tests** – after every major change above, run the fast checks: `python tests/run_quick_instantiate.py`, `python tests/run_minimal_eval_safe.py`, and `python quick_test_vs_stockfish.py` to ensure model loading + a single game still work.
- [ ] **Structured benchmarks** – use `run_benchmark.py` and `tests/auto_elo_test.py` to gather ≥12 games per Stockfish strength (1350–2100). Save PGNs, JSON summaries, and selector/time stats to `logs/benchmarks/<timestamp>/`.
- [ ] **Compare against pure Stockfish** – run `play_vs_stockfish.py` in a “mirror” mode where Stockfish plays both sides (or use python-chess to pit Stockfish vs itself) to establish the baseline for draw rate/style. Store those baselines alongside hybrid results so regressions are easy to catch.
- [ ] **Track KPIs** – for each test suite, record win/draw/loss counts, estimated Elo (`tests/test_elo_rating.py`), transformer usage %, book/tablebase invocation counts, and average time per move. Publish the summary in `misc/STOCKFISH_TESTING.md` and mention any regressions or gains relative to earlier logs.
- [ ] **Gate commits on benchmarks** – optionally add a CI-style script (`check_elo_progress.sh`) that runs a short benchmark and refuses to proceed if the hybrid bot underperforms a configurable Stockfish baseline. This keeps future changes honest.

> Following the sequence above ensures: (1) the Chessformer component demonstrably improves over baseline Stockfish, (2) the engine leverages curated knowledge in openings and tablebases, (3) time controls behave predictably, and (4) every change is validated against Stockfish before shipping.
