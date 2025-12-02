import time
import torch
import json
import argparse
import os
from HybridChessBot import HybridChessBot
import chess
from src.utils.chess_utils import extract_selection_features
import config

parser = argparse.ArgumentParser(description="Diagnose transformer/selector behaviour.")
parser.add_argument(
    "--eval-mode",
    choices=["auto", "nnue", "transformer"],
    default=os.environ.get("HYBRID_EVAL_MODE", "auto"),
    help="Force HybridEvaluator mode for ablation experiments."
)
args = parser.parse_args()
eval_mode = args.eval_mode

print("Instantiating bot (will load selector+projection checkpoint)...")
bot = HybridChessBot(
    checkpoint='checkpoints/best_phase2.pt',
    depth=5,
    time_limit=1.0,
    verbose=False,
    evaluation_mode=eval_mode
)
print('device=', bot.device)
print('CONFIG: TRANSFORMER_VALUE_WEIGHT=', config.TRANSFORMER_VALUE_WEIGHT,
      'NNUE_VALUE_WEIGHT=', config.NNUE_VALUE_WEIGHT,
      'SELECTOR_THRESHOLD=', config.SELECTOR_THRESHOLD,
      'EVAL_MODE=', bot.evaluation_mode)

sel = bot.selector
threshold = config.SELECTOR_THRESHOLD
# sample positions
samples = {
    'start': chess.Board(),
    'after_1_e4': (lambda: (lambda b: (b.push_san('e4'), b)[1])(chess.Board()))(),
    'tactical_line': (lambda: (lambda b: (b.push_san('e4'), b.push_san('e5'), b.push_san('Nf3'), b.push_san('Nc6'), b)[-1])(chess.Board()))(),
}
print('\nSelector probabilities for sample positions:')
for name, board in samples.items():
    feats = extract_selection_features(board, 5)
    # Ensure features are on the same device as the selector parameters
    try:
        sel_device = next(sel.parameters()).device
    except StopIteration:
        sel_device = torch.device('cpu')
    feats = feats.to(sel_device)
    with torch.no_grad():
        p = sel.forward(feats.unsqueeze(0).to(sel_device)).squeeze(0)
        try:
            prob = float(p.item())
        except Exception:
            prob = float(p.squeeze().item())
    print(f"  {name}: prob={prob:.4f} -> use_transformer={(prob>threshold)}")

# Print simple selector param magnitude summary
param_means = [p.abs().mean().item() for _,p in sel.named_parameters() if p.numel()>0]
print('\nSelector params mean abs (per-tensor avg)=', sum(param_means)/len(param_means) if param_means else 0.0)

# Measure direct transformer forward times (first call + steady state) by invoking projection+transformer directly
board = chess.Board()
print('\nMeasuring transformer forward times (project->transformer) on starting position...')
# get nnue features from bot.nnue; nnue.forward returns feat, val
with torch.inference_mode():
    nnue_feat, nnue_val = bot.nnue.forward(board)
    nnue_feat = nnue_feat.to(bot.device)
# project
proj = bot.projection
tr = bot.transformer

# first call (may include compile/warmup)
try:
    import time
    t0 = time.time()
    with torch.inference_mode():
        strategic = proj(nnue_feat)
        policy_logits, transformer_value = tr.forward(strategic)
    t1 = time.time()
    first_dt = t1 - t0
    print(f'  first forward dt = {first_dt:.3f}s')
except Exception as e:
    print('  Transformer forward failed:', e)
    first_dt = None

# steady-state: run 5 more forwards and average
import statistics
dts = []
for i in range(5):
    with torch.inference_mode():
        t0 = time.time()
        strategic = proj(nnue_feat)
        policy_logits, transformer_value = tr.forward(strategic)
        t1 = time.time()
    dts.append(t1-t0)
print('  steady forward times (s):', [f"{d:.4f}" for d in dts])
print('  steady avg (ms)=', statistics.mean(dts)*1000)

# Print current evaluator stats and search stats (if available)
try:
    stats = bot.get_statistics()
    print('\nBot.get_statistics() summary:')
    print(json.dumps({'search': stats['search'], 'evaluation': stats['evaluation']}, indent=2, default=str))
except Exception as e:
    print('Failed to get bot statistics:', e)

print('\nDiagnostic script complete.')
