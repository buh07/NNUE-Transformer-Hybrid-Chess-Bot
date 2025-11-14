"""
Analyze selector behavior: precision, recall, and class distribution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm
import config
from models.nnue_evaluator import create_nnue_evaluator
from models.transformer_model import create_transformer_model
from models.projection_layer import create_projection_layer
from models.selector import create_selector
from dataset import create_dataloaders


def analyze_selector(checkpoint_path='checkpoints/production_model.pt'):
    """Analyze selector performance and class distribution"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading models...")
    nnue = create_nnue_evaluator().to(device).eval()
    transformer = create_transformer_model().to(device).eval()
    projection = create_projection_layer().to(device)
    selector = create_selector().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    projection.load_state_dict(checkpoint['projection_state_dict'])
    selector.load_state_dict(checkpoint['selector_state_dict'])
    projection.eval()
    selector.eval()
    
    # Load validation data
    print("Loading validation data...")
    _, val_loader = create_dataloaders(
        config.PGN_FILES,
        config.PGN_FILES,
        batch_size=config.BATCH_SIZE
    )
    
    # Analyze
    all_labels = []
    all_preds = []
    all_pred_probs = []
    improvements = []
    
    print("\nAnalyzing selector behavior...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            boards = batch['boards']
            selection_features = batch['selection_features'].to(device)
            target_values = batch['target_values'].to(device)
            
            # Get NNUE and transformer predictions
            nnue_features_list = []
            nnue_values_list = []
            for board in boards:
                feat, val = nnue.forward(board)
                nnue_features_list.append(feat)
                nnue_values_list.append(val)
            
            nnue_features = torch.stack(nnue_features_list).to(device)
            nnue_values = torch.tensor(nnue_values_list, dtype=torch.float32).to(device)
            
            # Project and get transformer predictions
            projected_features = projection(nnue_features)
            _, transformer_values = transformer.forward(projected_features)
            
            # Calculate ground truth labels
            target_values_scaled = target_values * 100.0
            nnue_error = torch.abs(nnue_values - target_values_scaled)
            transformer_error = torch.abs(transformer_values - target_values_scaled)
            improvement = transformer_error < nnue_error
            improvement_amount = nnue_error - transformer_error
            
            # Get selector predictions
            selector_probs = selector(selection_features)
            selector_preds = (selector_probs > config.SELECTOR_THRESHOLD)
            
            all_labels.extend(improvement.cpu().numpy())
            all_preds.extend(selector_preds.squeeze().cpu().numpy())
            all_pred_probs.extend(selector_probs.squeeze().cpu().numpy())
            improvements.extend(improvement_amount.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_pred_probs = np.array(all_pred_probs)
    improvements = np.array(improvements)
    
    # Calculate metrics
    true_positives = np.sum((all_preds == 1) & (all_labels == 1))
    false_positives = np.sum((all_preds == 1) & (all_labels == 0))
    true_negatives = np.sum((all_preds == 0) & (all_labels == 0))
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1))
    
    accuracy = (true_positives + true_negatives) / len(all_labels)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    positive_rate = np.mean(all_labels)
    
    print("\n" + "="*70)
    print("SELECTOR ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nClass Distribution:")
    print(f"  Cases where transformer helps: {np.sum(all_labels)} ({positive_rate*100:.1f}%)")
    print(f"  Cases where NNUE is better:    {np.sum(~all_labels)} ({(1-positive_rate)*100:.1f}%)")
    
    print(f"\nCurrent Performance (Threshold = {config.SELECTOR_THRESHOLD}):")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}% (when selector says 'use transformer', how often is it right?)")
    print(f"  Recall:    {recall*100:.2f}% (what % of helpful cases does it catch?)")
    print(f"  F1 Score:  {f1:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Use Trans | Use NNUE")
    print(f"  Actual Use Trans:   {true_positives:5d}  |  {false_negatives:5d}")
    print(f"  Actual Use NNUE:    {false_positives:5d}  |  {true_negatives:5d}")
    
    print(f"\nMissed Opportunities:")
    print(f"  False Negatives: {false_negatives} (should use transformer but didn't)")
    print(f"  Average improvement missed: {np.mean(improvements[all_labels & ~all_preds]):.1f} centipawns")
    
    # Find optimal threshold for 90% recall
    print(f"\nFinding threshold for 90% recall...")
    sorted_probs = np.sort(all_pred_probs[all_labels])
    threshold_90_recall = sorted_probs[int(len(sorted_probs) * 0.1)] if len(sorted_probs) > 0 else 0.5
    
    new_preds = all_pred_probs > threshold_90_recall
    new_tp = np.sum((new_preds == 1) & (all_labels == 1))
    new_fp = np.sum((new_preds == 1) & (all_labels == 0))
    new_recall = new_tp / (new_tp + false_negatives) if (new_tp + false_negatives) > 0 else 0
    new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    
    print(f"\nRecommended Threshold for 90% Recall:")
    print(f"  New Threshold: {threshold_90_recall:.3f} (current: {config.SELECTOR_THRESHOLD})")
    print(f"  Expected Recall: {new_recall*100:.1f}%")
    print(f"  Expected Precision: {new_precision*100:.1f}%")
    print(f"  Trade-off: Catches more cases but activates more often")
    
    # ROC-like analysis
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nThreshold Analysis:")
    print(f"  Threshold | Recall | Precision | Accuracy")
    print(f"  ----------|--------|-----------|----------")
    for thresh in thresholds:
        preds_t = all_pred_probs > thresh
        tp_t = np.sum((preds_t == 1) & (all_labels == 1))
        fp_t = np.sum((preds_t == 1) & (all_labels == 0))
        fn_t = np.sum((preds_t == 0) & (all_labels == 1))
        tn_t = np.sum((preds_t == 0) & (all_labels == 0))
        
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        accuracy_t = (tp_t + tn_t) / len(all_labels)
        
        marker = " ← 90% recall" if 0.88 <= recall_t <= 0.92 else ""
        print(f"    {thresh:.1f}     | {recall_t*100:5.1f}% | {precision_t*100:6.1f}%  | {accuracy_t*100:5.1f}%{marker}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR 90% RECALL:")
    print("="*70)
    print(f"1. Lower SELECTOR_THRESHOLD to ~{threshold_90_recall:.2f}")
    print(f"2. Use recall-focused loss during training")
    print(f"3. Current recall is only {recall*100:.1f}% - need better features!")
    print(f"4. Class balance is {positive_rate*100:.1f}% positive - consider weighted loss")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/production_model.pt',
                       help='Path to checkpoint')
    args = parser.parse_args()
    
    analyze_selector(args.checkpoint)
