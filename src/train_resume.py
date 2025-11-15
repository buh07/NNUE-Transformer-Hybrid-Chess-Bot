"""
Resume training from checkpoint for complete training run

This script loads the existing checkpoint and continues training
with more epochs for a full training run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from train import HybridTrainer
from models.nnue_evaluator import create_nnue_evaluator
from models.transformer_model import create_transformer_model
from models.projection_layer import create_projection_layer
from models.selector import create_selector
from dataset import create_dataloaders, create_dummy_dataset
from torch.utils.data import DataLoader
import config
import time
import json


def load_checkpoint(trainer, checkpoint_path):
    """Load checkpoint and restore training state"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    
    trainer.projection.load_state_dict(checkpoint['projection_state_dict'])
    trainer.selector.load_state_dict(checkpoint['selector_state_dict'])
    
    # Restore history if available
    if 'history' in checkpoint:
        trainer.history = checkpoint['history']
        total_history_entries = len(trainer.history['train_loss'])
        
        # Selector accuracy only exists for Phase 2 epochs
        # Use this to determine how many Phase 1 vs Phase 2 epochs were completed
        phase2_epochs_completed = len([x for x in trainer.history['selector_accuracy'] if x > 0])
        phase1_epochs_completed = total_history_entries - phase2_epochs_completed
        
        print(f"Restored training history: {total_history_entries} total epochs")
        print(f"  Phase 1 completed: {phase1_epochs_completed} epochs")
        print(f"  Phase 2 completed: {phase2_epochs_completed} epochs")
    else:
        phase1_epochs_completed = 0
        phase2_epochs_completed = 0
    
    # Phase tells us which training phase we're in (1 or 2)
    phase = checkpoint.get('phase', 1)
    epoch_in_phase = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint saved at: Phase {phase}, Epoch {epoch_in_phase} (within phase)")
    
    return phase1_epochs_completed, phase2_epochs_completed, phase


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='best_phase2.pt',
                      help='Checkpoint filename to resume from')
    parser.add_argument('--start-phase', type=int, choices=[1, 2], default=None,
                      help='Override: which phase to start from (1 or 2)')
    parser.add_argument('--start-phase-epoch', type=int, default=None,
                      help='Override: which epoch within the phase to start from')
    parser.add_argument('--phase1-epochs', type=int, default=25,
                      help='Total epochs for phase 1 (projection only)')
    parser.add_argument('--phase2-epochs', type=int, default=25,
                      help='Total epochs for phase 2 (joint training)')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU device to use (default: CUDA_VISIBLE_DEVICES or 0)')
    args = parser.parse_args()
    
    print("="*70)
    print("HYBRID NNUE-TRANSFORMER TRAINING (RESUME)")
    print("="*70)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create models
    print("\nLoading models...")
    nnue = create_nnue_evaluator()
    transformer = create_transformer_model()
    projection = create_projection_layer()
    selector = create_selector()
    
    # Create dataloaders
    print("\nPreparing data...")
    all_pgn_files = config.PGN_FILES if config.PGN_FILES else []
    
    if not all_pgn_files:
        print("Warning: No PGN files found. Using dummy dataset for testing.")
        from dataset import collate_fn
        
        train_dataset = create_dummy_dataset(num_positions=1000)
        val_dataset = create_dummy_dataset(num_positions=200)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
    else:
        train_loader, val_loader = create_dataloaders(
            all_pgn_files,
            all_pgn_files,
            batch_size=config.BATCH_SIZE
        )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    trainer = HybridTrainer(
        nnue, transformer, projection, selector,
        train_loader, val_loader,
        device=device
    )
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    # If it's not an absolute path and doesn't already start with a directory, add CHECKPOINT_DIR
    if not os.path.isabs(checkpoint_path) and not os.path.dirname(checkpoint_path):
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        completed_phase1, completed_phase2, last_phase = load_checkpoint(trainer, checkpoint_path)
        print(f"Checkpoint was at: Phase {last_phase}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Starting fresh training...")
        completed_phase1 = 0
        completed_phase2 = 0
        last_phase = 0
    
    # Apply manual overrides if specified
    if args.start_phase is not None:
        print(f"\n⚠️  Manual override: Starting from Phase {args.start_phase}")
        if args.start_phase == 1:
            completed_phase1 = args.start_phase_epoch if args.start_phase_epoch else 0
            completed_phase2 = 0
        else:  # Phase 2
            completed_phase1 = args.phase1_epochs  # Assume Phase 1 is complete
            completed_phase2 = args.start_phase_epoch if args.start_phase_epoch else 0
    
    # Training schedule
    print("\nTraining Schedule:")
    print(f"  Phase 1 target: {args.phase1_epochs} epochs (projection only)")
    print(f"  Phase 2 target: {args.phase2_epochs} epochs (joint training)")
    print(f"  Total target: {args.phase1_epochs + args.phase2_epochs} epochs")
    
    if completed_phase1 > 0 or completed_phase2 > 0:
        print(f"\nStarting from:")
        print(f"  Phase 1: {completed_phase1} epochs completed")
        print(f"  Phase 2: {completed_phase2} epochs completed")
        remaining_phase1 = max(0, args.phase1_epochs - completed_phase1)
        remaining_phase2 = max(0, args.phase2_epochs - completed_phase2)
        total_remaining = remaining_phase1 + remaining_phase2
        print(f"  Remaining: {total_remaining} epochs ({remaining_phase1} Phase 1, {remaining_phase2} Phase 2)")
    
    start_time = time.time()
    
    # Phase 1: Projection layer only
    if completed_phase1 < args.phase1_epochs:
        remaining_phase1 = args.phase1_epochs - completed_phase1
        print(f"\n{'='*70}")
        print(f"CONTINUING PHASE 1 for {remaining_phase1} more epochs...")
        print(f"{'='*70}")
        trainer.train_phase1(num_epochs=remaining_phase1)
    else:
        print(f"\n{'='*70}")
        print("PHASE 1 ALREADY COMPLETE - Skipping to Phase 2")
        print(f"{'='*70}")
    
    # Phase 2: Joint training
    if completed_phase2 < args.phase2_epochs:
        remaining_phase2 = args.phase2_epochs - completed_phase2
        print(f"\n{'='*70}")
        print(f"CONTINUING PHASE 2 for {remaining_phase2} more epochs...")
        print(f"  Already completed: {completed_phase2} Phase 2 epochs")
        print(f"  Starting from Phase 2 epoch {completed_phase2 + 1}")
        print(f"{'='*70}")
        trainer.train_phase2(num_epochs=remaining_phase2)
    else:
        print(f"\n{'='*70}")
        print("PHASE 2 ALREADY COMPLETE")
        print(f"{'='*70}")
    
    # Save final model
    final_epoch = args.phase1_epochs + args.phase2_epochs
    trainer.save_checkpoint(f'final_model_epoch{final_epoch}.pt', final_epoch, phase=2)
    trainer.save_history(filename=f'training_history_epoch{final_epoch}.json')
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Final train loss: {trainer.history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {trainer.history['val_loss'][-1]:.4f}")
    if trainer.history['selector_accuracy']:
        print(f"  Selector accuracy: {trainer.history['selector_accuracy'][-1]:.2%}")
    
    # Save summary
    summary = {
        'total_epochs': final_epoch,
        'total_time_hours': elapsed_time/3600,
        'final_train_loss': trainer.history['train_loss'][-1],
        'final_val_loss': trainer.history['val_loss'][-1],
        'final_selector_accuracy': trainer.history['selector_accuracy'][-1] if trainer.history['selector_accuracy'] else 0.0,
        'best_train_loss': min(trainer.history['train_loss']),
        'best_val_loss': min(trainer.history['val_loss']),
    }
    
    summary_path = os.path.join(config.CHECKPOINT_DIR, f'training_summary_epoch{final_epoch}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")


if __name__ == '__main__':
    main()
