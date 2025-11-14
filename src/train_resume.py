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
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    trainer.projection.load_state_dict(checkpoint['projection_state_dict'])
    trainer.selector.load_state_dict(checkpoint['selector_state_dict'])
    
    # Restore history if available
    if 'history' in checkpoint:
        trainer.history = checkpoint['history']
        print(f"Restored training history with {len(trainer.history['train_loss'])} epochs")
    
    return checkpoint.get('epoch', 0), checkpoint.get('phase', 1)


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='best_phase2.pt',
                      help='Checkpoint filename to resume from')
    parser.add_argument('--phase1-epochs', type=int, default=100,
                      help='Total epochs for phase 1 (projection only)')
    parser.add_argument('--phase2-epochs', type=int, default=100,
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
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint)
    if os.path.exists(checkpoint_path):
        last_epoch, last_phase = load_checkpoint(trainer, checkpoint_path)
        print(f"Resuming from epoch {last_epoch}, phase {last_phase}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Starting fresh training...")
        last_epoch = 0
        last_phase = 0
    
    # Training schedule
    print("\nTraining Schedule:")
    print(f"  Phase 1: {args.phase1_epochs} epochs (projection only)")
    print(f"  Phase 2: {args.phase2_epochs} epochs (joint training)")
    print(f"  Total: {args.phase1_epochs + args.phase2_epochs} epochs")
    
    if last_epoch > 0:
        completed_phase1 = min(last_epoch, args.phase1_epochs)
        completed_phase2 = max(0, last_epoch - args.phase1_epochs)
        print(f"\nAlready completed:")
        print(f"  Phase 1: {completed_phase1} epochs")
        print(f"  Phase 2: {completed_phase2} epochs")
        print(f"  Remaining: {args.phase1_epochs + args.phase2_epochs - last_epoch} epochs")
    
    start_time = time.time()
    
    # Phase 1: Projection layer only
    if last_epoch < args.phase1_epochs:
        remaining_phase1 = args.phase1_epochs - last_epoch
        print(f"\nContinuing Phase 1 for {remaining_phase1} more epochs...")
        trainer.train_phase1(num_epochs=remaining_phase1)
    else:
        print("\nPhase 1 already complete, skipping...")
    
    # Phase 2: Joint training
    if last_epoch < args.phase1_epochs + args.phase2_epochs:
        remaining_phase2 = args.phase2_epochs - max(0, last_epoch - args.phase1_epochs)
        print(f"\nContinuing Phase 2 for {remaining_phase2} more epochs...")
        trainer.train_phase2(num_epochs=remaining_phase2)
    else:
        print("\nPhase 2 already complete, skipping...")
    
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
