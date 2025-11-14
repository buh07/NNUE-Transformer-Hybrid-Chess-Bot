"""
Training script for Hybrid NNUE-Transformer Chess Engine

Training Schedule (as per specification):
- Phase 1 (Day 1): Train projection layer only (selector frozen) - 25 epochs
- Phase 2 (Day 2): Joint training (projection + selector) - 25 epochs

Total: 50 epochs, ~2 days on RTX6000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, Tuple
import json

from models.nnue_evaluator import create_nnue_evaluator
from models.transformer_model import create_transformer_model
from models.projection_layer import create_projection_layer
from models.selector import create_selector
from dataset import create_dataloaders, create_dummy_dataset
from utils.chess_utils import legal_softmax
import config


class HybridTrainer:
    """
    Trainer for projection layer and selection function
    
    Training objectives:
    1. Projection layer: Minimize difference between NNUE->Transformer and ground truth
    2. Selector: Predict when transformer improves over NNUE
    """
    
    def __init__(self, nnue_model, transformer_model, projection_layer, selector,
                 train_loader, val_loader, device='cuda'):
        """
        Initialize trainer
        
        Args:
            nnue_model: Frozen NNUE evaluator
            transformer_model: Frozen transformer
            projection_layer: Trainable projection
            selector: Trainable selector
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for computation
        """
        self.device = device
        
        # Models
        self.nnue = nnue_model.to(device)
        self.transformer = transformer_model.to(device)
        self.projection = projection_layer.to(device)
        self.selector = selector.to(device)
        
        # Freeze pre-trained models
        self.nnue.eval()
        self.transformer.eval()
        for param in self.nnue.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.selector_criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_selector_loss': [],
            'val_policy_loss': [],
            'val_value_loss': [],
            'val_selector_loss': [],
            'selector_accuracy': []
        }
        
        print(f"Trainer initialized on device: {device}")
        print(f"Projection parameters: {sum(p.numel() for p in self.projection.parameters()):,}")
        print(f"Selector parameters: {sum(p.numel() for p in self.selector.parameters()):,}")
    
    def compute_losses(self, batch: Dict, phase: str = 'projection_only') -> Tuple[torch.Tensor, Dict]:
        """
        Compute losses for a batch
        
        Args:
            batch: Batch of data
            phase: 'projection_only' or 'joint'
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        boards = batch['boards']
        selection_features = batch['selection_features'].to(self.device)
        legal_masks = batch['legal_masks'].to(self.device)
        target_move_indices = batch['target_move_indices'].to(self.device)
        target_values = batch['target_values'].to(self.device)
        
        batch_size = len(boards)
        
        # Get NNUE features (no gradients)
        with torch.no_grad():
            nnue_features_list = []
            nnue_values_list = []
            for board in boards:
                feat, val = self.nnue.forward(board)
                nnue_features_list.append(feat)
                nnue_values_list.append(val)
            
            nnue_features = torch.stack(nnue_features_list).to(self.device)
            nnue_values = torch.tensor(nnue_values_list, dtype=torch.float32).to(self.device)
        
        # Project to transformer space (with gradients)
        projected_features = self.projection(nnue_features)
        
        # Get transformer outputs (transformer frozen but gradients flow through projection)
        policy_logits, transformer_values = self.transformer.forward(projected_features)
        
        # Loss 1: Policy prediction (cross-entropy on legal moves)
        # Mask illegal moves
        masked_logits = policy_logits.clone()
        masked_logits[~legal_masks] = -1e9
        policy_loss = self.policy_criterion(masked_logits, target_move_indices)
        
        # Loss 2: Value prediction
        # Blend NNUE and transformer values
        blended_values = (
            config.NNUE_VALUE_WEIGHT * nnue_values +
            config.TRANSFORMER_VALUE_WEIGHT * transformer_values
        )
        # Normalize target values to match scale
        target_values_scaled = target_values * 100.0  # Scale to centipawns
        value_loss = self.value_criterion(blended_values, target_values_scaled)
        
        # Loss 3: Selector training (only in joint phase)
        selector_loss = torch.tensor(0.0, device=self.device)
        selector_accuracy = 0.0
        
        if phase == 'joint':
            # Label: 1 if transformer improves, 0 otherwise
            with torch.no_grad():
                nnue_error = torch.abs(nnue_values - target_values_scaled)
                transformer_error = torch.abs(transformer_values - target_values_scaled)
                improvement = transformer_error < nnue_error
                selector_labels = improvement.float().unsqueeze(1)
            
            selector_preds = self.selector(selection_features)
            selector_loss = self.selector_criterion(selector_preds, selector_labels)
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = (selector_preds > config.SELECTOR_THRESHOLD).float()
                selector_accuracy = (predictions == selector_labels).float().mean().item()
        
        # Combined loss
        if phase == 'projection_only':
            total_loss = (
                config.POLICY_LOSS_WEIGHT * policy_loss +
                config.VALUE_LOSS_WEIGHT * value_loss
            )
        else:  # joint
            total_loss = (
                config.POLICY_LOSS_WEIGHT * policy_loss +
                config.VALUE_LOSS_WEIGHT * value_loss +
                config.SELECTOR_LOSS_WEIGHT * selector_loss
            )
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'selector_loss': selector_loss.item(),
            'selector_accuracy': selector_accuracy
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int, phase: str = 'projection_only') -> Dict:
        """
        Train for one epoch
        
        Args:
            epoch: Epoch number
            phase: 'projection_only' or 'joint'
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.projection.train()
        if phase == 'joint':
            self.selector.train()
        else:
            self.selector.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_selector_loss = 0.0
        total_selector_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} ({phase})")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Compute losses
            loss, loss_dict = self.compute_losses(batch, phase)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.projection.parameters(), 
                max_norm=config.GRADIENT_CLIP_NORM
            )
            if phase == 'joint':
                torch.nn.utils.clip_grad_norm_(
                    self.selector.parameters(),
                    max_norm=config.GRADIENT_CLIP_NORM
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            total_selector_loss += loss_dict['selector_loss']
            total_selector_accuracy += loss_dict['selector_accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{loss_dict['policy_loss']:.4f}",
                'value': f"{loss_dict['value_loss']:.4f}"
            })
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'selector_loss': total_selector_loss / num_batches,
            'selector_accuracy': total_selector_accuracy / num_batches
        }
        
        return metrics
    
    def validate(self, phase: str = 'projection_only') -> Dict:
        """
        Validate on validation set
        
        Args:
            phase: 'projection_only' or 'joint'
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.projection.eval()
        self.selector.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_selector_loss = 0.0
        total_selector_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Compute losses
                loss, loss_dict = self.compute_losses(batch, phase)
                
                # Accumulate losses
                total_loss += loss.item()
                total_policy_loss += loss_dict['policy_loss']
                total_value_loss += loss_dict['value_loss']
                total_selector_loss += loss_dict['selector_loss']
                total_selector_accuracy += loss_dict['selector_accuracy']
                num_batches += 1
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'selector_loss': total_selector_loss / num_batches,
            'selector_accuracy': total_selector_accuracy / num_batches
        }
        
        return metrics
    
    def train_phase1(self, num_epochs: int = 25, learning_rate: float = None):
        """
        Phase 1: Train projection layer only (selector frozen)
        
        Args:
            num_epochs: Number of epochs
            learning_rate: Learning rate
        """
        print("\n" + "="*70)
        print("PHASE 1: TRAINING PROJECTION LAYER (SELECTOR FROZEN)")
        print("="*70)
        
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        
        # Freeze selector
        for param in self.selector.parameters():
            param.requires_grad = False
        
        # Optimizer for projection only
        self.optimizer = torch.optim.Adam(
            self.projection.parameters(),
            lr=learning_rate
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch, phase='projection_only')
            
            # Validate
            val_metrics = self.validate(phase='projection_only')
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.history['val_policy_loss'].append(val_metrics['policy_loss'])
            self.history['train_value_loss'].append(train_metrics['value_loss'])
            self.history['val_value_loss'].append(val_metrics['value_loss'])
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Policy: {train_metrics['policy_loss']:.4f} | Val Policy: {val_metrics['policy_loss']:.4f}")
            print(f"  Train Value: {train_metrics['value_loss']:.4f} | Val Value: {val_metrics['value_loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_phase1.pt', epoch, phase=1)
                print(f"  ✓ Saved best Phase 1 model (val_loss={best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f'phase1_epoch{epoch}.pt', epoch, phase=1)
        
        print("\n✓ Phase 1 training complete!")
    
    def train_phase2(self, num_epochs: int = 25, learning_rate: float = None):
        """
        Phase 2: Joint training (projection + selector)
        
        Args:
            num_epochs: Number of epochs
            learning_rate: Learning rate
        """
        print("\n" + "="*70)
        print("PHASE 2: JOINT TRAINING (PROJECTION + SELECTOR)")
        print("="*70)
        
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE * 0.5  # Lower LR for fine-tuning
        
        # Unfreeze selector
        for param in self.selector.parameters():
            param.requires_grad = True
        
        # Optimizer for both projection and selector
        self.optimizer = torch.optim.Adam([
            {'params': self.projection.parameters(), 'lr': learning_rate},
            {'params': self.selector.parameters(), 'lr': learning_rate * config.SELECTOR_LR_MULTIPLIER}
        ])
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch, phase='joint')
            
            # Validate
            val_metrics = self.validate(phase='joint')
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.history['val_policy_loss'].append(val_metrics['policy_loss'])
            self.history['train_value_loss'].append(train_metrics['value_loss'])
            self.history['val_value_loss'].append(val_metrics['value_loss'])
            self.history['train_selector_loss'].append(train_metrics['selector_loss'])
            self.history['val_selector_loss'].append(val_metrics['selector_loss'])
            self.history['selector_accuracy'].append(val_metrics['selector_accuracy'])
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Policy: {train_metrics['policy_loss']:.4f} | Val Policy: {val_metrics['policy_loss']:.4f}")
            print(f"  Train Value: {train_metrics['value_loss']:.4f} | Val Value: {val_metrics['value_loss']:.4f}")
            print(f"  Train Selector: {train_metrics['selector_loss']:.4f} | Val Selector: {val_metrics['selector_loss']:.4f}")
            print(f"  Selector Accuracy: {val_metrics['selector_accuracy']:.2%}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_phase2.pt', epoch, phase=2)
                print(f"  ✓ Saved best Phase 2 model (val_loss={best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f'phase2_epoch{epoch}.pt', epoch, phase=2)
        
        print("\n✓ Phase 2 training complete!")
    
    def save_checkpoint(self, filename: str, epoch: int, phase: int):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, filename)
        
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'projection_state_dict': self.projection.state_dict(),
            'selector_state_dict': self.selector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON"""
        history_path = os.path.join(config.CHECKPOINT_DIR, filename)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training function"""
    print("="*70)
    print("HYBRID NNUE-TRANSFORMER TRAINING")
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
    
    # Check if PGN files exist
    train_pgn_files = config.PGN_FILES if config.PGN_FILES else []
    val_pgn_files = []  # Add validation PGN files if available
    
    if not train_pgn_files:
        print("Warning: No PGN files found. Using dummy dataset for testing.")
        print("For real training, add PGN files to config.PGN_FILES")
        
        # Create dummy datasets
        from dataset import collate_fn
        from torch.utils.data import DataLoader
        
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
            train_pgn_files,
            val_pgn_files,
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
    
    # Training schedule
    print("\nTraining Schedule:")
    print(f"  Phase 1: {config.PHASE1_EPOCHS} epochs (projection only)")
    print(f"  Phase 2: {config.PHASE2_EPOCHS} epochs (joint training)")
    print(f"  Total: {config.PHASE1_EPOCHS + config.PHASE2_EPOCHS} epochs")
    
    start_time = time.time()
    
    # Phase 1: Projection layer only
    trainer.train_phase1(num_epochs=config.PHASE1_EPOCHS)
    
    # Phase 2: Joint training
    trainer.train_phase2(num_epochs=config.PHASE2_EPOCHS)
    
    # Save final model
    trainer.save_checkpoint('final_model.pt', config.PHASE1_EPOCHS + config.PHASE2_EPOCHS, phase=2)
    trainer.save_history()
    
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


if __name__ == '__main__':
    main()
