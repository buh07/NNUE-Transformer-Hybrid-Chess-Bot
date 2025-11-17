"""
Load pretrained transformer weights into our model architecture

This extracts the useful parts from the CT-E-20 checkpoint
and adapts them to our hybrid architecture.
"""

import torch
import sys
import os

def load_pretrained_transformer_weights():
    """
    Load the CT-E-20 pretrained weights and extract what we need
    """
    checkpoint_path = "chess-transformers/checkpoints/CT-E-20/CT-E-20.pt"
    
    print(f"Loading pretrained transformer from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"✓ Loaded {len(state_dict)} layers")
        
        # Print what's in the checkpoint
        print("\nCheckpoint contains:")
        for key in list(state_dict.keys())[:10]:
            print(f"  - {key}: {state_dict[key].shape}")
        print(f"  ... and {len(state_dict)-10} more layers")
        
        return state_dict
    else:
        print("✗ Unexpected checkpoint format")
        return None

if __name__ == '__main__':
    weights = load_pretrained_transformer_weights()
    
    if weights:
        print("\n" + "="*60)
        print("SUCCESS: Pretrained weights are loadable!")
        print("="*60)
        print("\nNow we need to adapt your model architecture")
        print("to match CT-E-20's architecture, OR")
        print("use the models as-is for evaluation only (frozen).")
        print()
        print("Since your training approach is:")
        print("1. Freeze NNUE (✓ can use Stockfish NNUE)")
        print("2. Freeze Transformer (✓ can use CT-E-20)")
        print("3. Train projection + selector")
        print()
        print("You just need to:")
        print("- Load NNUE for position evaluation")
        print("- Load CT-E-20 for move prediction")
        print("- Train your projection/selector with these frozen")
