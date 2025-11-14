#!/usr/bin/env python3
"""
Quick test of the training pipeline with small dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch

# Set smaller config for testing
import config
config.BATCH_SIZE = 8
config.PHASE1_EPOCHS = 2
config.PHASE2_EPOCHS = 2

from src.train import main

print("Testing training pipeline with small dataset...")
print("This will run 2 epochs of Phase 1 and 2 epochs of Phase 2")
print()

if __name__ == '__main__':
    main()
