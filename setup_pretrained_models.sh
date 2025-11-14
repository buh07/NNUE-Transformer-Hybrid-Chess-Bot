#!/bin/bash
# Download and setup pretrained models for Chess Transformer

set -e

echo "========================================"
echo "Setting up Pretrained Chess Models"
echo "========================================"

# 1. Download Stockfish NNUE weights
echo ""
echo "Step 1/3: Downloading Stockfish NNUE weights..."
NNUE_DIR="Stockfish/src"
NNUE_FILE="nn-0000000000a0.nnue"
NNUE_PATH="$NNUE_DIR/$NNUE_FILE"

if [ -f "$NNUE_PATH" ]; then
    echo "✓ NNUE weights already exist at $NNUE_PATH"
else
    mkdir -p "$NNUE_DIR"
    echo "  Downloading from Stockfish tests server..."
    wget -q --show-progress \
        https://tests.stockfishchess.org/api/nn/$NNUE_FILE \
        -O "$NNUE_PATH" || {
        echo "❌ Failed to download NNUE. Trying alternate source..."
        wget -q --show-progress \
            https://github.com/official-stockfish/networks/raw/master/$NNUE_FILE \
            -O "$NNUE_PATH"
    }
    echo "✓ Downloaded NNUE weights to $NNUE_PATH"
fi

# 2. Download Chess Transformer pretrained model (CT-E-20)
echo ""
echo "Step 2/3: Downloading Chess Transformer (CT-E-20) weights..."
TRANSFORMER_DIR="chess-transformers/checkpoints/CT-E-20"
TRANSFORMER_FILE="CT-E-20.pt"
TRANSFORMER_PATH="$TRANSFORMER_DIR/$TRANSFORMER_FILE"

if [ -f "$TRANSFORMER_PATH" ]; then
    echo "✓ Transformer weights already exist at $TRANSFORMER_PATH"
else
    mkdir -p "$TRANSFORMER_DIR"
    echo "  Downloading CT-E-20 model (20M parameters)..."
    # This will be downloaded automatically by chess-transformers when first used
    # We'll trigger it by running a quick test
    cd chess-transformers
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from chess_transformers.configs.models import CT_E_20
from chess_transformers.play.utils import load_model

print("  Loading model (will auto-download if needed)...")
try:
    model = load_model(CT_E_20)
    print("✓ Transformer model ready!")
except Exception as e:
    print(f"⚠ Warning: {e}")
    print("  Model will be downloaded on first use")
EOF
    cd ..
fi

# 3. Update config.py with correct paths
echo ""
echo "Step 3/3: Updating config.py with model paths..."

# Backup config
cp config.py config.py.backup

# Update paths
python3 << 'EOF'
import re

with open('config.py', 'r') as f:
    config = f.read()

# Update NNUE path
config = re.sub(
    r"STOCKFISH_NNUE_PATH = .*",
    "STOCKFISH_NNUE_PATH = os.path.join(STOCKFISH_DIR, 'src', 'nn-0000000000a0.nnue')",
    config
)

# Update transformer path
config = re.sub(
    r"TRANSFORMER_WEIGHTS_PATH = .*",
    "TRANSFORMER_WEIGHTS_PATH = os.path.join(CHESS_TRANSFORMERS_DIR, 'checkpoints', 'CT-E-20', 'CT-E-20.pt')",
    config
)

with open('config.py', 'w') as f:
    f.write(config)

print("✓ Updated config.py")
EOF

# 4. Verify setup
echo ""
echo "========================================"
echo "Verification"
echo "========================================"

if [ -f "$NNUE_PATH" ]; then
    NNUE_SIZE=$(du -h "$NNUE_PATH" | cut -f1)
    echo "✓ NNUE weights: $NNUE_SIZE at $NNUE_PATH"
else
    echo "❌ NNUE weights not found"
fi

if [ -f "$TRANSFORMER_PATH" ]; then
    TRANS_SIZE=$(du -h "$TRANSFORMER_PATH" | cut -f1)
    echo "✓ Transformer weights: $TRANS_SIZE at $TRANSFORMER_PATH"
else
    echo "⚠ Transformer weights will be downloaded on first use"
fi

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo "1. Run analysis to verify models work:"
echo "   python src/analyze_selector.py"
echo ""
echo "2. This should now show transformer helping in 20-40% of cases"
echo ""
echo "3. If it works, proceed with selector improvement training"
echo "========================================"
