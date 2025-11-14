#!/bin/bash
# Download the BEST available pretrained models

set -e

echo "=========================================="
echo "Downloading BEST Pretrained Models"
echo "=========================================="

# 1. Download latest Stockfish NNUE (Big network)
echo ""
echo "Step 1/2: Downloading latest Stockfish NNUE..."
NNUE_DIR="Stockfish/src"
NNUE_NEW="nn-49c1193b131c.nnue"  # Latest big network
NNUE_PATH="$NNUE_DIR/$NNUE_NEW"

if [ -f "$NNUE_PATH" ]; then
    echo "✓ Latest NNUE already exists"
else
    echo "  Downloading $NNUE_NEW (latest Stockfish network)..."
    wget -q --show-progress \
        "https://github.com/official-stockfish/networks/raw/master/$NNUE_NEW" \
        -O "$NNUE_PATH"
    echo "✓ Downloaded latest NNUE: $NNUE_NEW"
fi

# 2. Download CT-EFT-85 (largest transformer - 85M params)
echo ""
echo "Step 2/2: Setting up CT-EFT-85 (85M parameter transformer)..."
echo "  This is the LARGEST and BEST transformer model available"

TRANSFORMER_DIR="chess-transformers/checkpoints/CT-EFT-85"
mkdir -p "$TRANSFORMER_DIR"

echo "  Model will auto-download on first use"
echo "  Expected size: ~340 MB"

# 3. Update config.py
echo ""
echo "Updating config.py with best models..."

python3 << 'EOF'
import re

with open('config.py', 'r') as f:
    config = f.read()

# Update NNUE to latest
config = re.sub(
    r"STOCKFISH_NNUE_PATH = .*",
    "STOCKFISH_NNUE_PATH = os.path.join(STOCKFISH_DIR, 'src', 'nn-49c1193b131c.nnue')  # Latest big network",
    config
)

# Update transformer to CT-EFT-85
config = re.sub(
    r"TRANSFORMER_WEIGHTS_PATH = .*",
    "TRANSFORMER_WEIGHTS_PATH = os.path.join(CHESS_TRANSFORMERS_DIR, 'checkpoints', 'CT-EFT-85', 'CT-EFT-85.pt')  # 85M params (best)",
    config
)

with open('config.py', 'w') as f:
    f.write(config)

print("✓ Updated config.py with best models")
EOF

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ NNUE: nn-49c1193b131c.nnue (latest Stockfish)"
echo "✓ Transformer: CT-EFT-85 (85M params, best available)"
echo ""
echo "Next: Run selector analysis to see improvement"
echo "  python src/analyze_selector.py"
echo "=========================================="
