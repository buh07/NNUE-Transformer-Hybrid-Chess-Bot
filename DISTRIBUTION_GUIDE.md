# Distribution Guide - How to Share Your Chess Bot

## Option 1: Complete Package (Recommended for Easy Setup)

**Best for:** Someone who wants to use your bot immediately

### What to Send:
```
hybrid-chess-bot-package/
├── HybridChessBot.py           # Main bot file
├── requirements.txt            # Dependencies
├── BOT_USAGE.md               # Documentation
├── quick_start.py             # Demo script
├── checkpoints/
│   └── best_phase2.pt         # Your trained weights (~2.2 MB)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── nnue.py
│   │   ├── transformer.py
│   │   ├── hybrid_evaluator.py
│   │   └── projection.py
│   ├── search.py
│   └── config.py
└── chess-transformers/         # Pre-trained transformer model
    └── (model files)
```

### How to Package:
```bash
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"

# Create distribution directory
mkdir -p hybrid-chess-bot-package

# Copy essential files
cp HybridChessBot.py hybrid-chess-bot-package/
cp requirements.txt hybrid-chess-bot-package/
cp BOT_USAGE.md hybrid-chess-bot-package/
cp quick_start.py hybrid-chess-bot-package/
cp config.py hybrid-chess-bot-package/

# Copy checkpoint
mkdir -p hybrid-chess-bot-package/checkpoints
cp checkpoints/best_phase2.pt hybrid-chess-bot-package/checkpoints/

# Copy source code
cp -r src/ hybrid-chess-bot-package/

# Copy transformer model
cp -r chess-transformers/ hybrid-chess-bot-package/

# Create archive
tar -czf hybrid-chess-bot.tar.gz hybrid-chess-bot-package/
# or
zip -r hybrid-chess-bot.zip hybrid-chess-bot-package/
```

### Installation Instructions for Recipient:
```bash
# Extract
tar -xzf hybrid-chess-bot.tar.gz
cd hybrid-chess-bot-package

# Create environment
python3 -m venv bot_env
source bot_env/bin/activate  # On Windows: bot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test the bot
python quick_start.py
```

**Package size:** ~2-3 GB (includes transformer model)

---

## Option 2: Minimal Package (Source + Checkpoint Only)

**Best for:** Developers who can download models themselves

### What to Send:
```
hybrid-chess-bot-minimal/
├── HybridChessBot.py
├── requirements.txt
├── BOT_USAGE.md
├── setup_instructions.md      # Include model download instructions
├── checkpoints/
│   └── best_phase2.pt         # Your trained weights
└── src/
    └── (all source files)
```

### How to Package:
```bash
mkdir -p hybrid-chess-bot-minimal
cp HybridChessBot.py hybrid-chess-bot-minimal/
cp requirements.txt hybrid-chess-bot-minimal/
cp BOT_USAGE.md hybrid-chess-bot-minimal/
cp config.py hybrid-chess-bot-minimal/
mkdir -p hybrid-chess-bot-minimal/checkpoints
cp checkpoints/best_phase2.pt hybrid-chess-bot-minimal/checkpoints/
cp -r src/ hybrid-chess-bot-minimal/

# Create setup instructions
cat > hybrid-chess-bot-minimal/setup_instructions.md << 'EOF'
# Setup Instructions

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Download Pre-trained Transformer Model
The bot uses the chess-transformers model. Download it:
```bash
git clone https://huggingface.co/adamkarvonen/chess_llms_2_1024d_12layers
mv chess_llms_2_1024d_12layers chess-transformers
```

## 3. Run the Bot
```bash
python quick_start.py
```
EOF

tar -czf hybrid-chess-bot-minimal.tar.gz hybrid-chess-bot-minimal/
```

**Package size:** ~2.3 MB (recipient downloads models separately)

---

## Option 3: GitHub Repository (Best for Collaboration)

**Best for:** Open source sharing, collaboration, version control

### Steps:

1. **Create a new repository** (or use existing one)
2. **Add a .gitignore** to exclude large files:
```bash
cat > .gitignore << 'EOF'
# Virtual environments
chess_env/
bot_env/
*_env/
venv/

# Data files
data/
logs/
*.pgn

# Checkpoints (too large for git)
checkpoints/*.pt
!checkpoints/.gitkeep

# Large models (use Git LFS or external hosting)
chess-transformers/
Stockfish/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# IDE
.vscode/
.idea/
*.swp
EOF
```

3. **Use Git LFS for checkpoint** (optional):
```bash
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes
```

4. **Provide download script** for large files:
```bash
cat > download_models.sh << 'EOF'
#!/bin/bash
# Download pre-trained transformer
git clone https://huggingface.co/adamkarvonen/chess_llms_2_1024d_12layers chess-transformers

# Download checkpoint (host this somewhere like Google Drive, Dropbox, or Hugging Face)
# wget -O checkpoints/best_phase2.pt "YOUR_HOSTED_URL"
echo "Please download best_phase2.pt from [YOUR_LINK] and place in checkpoints/"
EOF
chmod +x download_models.sh
```

5. **Create comprehensive README.md**:
```markdown
# Hybrid NNUE-Transformer Chess Bot

A chess engine that adaptively uses NNUE for tactical positions and Transformers for strategic positions.

## Features
- 97.56% accurate position type classification
- 3.3x speedup vs pure Transformer
- Alpha-beta search with transposition table
- Compatible with python-chess

## Quick Start
See [BOT_USAGE.md](BOT_USAGE.md) for details.

## Installation
See [INSTALLATION.md](INSTALLATION.md) for setup instructions.
```

6. **Host checkpoint externally:**
   - **Hugging Face:** Upload to huggingface.co/models (recommended)
   - **Google Drive:** Share with link
   - **GitHub Releases:** For files under 2GB
   - **Dropbox/OneDrive:** Public link

---

## Option 4: Python Package (PyPI) - Advanced

**Best for:** Wide distribution, easy installation

Create a proper Python package:
```
hybrid-chess-bot/
├── setup.py
├── README.md
├── LICENSE
├── hybrid_chess_bot/
│   ├── __init__.py
│   ├── bot.py
│   ├── models/
│   └── search.py
└── tests/
```

Users would install via:
```bash
pip install hybrid-chess-bot
```

---

## Recommended Approach

### For a Friend/Colleague:
**Use Option 1** - Send complete package via:
- Google Drive / Dropbox link
- USB drive
- Cloud storage (Box, OneDrive)

### For Public Sharing:
**Use Option 3** - GitHub repository with:
- Source code in Git
- Checkpoint on Hugging Face
- Clear installation instructions

---

## File Size Breakdown

| Component | Size | Required? |
|-----------|------|-----------|
| Your trained checkpoint | ~2.2 MB | ✅ Yes |
| Source code | ~500 KB | ✅ Yes |
| chess-transformers model | ~2 GB | ✅ Yes |
| NNUE weights (in model) | ~40 MB | ✅ Yes |
| Python dependencies | (pip install) | ✅ Yes |
| **Total** | **~2.3 GB** | |

---

## Quick Commands

### Create Complete Package:
```bash
cd "/scratch2/f004ndc/NNUE Transformer Hybrid Chess Bot"
./create_distribution_package.sh  # (script below)
```

### Create Distribution Script:
```bash
cat > create_distribution_package.sh << 'SCRIPT'
#!/bin/bash
set -e

DIST_DIR="hybrid-chess-bot-distribution"
echo "Creating distribution package..."

# Clean previous build
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy files
echo "Copying files..."
cp HybridChessBot.py "$DIST_DIR/"
cp requirements.txt "$DIST_DIR/"
cp BOT_USAGE.md "$DIST_DIR/"
cp quick_start.py "$DIST_DIR/"
cp config.py "$DIST_DIR/"
cp -r src/ "$DIST_DIR/"

# Copy checkpoint
mkdir -p "$DIST_DIR/checkpoints"
cp checkpoints/best_phase2.pt "$DIST_DIR/checkpoints/"

# Copy transformer model
echo "Copying transformer model (this may take a while)..."
cp -r chess-transformers/ "$DIST_DIR/"

# Create README
cat > "$DIST_DIR/README.md" << 'EOF'
# Hybrid Chess Bot - Distribution Package

## Quick Start

1. Create virtual environment:
   ```bash
   python3 -m venv bot_env
   source bot_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run demo:
   ```bash
   python quick_start.py
   ```

4. Use in your code:
   ```python
   from HybridChessBot import HybridChessBot
   bot = HybridChessBot()
   move = bot.choose_move(board)
   ```

See BOT_USAGE.md for complete documentation.
EOF

# Create archive
echo "Creating archive..."
tar -czf hybrid-chess-bot-distribution.tar.gz "$DIST_DIR/"

echo "✅ Distribution package created: hybrid-chess-bot-distribution.tar.gz"
echo "Size: $(du -h hybrid-chess-bot-distribution.tar.gz | cut -f1)"
SCRIPT

chmod +x create_distribution_package.sh
```

---

## Testing Before Sending

Make sure the package works on a fresh system:
```bash
# Extract to temp location
mkdir /tmp/test_bot
cd /tmp/test_bot
tar -xzf /path/to/hybrid-chess-bot-distribution.tar.gz
cd hybrid-chess-bot-distribution

# Fresh environment
python3 -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt

# Test
python quick_start.py
```

---

## What the Recipient Needs

### System Requirements:
- Python 3.8+
- 4GB+ RAM
- (Optional) NVIDIA GPU with CUDA for faster inference

### They DON'T need:
- Your training data
- Training scripts
- Stockfish
- Multiple checkpoints (just best_phase2.pt)
- Virtual environment folder

### They DO need:
- HybridChessBot.py
- checkpoints/best_phase2.pt
- src/ folder (models, search, config)
- chess-transformers/ model
- requirements.txt
- Documentation
