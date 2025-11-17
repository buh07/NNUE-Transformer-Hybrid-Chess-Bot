#!/usr/bin/env python3
"""
Download high-quality chess games from Lichess Elite Database.
Downloads games from 2500+ Elo players vs 2300+ opponents for training the hybrid model.
Source: https://database.nikonoel.fr/
"""

import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            pbar.update(len(chunk))

def decompress_zip(input_path, output_dir):
    """Decompress a .zip file."""
    print(f"Decompressing {os.path.basename(input_path)}...")
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Decompressed to {output_dir}")

def main():
    # Create data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Lichess Elite database - games from 2500+ vs 2300+ players (no bullet)
    # Using recent months for high-quality modern chess
    # Source: https://database.nikonoel.fr/
    datasets = [
        {
            'url': 'https://database.nikonoel.fr/lichess_elite_2024-10.zip',
            'filename': 'lichess_elite_2024-10.zip',
            'pgn_name': 'lichess_elite_2024-10.pgn'
        },
        {
            'url': 'https://database.nikonoel.fr/lichess_elite_2024-09.zip',
            'filename': 'lichess_elite_2024-09.zip',
            'pgn_name': 'lichess_elite_2024-09.pgn'
        },
        {
            'url': 'https://database.nikonoel.fr/lichess_elite_2024-08.zip',
            'filename': 'lichess_elite_2024-08.zip',
            'pgn_name': 'lichess_elite_2024-08.pgn'
        },
        {
            'url': 'https://database.nikonoel.fr/lichess_elite_2024-07.zip',
            'filename': 'lichess_elite_2024-07.zip',
            'pgn_name': 'lichess_elite_2024-07.pgn'
        }
    ]
    
    for dataset in datasets:
        zip_path = os.path.join(data_dir, dataset['filename'])
        pgn_path = os.path.join(data_dir, dataset['pgn_name'])
        
        # Skip if already downloaded and decompressed
        if os.path.exists(pgn_path):
            print(f"✓ {dataset['pgn_name']} already exists, skipping...")
            continue
        
        # Download compressed file
        if not os.path.exists(zip_path):
            print(f"\nDownloading {dataset['filename']}...")
            try:
                download_file(dataset['url'], zip_path)
                print(f"✓ Downloaded {dataset['filename']}")
            except Exception as e:
                print(f"✗ Error downloading {dataset['filename']}: {e}")
                continue
        else:
            print(f"✓ {dataset['filename']} already downloaded")
        
        # Decompress
        try:
            decompress_zip(zip_path, data_dir)
            print(f"✓ Extracted {dataset['pgn_name']}")
            
            # Remove compressed file to save space
            os.remove(zip_path)
            print(f"✓ Removed {dataset['filename']} (compressed file)")
        except Exception as e:
            print(f"✗ Error decompressing {dataset['filename']}: {e}")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nPGN files saved to: {os.path.abspath(data_dir)}/")
    print("\nNext steps:")
    print("1. Update config.py to include these PGN files:")
    print("   PGN_FILES = [")
    for dataset in datasets:
        print(f"       'data/{dataset['pgn_name']}',")
    print("   ]")
    print("2. Run training: chess_env/bin/python src/train.py")

if __name__ == '__main__':
    main()
