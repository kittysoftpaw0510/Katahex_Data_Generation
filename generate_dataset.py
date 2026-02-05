#!/usr/bin/env python3
"""
Conversation Dataset Generator
Processes all SGFS files in a directory and generates conversation datasets.

Input: Directory containing .sgfs files
Output: Directory with subdirectories containing .jsonl files (one per game)
"""

import sys
import argparse
from pathlib import Path
from typing import List

from orchestrator import SGFSProcessor


def find_sgfs_files(input_dir: str) -> List[Path]:
    """Find all .sgfs files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sgfs_files = list(input_path.glob("*.sgfs"))
    return sorted(sgfs_files)


def main():
    parser = argparse.ArgumentParser(
        description='Generate conversation datasets from SGFS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all SGFS files in a directory
  python generate_dataset.py --input sgfs_dir --output dataset_dir

  # With custom KataHex settings and multi-threading
  python generate_dataset.py --input sgfs_dir --output dataset_dir \\
      --katahex build/katahex.exe --model model.bin.gz --threads 4
        """
    )

    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing .sgfs files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for conversation .jsonl files')
    parser.add_argument('--katahex', default='bin/katahex',
                       help='Path to KataHex executable (default: bin/katahex)')
    parser.add_argument('--model', default='katahex_model_20220618.bin.gz',
                       help='Path to neural network model (default: katahex_model_20220618.bin.gz)')
    parser.add_argument('--config', default='processing_config.cfg',
                       help='Path to config file (default: processing_config.cfg)')
    parser.add_argument('--threads', '-t', type=int, default=1,
                       help='Number of parallel threads for conversation generation (default: 1)')

    args = parser.parse_args()

    # Find all SGFS files
    print(f"Searching for SGFS files in: {args.input}")
    sgfs_files = find_sgfs_files(args.input)

    if not sgfs_files:
        print("No .sgfs files found!")
        return 1

    print(f"Found {len(sgfs_files)} SGFS files\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Create processor
    processor = SGFSProcessor(
        katahex_path=args.katahex,
        model_path=args.model,
        config_path=args.config
    )

    # Process all SGFS files
    total_games = 0
    for idx, sgfs_file in enumerate(sgfs_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {idx}/{len(sgfs_files)}: {sgfs_file.name}")
        print(f"{'='*60}")

        # Create subdirectory for this file's output
        file_output_dir = output_dir / sgfs_file.stem

        try:
            processor.process_sgfs_file(
                sgfs_path=str(sgfs_file),
                output_dir=str(file_output_dir),
                num_threads=args.threads
            )

            # Count generated files
            if file_output_dir.exists():
                game_files = list(file_output_dir.glob("*.jsonl"))
                total_games += len(game_files)
                print(f"Generated {len(game_files)} conversation files")

        except Exception as e:
            print(f"Error processing {sgfs_file.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"Total conversation files generated: {total_games}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

