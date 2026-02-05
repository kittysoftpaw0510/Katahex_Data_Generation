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

    sgfs_files = list(input_path.rglob("*.sgfs"))
    return sorted(sgfs_files)


def main():
    parser = argparse.ArgumentParser(
        description='Generate conversation datasets from SGFS files (multiprocessing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Fast mode (default) - uses raw NN evaluation
python generate_dataset.py --input sgfs_dir --output dataset_dir --num-gpus 8 --processes 8

# Quality mode - uses MCTS search (slower but better quality)
python generate_dataset.py --input sgfs_dir --output dataset_dir --num-gpus 8 --processes 8 --use-mcts

# With custom KataHex path and model
python generate_dataset.py \
  --input rollouts_output \
  --output out_processed \
  --num-gpus 8 \
  --processes 24 \
  --use-mcts \
  --max-visits 500 \
  --katahex bin/katahex \
  --config processor_gtp.cfg
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
    parser.add_argument('--config', default='processor_config.cfg',
                       help='Path to config file (default: processing_config.cfg)')
    parser.add_argument('--processes', '-p', type=int, default=1,
                       help='Number of parallel worker processes (default: 1)')
    parser.add_argument('--num-gpus', '-g', type=int, default=1,
                       help='Number of GPUs to use for parallel processing (default: 1)')
    parser.add_argument('--use-mcts', action='store_true',
                       help='Use MCTS search for evaluation (slow but high quality). Default: raw NN (fast)')
    parser.add_argument('--max-visits', type=int, default=1600,
                       help='Maximum number of MCTS visits (only used with --use-mcts, default: 1600)')

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

    # Validate GPU/process configuration
    if args.num_gpus > 1 and args.processes < args.num_gpus:
        print(f"WARNING: Using {args.num_gpus} GPUs but only {args.processes} processes.")
        print(f"         Recommend setting --processes >= --num-gpus for best performance.")
        print(f"         Auto-adjusting processes to {args.num_gpus}")
        args.processes = args.num_gpus

    # Create processor
    processor = SGFSProcessor(
        katahex_path=args.katahex,
        model_path=args.model,
        config_path=args.config,
        use_mcts=args.use_mcts,
        max_visits=args.max_visits
    )

    print(f"Evaluation mode: {'MCTS search (slow, high quality)' if args.use_mcts else 'Raw NN (fast)'}")
    print(f"GPU configuration: {args.num_gpus} GPU(s), {args.processes} process(es)")
    print(f"\nProcessing {len(sgfs_files)} SGFS files...")

    # Process each file with the unified multiprocessing architecture
    # The new orchestrator automatically handles multi-GPU distribution
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
                num_processes=args.processes,
                num_gpus=args.num_gpus
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

