#!/usr/bin/env python3
"""
Module 4: Orchestrator
Main orchestration module with unified worker-based architecture using multiprocessing.

Architecture:
- One persistent KataHex process per worker process (true parallelism)
- Workers pull games from a shared multiprocessing queue
- Supports single GPU, multi-GPU, and CPU modes
- Maximum GPU utilization through multiple workers per GPU
- Uses multiprocessing instead of threading for better performance
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Optional
from multiprocessing import Process, Queue, Manager
from queue import Empty
import time
from sgf_parser import parse_sgfs_file
from nn_evaluator import KataHexEvaluator
from game_processor import GameHistoryProcessor, GameData
from conversation_generator import generate_conversation_from_game


class SGFSProcessor:
    """
    Main orchestrator for processing SGFS files with neural network evaluation.
    Uses a unified worker-based architecture for maximum GPU utilization.
    """

    def __init__(self,
                 katahex_path: str = "build/katahex-win64-19-eigen.exe",
                 model_path: str = "katahex_model_20220618.bin.gz",
                 config_path: Optional[str] = None,
                 use_mcts: bool = False,
                 max_visits: int = 1600):
        """
        Initialize the processor.

        Args:
            katahex_path: Path to KataHex executable
            model_path: Path to neural network model
            config_path: Optional path to config file
            use_mcts: If True, use MCTS search (slow, high quality). If False, use raw NN (fast)
            max_visits: Maximum number of MCTS visits (only used when use_mcts=True)
        """
        self.katahex_path = katahex_path
        self.model_path = model_path
        self.config_path = config_path
        self.use_mcts = use_mcts
        self.max_visits = max_visits

    @staticmethod
    def _worker_process(worker_id: int, num_gpus: int, game_queue: Queue,
                       output_dir: str, stats_dict: dict, start_time: float,
                       katahex_path: str, model_path: str, config_path: Optional[str],
                       use_mcts: bool, max_visits: int):
        """
        Worker process that processes games from the queue.
        Each worker has its own persistent KataHex process.

        Args:
            worker_id: Unique worker ID
            num_gpus: Total number of GPUs (0 for CPU mode)
            game_queue: Multiprocessing Queue of (idx, game) tuples to process
            output_dir: Directory to save output files
            stats_dict: Shared statistics dict (Manager.dict)
            start_time: Processing start time for ETA calculation
            katahex_path: Path to KataHex executable
            model_path: Path to neural network model
            config_path: Optional path to config file
            use_mcts: If True, use MCTS search
            max_visits: Maximum number of MCTS visits
        """
        # Determine GPU assignment (round-robin, None for CPU)
        gpu_id = worker_id % num_gpus if num_gpus > 0 else None

        # Create ONE persistent evaluator for this worker
        evaluator = KataHexEvaluator(
            katahex_path=katahex_path,
            model_path=model_path,
            config_path=config_path,
            use_mcts=use_mcts,
            gpu_id=gpu_id,
            max_visits=max_visits
        )

        try:
            evaluator.start()
            processor = GameHistoryProcessor(evaluator)

            # Process games from queue until empty
            while True:
                try:
                    idx, game = game_queue.get(timeout=1)
                except Empty:
                    break  # Queue is empty, worker is done

                try:
                    # Skip games with no moves
                    if not game.moves:
                        stats_dict['errors'] = stats_dict.get('errors', 0) + 1
                        stats_dict['skipped_no_moves'] = stats_dict.get('skipped_no_moves', 0) + 1
                        continue

                    # Process game
                    game_data = processor.process_game(game)

                    # Generate conversations
                    conversations = generate_conversation_from_game(
                        game_data,
                        include_policy=True,
                        include_value=True
                    )

                    # Skip if no valid conversations generated
                    if not conversations:
                        stats_dict['errors'] = stats_dict.get('errors', 0) + 1
                        stats_dict['skipped_no_conversations'] = stats_dict.get('skipped_no_conversations', 0) + 1
                        continue

                    # Save conversation
                    game_id = game_data.game_id or f"game_{id(game_data)}"
                    output_path = os.path.join(output_dir, f"{game_id}.jsonl")

                    with open(output_path, 'w', encoding='utf-8') as f:
                        for conv in conversations:
                            json.dump(conv, f)
                            f.write('\n')

                    # Update statistics (process-safe with Manager.dict)
                    stats_dict['processed'] = stats_dict.get('processed', 0) + 1
                    current_count = stats_dict['processed']
                    total_count = stats_dict['total']

                    # Progress update (every 10 games)
                    if current_count % 10 == 0 or current_count == total_count:
                        elapsed = time.time() - start_time
                        rate = current_count / elapsed if elapsed > 0 else 0
                        eta = (total_count - current_count) / rate if rate > 0 else 0

                        device = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
                        print(f"Progress: {current_count}/{total_count} games | "
                              f"Rate: {rate:.2f} games/sec | "
                              f"ETA: {eta/60:.1f} min | Worker {worker_id} ({device})")

                except Exception as e:
                    print(f"Error processing game {idx} on worker {worker_id}: {e}")
                    stats_dict['errors'] = stats_dict.get('errors', 0) + 1
                    stats_dict['skipped_exceptions'] = stats_dict.get('skipped_exceptions', 0) + 1

        finally:
            # Cleanup evaluator when worker is done
            evaluator.stop()
    
    def process_sgfs_file(self,
                         sgfs_path: str,
                         output_dir: str,
                         max_games: Optional[int] = None,
                         num_processes: int = 1,
                         num_gpus: int = 1) -> int:
        """
        Process an SGFS file and generate conversation data.
        Unified worker-based architecture for single/multi-GPU/CPU.

        Args:
            sgfs_path: Path to the .sgfs file
            output_dir: Directory to save conversation JSONL files
            max_games: Optional limit on number of games to process
            num_processes: Number of worker processes (default: 1)
            num_gpus: Number of GPUs to use (0 for CPU mode, default: 1)

        Returns:
            Number of games processed successfully
        """
        # Step 1: Parse SGFS file
        print(f"Step 1: Parsing SGFS file: {sgfs_path}")
        games = parse_sgfs_file(sgfs_path)
        print(f"Found {len(games)} games")

        if max_games:
            games = games[:max_games]
            print(f"Processing first {max_games} games")

        os.makedirs(output_dir, exist_ok=True)

        # Step 2: Setup worker architecture
        mode = "CPU" if num_gpus == 0 else f"{num_gpus} GPU(s)"
        eval_mode = "MCTS search (slow, high quality)" if self.use_mcts else "Raw NN (fast)"
        print(f"\nStep 2: Starting {num_processes} worker processes on {mode}")
        print(f"Mode: {eval_mode}")
        print(f"Each worker gets its own persistent KataHex process")

        # Create multiprocessing queue and shared statistics
        game_queue = Queue()
        for idx, game in enumerate(games):
            game_queue.put((idx, game))

        # Shared statistics using Manager
        manager = Manager()
        stats_dict = manager.dict()
        stats_dict['processed'] = 0
        stats_dict['errors'] = 0
        stats_dict['skipped_no_moves'] = 0
        stats_dict['skipped_no_conversations'] = 0
        stats_dict['skipped_exceptions'] = 0
        stats_dict['total'] = len(games)
        start_time = time.time()

        # Step 3: Start worker processes
        print(f"\nStep 3: Processing {len(games)} games with {num_processes} workers...")

        workers = []
        for i in range(num_processes):
            worker = Process(
                target=self._worker_process,
                args=(i, num_gpus, game_queue, output_dir, stats_dict, start_time,
                      self.katahex_path, self.model_path, self.config_path,
                      self.use_mcts, self.max_visits)
            )
            worker.start()
            workers.append(worker)

        # Wait for all workers to finish
        for worker in workers:
            worker.join()

        # Step 4: Report results
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Successfully processed: {stats_dict['processed']} games ({stats_dict['processed']*100//stats_dict['total']}%)")
        print(f"Total errors/skipped: {stats_dict['errors']} games ({stats_dict['errors']*100//stats_dict['total']}%)")
        print(f"\nError breakdown:")
        print(f"  - Games with no moves: {stats_dict.get('skipped_no_moves', 0)}")
        print(f"  - Games with no valid conversations: {stats_dict.get('skipped_no_conversations', 0)}")
        print(f"  - Games with exceptions: {stats_dict.get('skipped_exceptions', 0)}")
        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Average rate: {stats_dict['processed']/elapsed:.2f} games/sec")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        return stats_dict['processed']


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process SGFS files with neural network evaluation (multiprocessing architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU, single process (sequential)
  python orchestrator.py --input games.sgfs --output out --num-gpus 1 --processes 1

  # Single GPU, multiple processes (parallel on same GPU)
  python orchestrator.py --input games.sgfs --output out --num-gpus 1 --processes 4

  # Multi-GPU with multiple processes (maximum utilization)
  python orchestrator.py --input games.sgfs --output out --num-gpus 8 --processes 24

  # CPU mode (no GPU)
  python orchestrator.py --input games.sgfs --output out --num-gpus 0 --processes 4

  # MCTS mode (slow, high quality)
  python orchestrator.py --input games.sgfs --output out --use-mcts --max-visits 500

  # Process only first 100 games
  python orchestrator.py --input games.sgfs --output out --max-games 100
        """
    )

    parser.add_argument('--input', '-i', required=True,
                       help='Path to input .sgfs file')
    parser.add_argument('--output', '-o',
                       help='Output directory (default: input_name_processed)')
    parser.add_argument('--katahex', default='build/katahex-win64-19-eigen.exe',
                       help='Path to KataHex executable')
    parser.add_argument('--model', default='katahex_model_20220618.bin.gz',
                       help='Path to neural network model')
    parser.add_argument('--config',
                       help='Path to config file (optional)')
    parser.add_argument('--max-games', type=int,
                       help='Maximum number of games to process')
    parser.add_argument('--processes', '-p', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use (0 for CPU mode, default: 1)')
    parser.add_argument('--use-mcts', action='store_true',
                       help='Use MCTS search (slow, high quality) instead of raw NN')
    parser.add_argument('--max-visits', type=int, default=1600,
                       help='Maximum MCTS visits (only used with --use-mcts, default: 1600)')

    args = parser.parse_args()

    # Determine output directory
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_processed")

    # Create processor
    processor = SGFSProcessor(
        katahex_path=args.katahex,
        model_path=args.model,
        config_path=args.config,
        use_mcts=args.use_mcts,
        max_visits=args.max_visits
    )

    # Process file with unified multiprocessing architecture
    processor.process_sgfs_file(
        sgfs_path=args.input,
        output_dir=args.output,
        max_games=args.max_games,
        num_threads=args.processes,
        num_gpus=args.num_gpus
    )


if __name__ == '__main__':
    main()

