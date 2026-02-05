#!/usr/bin/env python3
"""
Module 4: Orchestrator
Main orchestration module that ties everything together.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from sgf_parser import parse_sgfs_file
from nn_evaluator import KataHexEvaluator
from game_processor import GameHistoryProcessor, GameData
from conversation_generator import generate_conversation_from_game


class SGFSProcessor:
    """
    Main orchestrator for processing SGFS files with neural network evaluation.
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
    
    def process_sgfs_file(self,
                         sgfs_path: str,
                         output_dir: str,
                         max_games: Optional[int] = None,
                         num_threads: int = 1,
                         num_gpus: int = 1) -> int:
        """
        Process an SGFS file and generate conversation data.
        MULTI-GPU VERSION: Can use multiple GPUs for parallel processing.

        Args:
            sgfs_path: Path to the .sgfs file
            output_dir: Directory to save conversation JSONL files
            max_games: Optional limit on number of games to process
            num_threads: Number of parallel game processing threads (should match num_gpus for best performance)
            num_gpus: Number of GPUs to use (default: 1)

        Returns:
            Number of games processed
        """
        print(f"Step 1: Parsing SGFS file: {sgfs_path}")
        games = parse_sgfs_file(sgfs_path)
        print(f"Found {len(games)} games")

        if max_games:
            games = games[:max_games]
            print(f"Processing first {max_games} games")

        os.makedirs(output_dir, exist_ok=True)

        if num_gpus > 1 and num_threads > 1:
            # Multi-GPU parallel processing
            return self._process_multi_gpu(games, output_dir, num_threads, num_gpus)
        else:
            # Single GPU sequential processing
            return self._process_single_gpu(games, output_dir)

    def _process_single_gpu(self, games: list, output_dir: str) -> int:
        """Process games sequentially on a single GPU."""
        print(f"\nStep 2: Starting neural network evaluator (Single GPU)...")
        print(f"Mode: {'MCTS search (slow, high quality)' if self.use_mcts else 'Raw NN (fast)'}")

        with KataHexEvaluator(
            katahex_path=self.katahex_path,
            model_path=self.model_path,
            config_path=self.config_path,
            use_mcts=self.use_mcts
        ) as evaluator:
            print("Evaluator started successfully")

            print(f"\nStep 3: Processing games (streaming mode - low memory usage)...")
            processor = GameHistoryProcessor(evaluator)

            games_processed = 0
            start_time = time.time()

            for idx, game in enumerate(games, 1):
                try:
                    # Process single game
                    game_data = processor.process_game(game)

                    # Immediately save conversation (don't accumulate in memory)
                    self._save_single_conversation(
                        game_data, output_dir,
                        include_policy=True,
                        include_value=True
                    )

                    games_processed += 1

                    # Progress update
                    if idx % 10 == 0 or idx == len(games):
                        elapsed = time.time() - start_time
                        rate = games_processed / elapsed if elapsed > 0 else 0
                        eta = (len(games) - idx) / rate if rate > 0 else 0
                        print(f"Progress: {idx}/{len(games)} games | "
                              f"Rate: {rate:.2f} games/sec | "
                              f"ETA: {eta/60:.1f} min")

                except Exception as e:
                    print(f"Error processing game {idx}: {e}")
                    continue

        print(f"\nProcessing complete!")
        print(f"Successfully processed {games_processed} games")
        print(f"Output directory: {output_dir}")

        return games_processed

    def _process_multi_gpu(self, games: list, output_dir: str, num_threads: int, num_gpus: int) -> int:
        """
        Process games in parallel using KataHex's built-in multi-GPU support.
        Creates ONE KataHex process with internal GPU threads (thread-safe).
        """
        print(f"\nStep 2: Creating multi-GPU config and starting KataHex...")
        print(f"Mode: {'MCTS search (slow, high quality)' if self.use_mcts else 'Raw NN (fast)'}")
        print(f"GPUs: {num_gpus}, Threads: {num_threads}")

        games_processed = 0
        games_lock = Lock()
        start_time = time.time()

        # Create temporary config file with multi-GPU settings
        import tempfile
        config_fd, temp_config_path = tempfile.mkstemp(suffix='.cfg', text=True)

        try:
            # Read base config
            base_config_lines = []
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    base_config_lines = f.readlines()

            # Add multi-GPU configuration
            with os.fdopen(config_fd, 'w') as f:
                # Write base config
                for line in base_config_lines:
                    # Skip any existing GPU settings
                    if not any(x in line for x in ['cudaGpuToUse', 'numNNServerThreads']):
                        f.write(line)

                # Add multi-GPU settings
                f.write(f"\n# Multi-GPU Configuration (auto-generated)\n")
                f.write(f"numNNServerThreadsPerModel = {num_gpus}\n")
                for i in range(num_gpus):
                    f.write(f"cudaGpuToUseModel0Thread{i} = {i}\n")

            print(f"Created multi-GPU config: {temp_config_path}")

            # Create ONE evaluator with built-in multi-GPU support
            print(f"Starting KataHex with {num_gpus} internal GPU threads...")
            evaluator = KataHexEvaluator(
                katahex_path=self.katahex_path,
                model_path=self.model_path,
                config_path=temp_config_path,
                use_mcts=self.use_mcts,
                gpu_id=None,  # Don't override - use config file settings
                max_visits=self.max_visits
            )
            evaluator.start()
            print(f"KataHex ready with {num_gpus} GPUs")

            processor = GameHistoryProcessor(evaluator)

            def process_single_game(idx, game):
                """Process a single game. KataHex handles GPU distribution internally."""
                try:
                    game_data = processor.process_game(game)

                    # Save conversation (thread-safe)
                    self._save_single_conversation(
                        game_data, output_dir,
                        include_policy=True,
                        include_value=True
                    )

                    # Update global progress (thread-safe)
                    nonlocal games_processed
                    with games_lock:
                        games_processed += 1
                        current_count = games_processed

                    # Progress update (every 10 games to reduce spam)
                    if current_count % 10 == 0 or current_count == len(games):
                        elapsed = time.time() - start_time
                        rate = current_count / elapsed if elapsed > 0 else 0
                        eta = (len(games) - current_count) / rate if rate > 0 else 0
                        print(f"Progress: {current_count}/{len(games)} games | "
                              f"Rate: {rate:.2f} games/sec | "
                              f"ETA: {eta/60:.1f} min")

                    return True

                except Exception as e:
                    print(f"Error processing game {idx}: {e}")
                    return False

            print(f"\nStep 3: Processing {len(games)} games with {num_threads} threads...")

            # Process all games with thread pool
            # KataHex's internal queue handles GPU distribution
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_single_game, idx, game)
                          for idx, game in enumerate(games)]

                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Worker error: {e}")

            # Cleanup
            print(f"\nShutting down KataHex...")
            evaluator.stop()

        finally:
            # Remove temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        print(f"\nProcessing complete!")
        print(f"Successfully processed {games_processed} games")
        print(f"Output directory: {output_dir}")

        return games_processed

    def _process_multi_gpu_with_file_mapping(self, games: list, output_dir: str, file_mapping: dict, num_threads: int, num_gpus: int) -> int:
        """
        Process games in parallel using KataHex's built-in multi-GPU support with file mapping.
        Creates ONE KataHex process with internal GPU threads (thread-safe).

        Args:
            games: List of all games from all files
            output_dir: Base output directory
            file_mapping: Dict mapping game id to original filename (stem)
            num_threads: Number of threads
            num_gpus: Number of GPUs
        """
        print(f"\nStep 2: Creating multi-GPU config and starting KataHex...")
        print(f"Mode: {'MCTS search (slow, high quality)' if self.use_mcts else 'Raw NN (fast)'}")
        print(f"GPUs: {num_gpus}, Threads: {num_threads}")

        games_processed = 0
        games_lock = Lock()
        start_time = time.time()

        # Create temporary config file with multi-GPU settings
        import tempfile
        config_fd, temp_config_path = tempfile.mkstemp(suffix='.cfg', text=True)

        try:
            # Read base config
            base_config_lines = []
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    base_config_lines = f.readlines()

            # Add multi-GPU configuration
            with os.fdopen(config_fd, 'w') as f:
                # Write base config
                for line in base_config_lines:
                    # Skip any existing GPU settings
                    if not any(x in line for x in ['cudaGpuToUse', 'numNNServerThreads']):
                        f.write(line)

                # Add multi-GPU settings
                f.write(f"\n# Multi-GPU Configuration (auto-generated)\n")
                f.write(f"numNNServerThreadsPerModel = {num_gpus}\n")
                for i in range(num_gpus):
                    f.write(f"cudaGpuToUseModel0Thread{i} = {i}\n")

            print(f"Created multi-GPU config: {temp_config_path}")

            # Create ONE evaluator with built-in multi-GPU support
            print(f"Starting KataHex with {num_gpus} internal GPU threads...")
            evaluator = KataHexEvaluator(
                katahex_path=self.katahex_path,
                model_path=self.model_path,
                config_path=temp_config_path,
                use_mcts=self.use_mcts,
                gpu_id=None,  # Don't override - use config file settings
                max_visits=self.max_visits
            )
            evaluator.start()
            print(f"KataHex ready with {num_gpus} GPUs")

            processor = GameHistoryProcessor(evaluator)

            def process_single_game(idx, game):
                """Process a single game. KataHex handles GPU distribution internally."""
                try:
                    game_data = processor.process_game(game)

                    # Determine output subdirectory based on original file
                    file_stem = file_mapping.get(id(game), "unknown")
                    file_output_dir = os.path.join(output_dir, file_stem)
                    os.makedirs(file_output_dir, exist_ok=True)

                    # Save conversation (thread-safe)
                    self._save_single_conversation(
                        game_data, file_output_dir,
                        include_policy=True,
                        include_value=True
                    )

                    # Update global progress (thread-safe)
                    nonlocal games_processed
                    with games_lock:
                        games_processed += 1
                        current_count = games_processed

                    # Progress update (every 10 games to reduce spam)
                    if current_count % 10 == 0 or current_count == len(games):
                        elapsed = time.time() - start_time
                        rate = current_count / elapsed if elapsed > 0 else 0
                        eta = (len(games) - current_count) / rate if rate > 0 else 0
                        print(f"Progress: {current_count}/{len(games)} games | "
                              f"Rate: {rate:.2f} games/sec | "
                              f"ETA: {eta/60:.1f} min")

                    return True

                except Exception as e:
                    print(f"Error processing game {idx}: {e}")
                    return False

            print(f"\nStep 3: Processing {len(games)} games with {num_threads} threads...")

            # Process all games with thread pool
            # KataHex's internal queue handles GPU distribution
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_single_game, idx, game)
                          for idx, game in enumerate(games)]

                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Worker error: {e}")

            # Cleanup
            print(f"\nShutting down KataHex...")
            evaluator.stop()

        finally:
            # Remove temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        print(f"\nProcessing complete!")
        print(f"Successfully processed {games_processed} games")
        print(f"Output directory: {output_dir}")

        return games_processed

    def _save_checkpoint(self, results: List[GameData], checkpoint_path: str):
        """
        Save checkpoint in JSONL format.

        Args:
            results: List of GameData objects
            checkpoint_path: Path to checkpoint file
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for game in results:
                json.dump(game.to_dict(), f)
                f.write('\n')

    def save_conversations(self,
                          results: List[GameData],
                          output_dir: str,
                          include_policy: bool = True,
                          include_value: bool = True,
                          num_threads: int = 1):
        """
        Save conversation data to output directory (one file per game).
        Each file contains one conversation per line (step).

        Args:
            results: List of GameData objects
            output_dir: Directory to save conversation files
            include_policy: Include top policy moves in output
            include_value: Include value estimates in output
            num_threads: Number of threads for parallel processing (default: 1)
        """
        os.makedirs(output_dir, exist_ok=True)

        if num_threads <= 1:
            # Sequential processing
            for game in results:
                self._save_single_conversation(game, output_dir, include_policy, include_value)
        else:
            # Multi-threaded processing
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for game in results:
                    future = executor.submit(
                        self._save_single_conversation,
                        game, output_dir, include_policy, include_value
                    )
                    futures.append(future)

                # Wait for all to complete
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 10 == 0 or completed == len(futures):
                        print(f"Progress: {completed}/{len(futures)} conversations generated")

        print(f"Saved {len(results)} game conversation files to {output_dir}")

    def _save_single_conversation(self, game: GameData, output_dir: str,
                                  include_policy: bool, include_value: bool):
        """Helper method to save a single game's conversation."""
        game_id = game.game_id or f"game_{id(game)}"
        output_path = os.path.join(output_dir, f"{game_id}.jsonl")

        conversations = generate_conversation_from_game(
            game,
            include_policy=include_policy,
            include_value=include_value
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                json.dump(conv, f)
                f.write('\n')

    def process_sgfs_file_threaded(self,
                                   sgfs_path: str,
                                   output_dir: str,
                                   max_games: Optional[int] = None,
                                   num_threads: int = 4,
                                   checkpoint_interval: int = 50,
                                   resume_from_checkpoint: Optional[str] = None) -> List[GameData]:
        """
        Process an SGFS file with multiple threads for parallel evaluation.

        Args:
            sgfs_path: Path to the .sgfs file
            output_dir: Directory to save conversation JSONL files
            max_games: Optional limit on number of games to process
            num_threads: Number of parallel worker threads for conversation generation
            checkpoint_interval: Save checkpoint every N games
            resume_from_checkpoint: Optional checkpoint file to resume from

        Returns:
            List of GameData objects
        """
        print(f"Step 1: Parsing SGFS file: {sgfs_path}")
        games = parse_sgfs_file(sgfs_path)
        print(f"Found {len(games)} games")

        if max_games:
            games = games[:max_games]
            print(f"Processing first {max_games} games")

        # Load checkpoint if resuming
        processed_game_ids = set()
        initial_results = []

        if resume_from_checkpoint:
            print(f"\nLoading checkpoint: {resume_from_checkpoint}")
            try:
                with open(resume_from_checkpoint, 'r', encoding='utf-8') as f:
                    for line in f:
                        game_data_dict = json.loads(line)
                        # Reconstruct GameData from dict
                        game_data = GameData.from_dict(game_data_dict)
                        initial_results.append(game_data)
                        processed_game_ids.add(game_data.game_id)

                print(f"Loaded {len(initial_results)} games from checkpoint")
                print(f"Resuming from game {len(initial_results) + 1}")

                # Filter out already processed games
                games = [g for g in games if g.game_id not in processed_game_ids]
                print(f"Remaining games to process: {len(games)}")

            except FileNotFoundError:
                print(f"Warning: Checkpoint file not found: {resume_from_checkpoint}")
                print("Starting from beginning...")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting from beginning...")

        print(f"\nStep 2: Starting {num_threads} worker threads with neural network evaluators...")

        # Results storage with thread-safe access
        results = initial_results.copy()  # Start with checkpoint data
        results_lock = Lock()
        processed_count = [len(initial_results)]  # Start from checkpoint count
        start_time = time.time()
        total_games = len(games) + len(initial_results)

        def process_game_worker(game_record, worker_id):
            """Worker function to process a single game."""
            try:
                # Each worker gets its own evaluator instance
                with KataHexEvaluator(
                    katahex_path=self.katahex_path,
                    model_path=self.model_path,
                    config_path=self.config_path
                ) as evaluator:
                    processor = GameHistoryProcessor(evaluator)
                    game_data = processor.process_game(game_record)

                    # Thread-safe result storage
                    with results_lock:
                        results.append(game_data)
                        processed_count[0] += 1
                        count = processed_count[0]

                        # Progress reporting
                        if count % 10 == 0 or count == total_games:
                            elapsed = time.time() - start_time
                            rate = (count - len(initial_results)) / elapsed if elapsed > 0 else 0
                            remaining = total_games - count
                            eta = remaining / rate if rate > 0 else 0
                            print(f"Progress: {count}/{total_games} games "
                                  f"({count*100//total_games}%) - "
                                  f"{rate:.1f} games/sec - "
                                  f"ETA: {eta:.0f}s")

                        # Checkpoint saving
                        if checkpoint_interval > 0 and count % checkpoint_interval == 0:
                            # Save checkpoint in output directory
                            checkpoint_path = os.path.join(output_dir, f"checkpoint_{count}.jsonl")
                            self._save_checkpoint(results.copy(), checkpoint_path)
                            print(f"Checkpoint saved: {checkpoint_path}")

                    return game_data

            except Exception as e:
                print(f"Error processing game (worker {worker_id}): {e}")
                return None

        # Process games in parallel
        print(f"\nStep 3: Processing {len(games)} games with {num_threads} threads...")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all games to the thread pool
            future_to_game = {
                executor.submit(process_game_worker, game, i % num_threads): game
                for i, game in enumerate(games)
            }

            # Wait for all to complete
            for future in as_completed(future_to_game):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker thread error: {e}")

        elapsed = time.time() - start_time
        print(f"\nStep 4: Processing complete!")
        print(f"Successfully processed {len(results)} games in {elapsed:.1f}s")
        print(f"Average: {len(results)/elapsed:.2f} games/sec")

        print(f"\nStep 5: Generating conversations and saving to {output_dir}")
        self.save_conversations(results, output_dir, num_threads=num_threads)

        return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process SGFS files with neural network evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate conversation data (default, 1 file per game to output dir)
  python orchestrator.py --input games.sgfs --output conversations_dir

  # Generate raw game data as JSONL
  python orchestrator.py --input games.sgfs --output results.jsonl --format jsonl

  # Multi-threaded processing with 4 threads
  python orchestrator.py --input games.sgfs --threads 4

  # Process only first 10 games
  python orchestrator.py --input games.sgfs --max-games 10
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input .sgfs file')
    parser.add_argument('--output', '-o',
                       help='Path to output JSON file (default: input_name_processed.json)')
    parser.add_argument('--katahex', default='build/katahex-win64-19-eigen.exe',
                       help='Path to KataHex executable')
    parser.add_argument('--model', default='katahex_model_20220618.bin.gz',
                       help='Path to neural network model')
    parser.add_argument('--config',
                       help='Path to config file (optional)')
    parser.add_argument('--max-games', type=int,
                       help='Maximum number of games to process')
    parser.add_argument('--threads', '-t', type=int, default=1,
                       help='Number of parallel worker threads for conversation generation (default: 1)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Save checkpoint every N games (default: 50, 0=disabled)')
    parser.add_argument('--resume', '-r',
                       help='Resume from checkpoint file (JSONL format)')

    args = parser.parse_args()

    # Determine output directory
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_conversations")

    # Create processor
    processor = SGFSProcessor(
        katahex_path=args.katahex,
        model_path=args.model,
        config_path=args.config
    )

    # Process file with multi-threaded batch processing
    if args.threads > 1 or args.resume or args.checkpoint_interval > 0:
        print(f"Using multi-threaded batch processing with {args.threads} conversation threads")
        processor.process_sgfs_file_threaded(
            sgfs_path=args.input,
            output_dir=args.output,
            max_games=args.max_games,
            num_threads=args.threads,
            checkpoint_interval=args.checkpoint_interval,
            resume_from_checkpoint=args.resume
        )
    else:
        print("Using single-threaded processing")
        processor.process_sgfs_file(
            sgfs_path=args.input,
            output_dir=args.output,
            max_games=args.max_games,
            num_threads=args.threads
        )


if __name__ == '__main__':
    main()

