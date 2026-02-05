#!/usr/bin/env python3
"""
KataHex Rollout Generation Script

This script generates game rollouts between models using the KataHex match engine.
It provides an easy-to-use interface for running matches and organizing output.

Usage:
    python generate_rollouts.py --num-games 100 --output-dir rollouts_output
    python generate_rollouts.py --config custom_config.cfg --num-games 50
    python generate_rollouts.py --model1 model_a.bin.gz --model2 model_b.bin.gz --num-games 200

    # Diversity modes for rich training data
    python generate_rollouts.py --diversity-mode strength --num-games 500
    python generate_rollouts.py --diversity-mode temperature --num-games 500
    python generate_rollouts.py --diversity-mode exploration --num-games 500
"""

import argparse
import os
import sys
import subprocess
import datetime
import shutil
import random
from pathlib import Path


def apply_diversity_preset(args):
    """Apply diversity preset configurations."""
    if not args.diversity_mode:
        return

    mode = args.diversity_mode.lower()

    if mode == 'strength':
        # Vary strength levels: weak vs strong
        args.model2 = args.model1  # Self-play with different strengths
        args.bot_name1 = "Weak_Bot"
        args.bot_name2 = "Strong_Bot"
        args.max_visits_bot1 = 100
        args.max_visits_bot2 = 800
        args.temp_early_bot1 = 0.80
        args.temp_early_bot2 = 0.50
        args.temp_bot1 = 0.30
        args.temp_bot2 = 0.10
        print("Applied STRENGTH diversity preset: weak (100 visits) vs strong (800 visits)")

    elif mode == 'temperature':
        # Vary temperature profiles: aggressive vs conservative
        args.model2 = args.model1
        args.bot_name1 = "Aggressive"
        args.bot_name2 = "Conservative"
        args.max_visits_bot1 = 400
        args.max_visits_bot2 = 400
        args.temp_early_bot1 = 0.90
        args.temp_early_bot2 = 0.40
        args.temp_bot1 = 0.40
        args.temp_bot2 = 0.10
        print("Applied TEMPERATURE diversity preset: aggressive vs conservative play")

    elif mode == 'exploration':
        # Vary exploration parameters
        args.model2 = args.model1
        args.bot_name1 = "Explorer"
        args.bot_name2 = "Exploiter"
        args.max_visits_bot1 = 500
        args.max_visits_bot2 = 500
        args.cpuct_bot1 = 1.2
        args.cpuct_bot2 = 0.7
        args.temp_early_bot1 = 0.70
        args.temp_early_bot2 = 0.50
        print("Applied EXPLORATION diversity preset: high exploration vs low exploration")

    elif mode == 'speed':
        # Fast vs slow thinking
        args.model2 = args.model1
        args.bot_name1 = "Fast"
        args.bot_name2 = "Slow"
        args.max_visits_bot1 = 150
        args.max_visits_bot2 = 1000
        args.temp_early_bot1 = 0.70
        args.temp_early_bot2 = 0.50
        print("Applied SPEED diversity preset: fast (150 visits) vs slow (1000 visits)")

    elif mode == 'random':
        # Randomize parameters for maximum diversity
        args.model2 = args.model1
        args.bot_name1 = "Random_A"
        args.bot_name2 = "Random_B"
        args.max_visits_bot1 = random.randint(10, 200)
        args.max_visits_bot2 = random.randint(10, 200)
        args.temp_early_bot1 = round(random.uniform(0.4, 1.3), 2)
        args.temp_early_bot2 = round(random.uniform(0.4, 1.3), 2)
        args.temp_bot1 = round(random.uniform(0.1, 1.4), 2)
        args.temp_bot2 = round(random.uniform(0.1, 1.4), 2)
        args.cpuct_bot1 = round(random.uniform(0.6, 1.3), 2)
        args.cpuct_bot2 = round(random.uniform(0.6, 1.3), 2)
        print(f"Applied RANDOM diversity preset:")
        print(f"  Bot1: visits={args.max_visits_bot1}, tempEarly={args.temp_early_bot1}, temp={args.temp_bot1}, cpuct={args.cpuct_bot1}")
        print(f"  Bot2: visits={args.max_visits_bot2}, tempEarly={args.temp_early_bot2}, temp={args.temp_bot2}, cpuct={args.cpuct_bot2}")
    else:
        print(f"Warning: Unknown diversity mode '{mode}', ignoring")


def create_match_config(args, config_path):
    """Create a match configuration file based on arguments."""

    config_lines = [
        "# Auto-generated KataHex Match Configuration",
        f"# Generated at: {datetime.datetime.now().isoformat()}",
        "",
        "# Logs",
        "logSearchInfo = false",
        "logMoves = true",
        f"logGamesEvery = {max(1, args.num_games // 10)}",
        "logToStdout = true",
        "",
        "# Bots",
    ]

    # Configure bots
    if args.model2:
        # Two different models or same model with different configs
        config_lines.extend([
            "numBots = 2",
            f"botName0 = {args.bot_name1}",
            f"botName1 = {args.bot_name2}",
            f"nnModelFile0 = {args.model1}",
            f"nnModelFile1 = {args.model2}",
        ])
    else:
        # Self-play with one model
        config_lines.extend([
            "numBots = 1",
            f"botName0 = {args.bot_name1}",
            f"nnModelFile0 = {args.model1}",
        ])

    config_lines.extend([
        "",
        "# Match settings",
        f"numGameThreads = {args.num_threads}",
        f"numGamesTotal = {args.num_games}",
        "maxMovesPerGame = 1200",
        "",
        "# Resignation - DISABLED for full trajectories",
        "allowResignation = false",
        "resignThreshold = -0.95",
        "resignConsecTurns = 1",
        "",
        "# Early draw - DISABLED for full trajectories",
        "allowEarlyDraw = false",
        "earlyDrawThreshold = 0.98",
        "earlyDrawConsecTurns = 4",
        "earlyDrawProbSelfplay = 0.9",
        "",
        "# Rules",
        f"bSizes = {args.board_sizes}",
        f"bSizeRelProbs = {args.board_sizes_probs}",
        "scoringRules = AREA",
        "",
        "# FIXED: Jump connections are now disabled in C++ code (gamelogic.cpp line 316)",
        "# No need for maxMoves workaround - let games run until natural completion",
        "# Setting moveLimitProb = 0.0 disables move limits entirely",
        "moveLimitProb = 0.0",
        "",
    ])

    # Per-bot search parameters
    if args.model2:
        # Bot-specific maxVisits
        config_lines.extend([
            "# Search limits - Per-bot configuration",
            f"maxVisits0 = {getattr(args, 'max_visits_bot1', args.max_visits)}",
            f"maxVisits1 = {getattr(args, 'max_visits_bot2', args.max_visits)}",
            "numSearchThreads = 1",
        ])
    else:
        # Single bot
        config_lines.extend([
            "# Search limits",
            f"maxVisits = {args.max_visits}",
            "numSearchThreads = 1",
        ])

    # GPU configuration
    num_gpus = getattr(args, 'num_gpus', 1)
    gpu_ids = getattr(args, 'gpu_ids', None)

    config_lines.extend([
        "",
        "# GPU Settings",
        "nnMaxBatchSize = 32",
        "nnCacheSizePowerOfTwo = 21",
        "nnMutexPoolSizePowerOfTwo = 17",
        "nnRandomize = true",
        "",
    ])

    # Multi-GPU configuration
    if num_gpus > 1 or gpu_ids:
        # Parse GPU IDs if provided
        if gpu_ids:
            gpu_list = [int(x.strip()) for x in gpu_ids.split(',')]
            num_gpus = len(gpu_list)
        else:
            gpu_list = list(range(num_gpus))

        config_lines.extend([
            "# Multi-GPU Configuration",
            f"numNNServerThreadsPerModel = {num_gpus}",
            "",
            "# CUDA GPU device assignment (one thread per GPU)",
        ])

        # Assign each thread to a specific GPU
        for i, gpu_id in enumerate(gpu_list):
            config_lines.append(f"cudaGpuToUseModel0Thread{i} = {gpu_id}")

        config_lines.extend([
            "",
            "cudaUseFP16 = auto",
            "cudaUseNHWC = auto",
            "",
        ])
    else:
        # Single GPU configuration
        config_lines.extend([
            "# Single GPU Configuration",
            "numNNServerThreadsPerModel = 1",
            f"cudaGpuToUse = {args.gpu_id}",
            "cudaUseFP16 = auto",
            "cudaUseNHWC = auto",
            "",
        ])

    config_lines.extend([
        "# Opening initialization - DISABLED for full MCTS on all moves",
        "initGamesWithPolicy = false",
        "policyInitAvgMoveNum = 0.0",
        "completelyRandomOpeningProb = 0.0",
        "specialOpeningProb = 0.0",
        "",
    ])

    # Per-bot move selection parameters
    if args.model2:
        config_lines.extend([
            "# Move selection - Per-bot configuration",
            f"chosenMoveTemperatureEarly0 = {getattr(args, 'temp_early_bot1', 0.60)}",
            f"chosenMoveTemperatureEarly1 = {getattr(args, 'temp_early_bot2', 0.60)}",
            f"chosenMoveTemperature0 = {getattr(args, 'temp_bot1', 0.20)}",
            f"chosenMoveTemperature1 = {getattr(args, 'temp_bot2', 0.20)}",
        ])
    else:
        config_lines.extend([
            "# Move selection",
            f"chosenMoveTemperatureEarly = {getattr(args, 'temp_early', 0.60)}",
            f"chosenMoveTemperature = {getattr(args, 'temp', 0.20)}",
        ])

    # Per-bot MCTS exploration parameters (if specified)
    if args.model2 and (hasattr(args, 'cpuct_bot1') or hasattr(args, 'cpuct_bot2')):
        config_lines.extend([
            "",
            "# MCTS Exploration - Per-bot configuration",
        ])
        if hasattr(args, 'cpuct_bot1'):
            config_lines.append(f"cpuctExploration0 = {args.cpuct_bot1}")
        if hasattr(args, 'cpuct_bot2'):
            config_lines.append(f"cpuctExploration1 = {args.cpuct_bot2}")
    elif hasattr(args, 'cpuct'):
        config_lines.extend([
            "",
            "# MCTS Exploration",
            f"cpuctExploration = {args.cpuct}",
        ])

    with open(config_path, 'w') as f:
        f.write('\n'.join(config_lines))

    print(f"Created config file: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate game rollouts between KataHex models with diversity options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Self-play with default model (100 games)
  python generate_rollouts.py --num-games 100

  # Match between two models
  python generate_rollouts.py --model1 model_a.bin.gz --model2 model_b.bin.gz --num-games 200

  # Custom configuration
  python generate_rollouts.py --config my_config.cfg --output-dir my_rollouts

  # High-quality games with more visits
  python generate_rollouts.py --num-games 50 --max-visits 1000

  # Diversity modes for rich training data
  python generate_rollouts.py --diversity-mode strength --num-games 500
  python generate_rollouts.py --diversity-mode temperature --num-games 500
  python generate_rollouts.py --diversity-mode exploration --num-games 500
  python generate_rollouts.py --diversity-mode speed --num-games 500
  python generate_rollouts.py --diversity-mode random --num-games 500

  # Manual per-bot configuration
  python generate_rollouts.py --model1 model.bin.gz --model2 model.bin.gz \\
      --max-visits-bot1 200 --max-visits-bot2 800 \\
      --temp-early-bot1 0.8 --temp-early-bot2 0.5 \\
      --num-games 500
        """
    )

    parser.add_argument('--config', type=str, help='Path to existing config file (overrides other options)')
    parser.add_argument('--model1', type=str, default='katahex_model_20220618.bin.gz',
                        help='Path to first model file')
    parser.add_argument('--model2', type=str, help='Path to second model file (for head-to-head matches)')
    parser.add_argument('--bot-name1', type=str, default='Model_A', help='Name for first bot')
    parser.add_argument('--bot-name2', type=str, default='Model_B', help='Name for second bot')

    parser.add_argument('--num-games', type=int, default=100, help='Number of games to generate')
    parser.add_argument('--max-visits', type=int, default=500, help='MCTS visits per move (default for both bots)')
    parser.add_argument('--num-threads', type=int, default=4, help='Number of parallel game threads')
    parser.add_argument('--board-sizes', type=str, default='11,9,7,5', help='Board sizes (comma-separated, max 11 for Hex)')
    parser.add_argument('--board-sizes-probs', type=str, default='40,30,20,10', help='Board sizes Provabilities (comma-separated)')

    # GPU configuration
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID (for single GPU)')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use (e.g., 2 for GPUs 0,1)')
    parser.add_argument('--gpu-ids', type=str, help='Specific GPU IDs to use (e.g., "0,2,3" for GPUs 0, 2, and 3)')

    # Diversity options
    parser.add_argument('--diversity-mode', type=str,
                        choices=['strength', 'temperature', 'exploration', 'speed', 'random'],
                        help='Preset diversity mode: strength, temperature, exploration, speed, or random')

    # Per-bot parameters
    parser.add_argument('--max-visits-bot1', type=int, help='MCTS visits for bot 1')
    parser.add_argument('--max-visits-bot2', type=int, help='MCTS visits for bot 2')
    parser.add_argument('--temp-early-bot1', type=float, help='Early game temperature for bot 1')
    parser.add_argument('--temp-early-bot2', type=float, help='Early game temperature for bot 2')
    parser.add_argument('--temp-bot1', type=float, help='Late game temperature for bot 1')
    parser.add_argument('--temp-bot2', type=float, help='Late game temperature for bot 2')
    parser.add_argument('--cpuct-bot1', type=float, help='CPUCT exploration parameter for bot 1')
    parser.add_argument('--cpuct-bot2', type=float, help='CPUCT exploration parameter for bot 2')

    # Global parameters (when not using per-bot)
    parser.add_argument('--temp-early', type=float, help='Early game temperature (global)')
    parser.add_argument('--temp', type=float, help='Late game temperature (global)')
    parser.add_argument('--cpuct', type=float, help='CPUCT exploration parameter (global)')

    parser.add_argument('--output-dir', type=str, default='rollouts_output', help='Output directory for SGF files')
    parser.add_argument('--katahex-path', type=str, default='./build/katahex', help='Path to katahex executable')

    args = parser.parse_args()

    # Apply diversity preset if specified
    apply_diversity_preset(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SGF output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sgf_dir = output_dir / f"sgfs_{timestamp}"
    sgf_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"match_{timestamp}.log"

    # Use provided config or create a new one
    if args.config:
        config_file = args.config
        print(f"Using existing config: {config_file}")
    else:
        config_file = output_dir / f"config_{timestamp}.cfg"
        create_match_config(args, config_file)

    # Check if katahex executable exists
    if not os.path.exists(args.katahex_path):
        print(f"Error: KataHex executable not found at {args.katahex_path}")
        print("Please build KataHex first or specify the correct path with --katahex-path")
        sys.exit(1)

    # Check if model files exist
    if not os.path.exists(args.model1):
        print(f"Error: Model file not found: {args.model1}")
        sys.exit(1)
    if args.model2 and not os.path.exists(args.model2):
        print(f"Error: Model file not found: {args.model2}")
        sys.exit(1)

    # Print summary
    print("\n" + "="*70)
    print("KataHex Rollout Generation")
    print("="*70)
    print(f"Model 1:        {args.model1}")
    if args.model2:
        print(f"Model 2:        {args.model2}")
        print(f"Bot 1 ({args.bot_name1}):")
        print(f"  Max visits:   {getattr(args, 'max_visits_bot1', args.max_visits)}")
        print(f"  Temp early:   {getattr(args, 'temp_early_bot1', 0.60)}")
        print(f"  Temp late:    {getattr(args, 'temp_bot1', 0.20)}")
        if hasattr(args, 'cpuct_bot1'):
            print(f"  CPUCT:        {args.cpuct_bot1}")
        print(f"Bot 2 ({args.bot_name2}):")
        print(f"  Max visits:   {getattr(args, 'max_visits_bot2', args.max_visits)}")
        print(f"  Temp early:   {getattr(args, 'temp_early_bot2', 0.60)}")
        print(f"  Temp late:    {getattr(args, 'temp_bot2', 0.20)}")
        if hasattr(args, 'cpuct_bot2'):
            print(f"  CPUCT:        {args.cpuct_bot2}")
    else:
        print(f"Mode:           Self-play")
        print(f"Max visits:     {args.max_visits}")
    print(f"Games:          {args.num_games}")
    print(f"Parallel games: {args.num_threads}")
    print(f"Board sizes:    {args.board_sizes}")
    print(f"Board sizes Probs:    {args.board_sizes_probs}")
    print(f"Output dir:     {output_dir}")
    print(f"SGF dir:        {sgf_dir}")
    print(f"Log file:       {log_file}")
    print("="*70 + "\n")

    # Run the match
    cmd = [
        # "build/katahex-win64-19-opencl.exe",
        # "build/katahex-win64-19-eigen.exe",
        # "./katahex-cuda",
        f"{args.katahex_path}",
        "match",
        "-config", str(config_file),
        "-log-file", str(log_file),
        "-sgf-output-dir", str(sgf_dir)
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("Rollout generation completed successfully!")
        print("="*70)
        print(f"SGF files saved to: {sgf_dir}")
        print(f"Log file: {log_file}")

        # Count generated SGF files
        sgf_files = list(sgf_dir.glob("*.sgfs"))
        if sgf_files:
            print(f"Generated {len(sgf_files)} SGF file(s)")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\nError: Match process failed with exit code {e.returncode}")
        print(f"Check the log file for details: {log_file}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be available in:")
        print(f"  SGF dir: {sgf_dir}")
        print(f"  Log file: {log_file}")
        return 130


if __name__ == "__main__":
    sys.exit(main())

