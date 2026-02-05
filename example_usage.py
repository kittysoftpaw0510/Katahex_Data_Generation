#!/usr/bin/env python3
"""
Example usage of the SGFS processor modules.
"""

from sgf_parser import parse_sgfs_file
from nn_evaluator import KataHexEvaluator, BoardState
from game_processor import GameHistoryProcessor
from orchestrator import SGFSProcessor
from conversation_generator import generate_conversation_from_game, generate_training_trajectory


def example_1_parse_only():
    """Example 1: Just parse an SGFS file without evaluation."""
    print("=" * 70)
    print("Example 1: Parse SGFS file")
    print("=" * 70)
    
    sgfs_file = 'dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs'
    
    print(f"\nParsing: {sgfs_file}")
    games = parse_sgfs_file(sgfs_file)
    
    print(f"\nFound {len(games)} games\n")
    
    for i, game in enumerate(games[:3], 1):  # Show first 3 games
        print(f"Game {i}:")
        print(f"  Board size: {game.board_size}x{game.board_size}")
        print(f"  Players: {game.black_player} (B) vs {game.white_player} (W)")
        print(f"  Result: {game.result}")
        print(f"  Moves: {game.get_move_count()}")
        
        # Show first few moves
        print(f"  First 5 moves:")
        for move in game.moves[:5]:
            comment_data = move.parse_comment()
            if comment_data:
                print(f"    {move.player}[{move.location}] - Win%: {comment_data['win_prob']:.2f}")
            else:
                print(f"    {move.player}[{move.location}]")
        print()


def example_2_evaluate_position():
    """Example 2: Evaluate a single board position."""
    print("=" * 70)
    print("Example 2: Evaluate a single position")
    print("=" * 70)
    
    # Create a simple board state
    board_state = BoardState(
        board_size=7,
        black_stones=['e3', 'f4'],
        white_stones=['d4', 'e5'],
        next_player='B',
        move_history=[
            ('B', 'e3'),
            ('W', 'd4'),
            ('B', 'f4'),
            ('W', 'e5')
        ]
    )
    
    print("\nBoard state:")
    print(f"  Size: {board_state.board_size}x{board_state.board_size}")
    print(f"  Black stones: {board_state.black_stones}")
    print(f"  White stones: {board_state.white_stones}")
    print(f"  Next player: {board_state.next_player}")
    
    print("\nStarting evaluator...")
    with KataHexEvaluator(
        katahex_path="build/katahex-win64-19-eigen.exe",
        model_path="katahex_model_20220618.bin.gz",
        config_path="cpp/configs/gtp_example.cfg"
    ) as evaluator:
        print("Evaluating position...")
        evaluation = evaluator.evaluate(board_state)
        
        print("\nEvaluation results:")
        print(f"  Value: {evaluation.value:.3f}")
        print(f"  Win probability: {evaluation.win_prob:.3f}")
        print(f"  Loss probability: {evaluation.loss_prob:.3f}")
        print(f"  Draw probability: {evaluation.draw_prob:.3f}")
        print(f"\n  Full policy map:")

        # Sort policy by probability for better readability
        sorted_policy = sorted(evaluation.policy.items(),
                              key=lambda x: x[1],
                              reverse=True)
        for move, prob in sorted_policy:
            print(f"    {move}: {prob:.6f}")


def example_3_process_game():
    """Example 3: Process a complete game with step-by-step evaluation."""
    print("=" * 70)
    print("Example 3: Process complete game")
    print("=" * 70)
    
    sgfs_file = 'dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs'
    
    print(f"\nParsing: {sgfs_file}")
    games = parse_sgfs_file(sgfs_file)
    
    if not games:
        print("No games found!")
        return
    
    game = games[0]  # Process first game
    print(f"\nProcessing game: {game.black_player} vs {game.white_player}")
    print(f"Total moves: {game.get_move_count()}")
    
    print("\nStarting evaluator...")
    with KataHexEvaluator(
        katahex_path="build/katahex-win64-19-eigen.exe",
        model_path="katahex_model_20220618.bin.gz",
        config_path="cpp/configs/gtp_example.cfg"
    ) as evaluator:
        processor = GameHistoryProcessor(evaluator)
        
        print("Processing game...")
        game_data = processor.process_game(game)
        
        print(f"\nProcessed {len(game_data.steps)} steps")
        print("\nFirst 5 steps:")
        
        for step in game_data.steps[:5]:
            print(f"\nMove {step.move_number}: {step.player}[{step.move_location}]")
            print(f"  Value: {step.value:.3f}")
            print(f"  Win%: {step.win_prob:.1%}")
            
            if step.original_comment:
                print(f"  Original SGF data: {step.original_comment}")


def example_4_full_pipeline():
    """Example 4: Full pipeline with orchestrator - generates conversation data."""
    print("=" * 70)
    print("Example 4: Full pipeline")
    print("=" * 70)

    processor = SGFSProcessor(
        katahex_path="build/katahex-win64-19-eigen.exe",
        model_path="katahex_model_20220618.bin.gz",
        config_path="processor_gtp.cfg"
    )

    processor.process_sgfs_file(
        sgfs_path='dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs',
        output_dir='example_conversations',
        max_games=2  # Process only first 2 games for demo
    )

    print("\n\nConversation data saved to: example_conversations/")
    print("Each game is saved as a separate JSONL file.")


def example_5_threaded_processing():
    """Example 5: Multi-threaded batch processing with checkpoints."""
    print("=" * 70)
    print("Example 5: Multi-threaded batch processing")
    print("=" * 70)

    processor = SGFSProcessor(
        katahex_path="build/katahex-win64-19-eigen.exe",
        model_path="katahex_model_20220618.bin.gz",
        config_path="processor_gtp.cfg"  # Use optimized config
    )

    # Process with 4 threads (limit to 10 games for demo)
    # Checkpoints will be saved in the output directory
    processor.process_sgfs_file_threaded(
        sgfs_path='dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs',
        output_dir='example_conversations_threaded',
        max_games=10,
        num_threads=4,
        checkpoint_interval=5
    )

    print("\n\nConversation data saved to: example_conversations_threaded/")
    print("Checkpoints saved at: example_conversations_threaded/checkpoint_5.jsonl, checkpoint_10.jsonl")


def example_6_conversation_generation():
    """Example 6: Generate conversation data with multi-threading."""
    print("=" * 70)
    print("Example 6: Generate conversation data with multi-threading")
    print("=" * 70)

    processor = SGFSProcessor(
        katahex_path="bin/katahex",
        model_path="katahex_model_20220618.bin.gz",
        config_path="processor_gtp.cfg"
    )

    # Process games with 4 conversation generation threads
    print("\nProcessing games with 4 conversation threads...")
    processor.process_sgfs_file(
        sgfs_path='dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs',
        output_dir='example_conversations_multi',
        max_games=10,
        num_threads=4  # Use 4 threads for conversation generation
    )

    print("\n\nConversation data saved to: example_conversations_multi/")
    print("Each game is saved as a separate JSONL file.")
    print("Conversation generation used 4 parallel threads for better performance.")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Parse SGFS file only', example_1_parse_only),
        '2': ('Evaluate single position', example_2_evaluate_position),
        '3': ('Process complete game', example_3_process_game),
        '4': ('Full pipeline', example_4_full_pipeline),
        '5': ('Multi-threaded batch processing', example_5_threaded_processing),
        '6': ('Generate conversation data', example_6_conversation_generation),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Available examples:")
        for key, (desc, _) in examples.items():
            print(f"  {key}: {desc}")
        print("\nUsage: python example_usage.py [1|2|3|4|5|6]")
        print("Or run without arguments to see this menu.")
        sys.exit(0)

    if choice in examples:
        _, func = examples[choice]
        func()
    else:
        print(f"Invalid choice: {choice}")
        print("Valid choices: 1, 2, 3, 4, 5, 6")

