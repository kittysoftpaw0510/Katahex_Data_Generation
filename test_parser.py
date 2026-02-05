#!/usr/bin/env python3
"""
Simple test script to verify the SGF parser works correctly.
"""

from sgf_parser import parse_sgfs_file
import sys


def test_parse_sgfs():
    """Test parsing an SGFS file."""
    
    # Use the example file from the dataset
    sgfs_file = 'dataset_100000/sgfs_20260204_043749/0B0256BE75A3F9A4.sgfs'
    
    print(f"Testing SGFS parser with: {sgfs_file}")
    print("=" * 70)
    
    try:
        games = parse_sgfs_file(sgfs_file)
        print(f"\n✓ Successfully parsed {len(games)} games\n")
        
        # Show details of first 3 games
        for i, game in enumerate(games[:3], 1):
            print(f"Game {i}:")
            print(f"  Board size: {game.board_size}x{game.board_size}")
            print(f"  Players: {game.black_player} (Black) vs {game.white_player} (White)")
            print(f"  Result: {game.result}")
            print(f"  Rules: {game.rules}")
            print(f"  Total moves: {game.get_move_count()}")
            
            # Show metadata
            if game.metadata:
                print(f"  Metadata:")
                for key, value in game.metadata.items():
                    print(f"    {key}: {value}")
            
            # Show initial stones if any
            if game.initial_stones['AB'] or game.initial_stones['AW']:
                print(f"  Initial stones:")
                if game.initial_stones['AB']:
                    print(f"    Black: {game.initial_stones['AB']}")
                if game.initial_stones['AW']:
                    print(f"    White: {game.initial_stones['AW']}")
            
            # Show first 5 moves
            print(f"  First 5 moves:")
            for j, move in enumerate(game.moves[:5], 1):
                comment_data = move.parse_comment()
                if comment_data:
                    print(f"    {j}. {move.player}[{move.location}] - "
                          f"Win: {comment_data['win_prob']:.2f}, "
                          f"Loss: {comment_data['loss_prob']:.2f}, "
                          f"Visits: {comment_data['visits']}")
                else:
                    print(f"    {j}. {move.player}[{move.location}]")
            
            print()
        
        # Statistics
        print("=" * 70)
        print("Statistics:")
        print(f"  Total games: {len(games)}")
        
        board_sizes = {}
        for game in games:
            size = game.board_size
            board_sizes[size] = board_sizes.get(size, 0) + 1
        
        print(f"  Board sizes:")
        for size, count in sorted(board_sizes.items()):
            print(f"    {size}x{size}: {count} games")
        
        total_moves = sum(game.get_move_count() for game in games)
        avg_moves = total_moves / len(games) if games else 0
        print(f"  Total moves: {total_moves}")
        print(f"  Average moves per game: {avg_moves:.1f}")
        
        # Count results
        results = {}
        for game in games:
            result = game.result
            results[result] = results.get(result, 0) + 1
        
        print(f"  Results:")
        for result, count in sorted(results.items()):
            print(f"    {result}: {count} games")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_parse_sgfs()
    sys.exit(0 if success else 1)

