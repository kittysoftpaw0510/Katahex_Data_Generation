#!/usr/bin/env python3
"""
Module 1: SGFS File Parser
Parses .sgfs files containing multiple SGF game records.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class Move:
    """Represents a single move in the game."""
    player: str  # 'B' or 'W'
    location: str  # e.g., 'e4', 'pass'
    comment: Optional[str] = None  # Comment data like "0.95 0.05 0.00 v=28"
    
    def parse_comment(self) -> Optional[Dict[str, float]]:
        """Parse comment to extract win/loss/draw probabilities and visit count."""
        if not self.comment:
            return None
        
        # Format: "0.95 0.05 0.00 v=28" or "0.95 0.05 0.00 v=28 result=W+"
        parts = self.comment.split()
        if len(parts) >= 3:
            try:
                return {
                    'win_prob': float(parts[0]),
                    'loss_prob': float(parts[1]),
                    'draw_prob': float(parts[2]),
                    'visits': int(parts[3].split('=')[1]) if len(parts) > 3 and 'v=' in parts[3] else None
                }
            except (ValueError, IndexError):
                return None
        return None


@dataclass
class GameRecord:
    """Represents a single game from the SGFS file."""
    board_size: int
    black_player: str
    white_player: str
    result: str
    rules: str
    moves: List[Move]
    initial_stones: Dict[str, List[str]]  # 'AB' and 'AW' for handicap stones
    metadata: Dict[str, str]  # Additional metadata from root node
    
    def get_move_count(self) -> int:
        """Return the total number of moves in the game."""
        return len(self.moves)


def parse_sgf_node(node_str: str) -> Tuple[Optional[Move], Optional[str]]:
    """
    Parse a single SGF node (move).
    Returns (Move, comment) or (None, None) if it's not a move node.
    """
    # Match move: B[loc] or W[loc]
    move_match = re.search(r'([BW])\[([^\]]*)\]', node_str)
    # Match comment: C[...]
    comment_match = re.search(r'C\[([^\]]*)\]', node_str)
    
    if move_match:
        player = move_match.group(1)
        location = move_match.group(2) if move_match.group(2) else 'pass'
        comment = comment_match.group(1) if comment_match else None
        return Move(player=player, location=location, comment=comment), comment
    
    return None, comment_match.group(1) if comment_match else None


def parse_sgf_game(sgf_str: str) -> GameRecord:
    """
    Parse a single SGF game string into a GameRecord.
    """
    # Extract root node properties
    root_match = re.search(r'\(;([^;]+)', sgf_str)
    if not root_match:
        raise ValueError("Invalid SGF format: no root node found")
    
    root_node = root_match.group(1)
    
    # Parse board size
    size_match = re.search(r'SZ\[(\d+)(?::(\d+))?\]', root_node)
    board_size = int(size_match.group(1)) if size_match else 19
    
    # Parse players
    pb_match = re.search(r'PB\[([^\]]*)\]', root_node)
    pw_match = re.search(r'PW\[([^\]]*)\]', root_node)
    black_player = pb_match.group(1) if pb_match else "Black"
    white_player = pw_match.group(1) if pw_match else "White"
    
    # Parse result
    re_match = re.search(r'RE\[([^\]]*)\]', root_node)
    result = re_match.group(1) if re_match else "?"
    
    # Parse rules
    ru_match = re.search(r'RU\[([^\]]*)\]', root_node)
    rules = ru_match.group(1) if ru_match else "unknown"
    
    # Parse initial stones (handicap)
    ab_match = re.search(r'AB\[([^\]]*)\]', root_node)
    aw_match = re.search(r'AW\[([^\]]*)\]', root_node)
    initial_stones = {
        'AB': ab_match.group(1).split('][') if ab_match else [],
        'AW': aw_match.group(1).split('][') if aw_match else []
    }
    
    # Parse metadata from root comment
    root_comment_match = re.search(r'C\[([^\]]*)\]', root_node)
    metadata = {}
    if root_comment_match:
        comment = root_comment_match.group(1)
        for item in comment.split(','):
            if '=' in item:
                key, value = item.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    # Parse moves
    moves = []
    # Find all nodes after the root (nodes start with ;)
    nodes = re.findall(r';([^;]+)', sgf_str[root_match.end():])
    
    for node_str in nodes:
        move, _ = parse_sgf_node(node_str)
        if move:
            moves.append(move)
    
    return GameRecord(
        board_size=board_size,
        black_player=black_player,
        white_player=white_player,
        result=result,
        rules=rules,
        moves=moves,
        initial_stones=initial_stones,
        metadata=metadata
    )


def parse_sgfs_file(file_path: str) -> List[GameRecord]:
    """
    Parse an SGFS file containing multiple SGF games (one per line).
    
    Args:
        file_path: Path to the .sgfs file
        
    Returns:
        List of GameRecord objects
    """
    games = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                game = parse_sgf_game(line)
                games.append(game)
            except Exception as e:
                print(f"Warning: Failed to parse game on line {line_num}: {e}")
                continue
    
    return games

