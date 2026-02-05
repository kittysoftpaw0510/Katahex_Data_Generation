#!/usr/bin/env python3
"""
Module 3: Game History Processor
Processes each step of a game history and collects policy/value data.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from sgf_parser import GameRecord, Move
from nn_evaluator import KataHexEvaluator, BoardState, NNEvaluation


@dataclass
class StepData:
    """Data for a single step in the game."""
    move_number: int
    player: str
    move_location: str
    board_size: int
    black_stones: List[str]
    white_stones: List[str]
    next_player: str
    
    # Neural network evaluation
    policy: Dict[str, float]
    value: float
    win_prob: float
    loss_prob: float
    draw_prob: float
    
    # Original comment data from SGF (if available)
    original_comment: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class GameData:
    """Complete data for one game with all steps."""
    game_metadata: dict
    steps: List[StepData]

    @property
    def game_id(self) -> str:
        """Get unique game ID from metadata."""
        return self.game_metadata.get('game_id', '')

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.game_metadata,
            'steps': [step.to_dict() for step in self.steps]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameData':
        """Create GameData from dictionary (for loading checkpoints)."""
        steps = [StepData(**step_dict) for step_dict in data['steps']]
        return cls(
            game_metadata=data['metadata'],
            steps=steps
        )


class GameHistoryProcessor:
    """
    Processes a game history step by step, evaluating each position.
    """
    
    def __init__(self, evaluator: KataHexEvaluator):
        """
        Initialize the processor.
        
        Args:
            evaluator: KataHexEvaluator instance
        """
        self.evaluator = evaluator
    
    def process_game(self, game: GameRecord) -> GameData:
        """
        Process a complete game, evaluating each position.
        
        Args:
            game: GameRecord from SGF parser
            
        Returns:
            GameData with all step evaluations
        """
        steps = []
        
        # Track board state
        black_stones = list(game.initial_stones.get('AB', []))
        white_stones = list(game.initial_stones.get('AW', []))
        move_history = []
        
        # Determine starting player
        if black_stones and not white_stones:
            next_player = 'W'
        else:
            next_player = 'B'
        
        # Process each move
        for move_num, move in enumerate(game.moves, 1):
            # Create board state before this move
            board_state = BoardState(
                board_size=game.board_size,
                black_stones=black_stones.copy(),
                white_stones=white_stones.copy(),
                next_player=move.player,
                move_history=move_history.copy()
            )
            
            # Evaluate position
            evaluation = self.evaluator.evaluate(board_state)
            
            # Parse original comment if available
            original_comment = move.parse_comment()
            
            # Create step data
            step = StepData(
                move_number=move_num,
                player=move.player,
                move_location=move.location,
                board_size=game.board_size,
                black_stones=black_stones.copy(),
                white_stones=white_stones.copy(),
                next_player=move.player,
                policy=evaluation.policy,
                value=evaluation.value,
                win_prob=evaluation.win_prob,
                loss_prob=evaluation.loss_prob,
                draw_prob=evaluation.draw_prob,
                original_comment=original_comment
            )
            
            steps.append(step)
            
            # Update board state with the move
            if move.location != 'pass':
                if move.player == 'B':
                    black_stones.append(move.location)
                else:
                    white_stones.append(move.location)
            
            move_history.append((move.player, move.location))
            
            # Update next player
            next_player = 'W' if move.player == 'B' else 'B'
        
        # Create game metadata
        # Generate unique game ID from metadata
        import hashlib
        game_id_str = f"{game.black_player}_{game.white_player}_{game.result}_{len(game.moves)}"
        game_id = hashlib.md5(game_id_str.encode()).hexdigest()[:16]

        metadata = {
            'game_id': game_id,
            'board_size': game.board_size,
            'black_player': game.black_player,
            'white_player': game.white_player,
            'result': game.result,
            'rules': game.rules,
            'total_moves': len(game.moves),
            **game.metadata
        }
        
        return GameData(
            game_metadata=metadata,
            steps=steps
        )
    
    def process_games(self, games: List[GameRecord]) -> List[GameData]:
        """
        Process multiple games.
        
        Args:
            games: List of GameRecord objects
            
        Returns:
            List of GameData objects
        """
        results = []
        
        for i, game in enumerate(games, 1):
            print(f"Processing game {i}/{len(games)}...")
            try:
                game_data = self.process_game(game)
                results.append(game_data)
            except Exception as e:
                print(f"Error processing game {i}: {e}")
                continue
        
        return results

