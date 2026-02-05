#!/usr/bin/env python3
"""
Conversation Generator Module
Generates conversation lists for model training from game history.

Input: GameData (from game_processor.py)
Output: List of conversation dicts for each step

Uses OpenSpiel for exact compatibility with generate_hex_rollout_dataset.py
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add openspiel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openspiel'))

import pyspiel
from agents import GAME_AGENTS
from game_processor import GameData


def sgf_coord_to_action(coord: str, board_size: int) -> int:
    """
    Convert SGF coordinate (e.g., "a1", "k11") to OpenSpiel action index.

    SGF Hex coordinates: column (a-k) + row (1-11)
    OpenSpiel hex action: row * board_size + col
    """
    col_char = coord[0].lower()
    row_str = coord[1:]

    col = ord(col_char) - ord('a')
    row = int(row_str) - 1

    # OpenSpiel hex uses row * board_size + col
    action = row * board_size + col
    return action


def action_to_sgf_coord(action: int, board_size: int) -> str:
    """Convert OpenSpiel action index to SGF coordinate."""
    row = action // board_size
    col = action % board_size

    col_char = chr(ord('a') + col)
    row_num = row + 1

    return f"{col_char}{row_num}"


def generate_conversation_from_game(
    game_data: GameData,
    include_policy: bool = False,
    include_value: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate conversation list from a complete game using OpenSpiel.

    Args:
        game_data: GameData object with steps
        include_policy: Include top policy moves in metadata
        include_value: Include value estimate in metadata

    Returns:
        List of conversation entries (one per step)
    """
    if not game_data.steps:
        return []

    board_size = game_data.steps[0].board_size

    # Create OpenSpiel game and agent
    game = pyspiel.load_game("hex", {"board_size": board_size})
    agent = GAME_AGENTS['hex']()

    # Initialize OpenSpiel state
    state = game.new_initial_state()

    # Track conversation history for each player
    player_conversations = {
        0: [],  # Black
        1: []   # White
    }

    # Generate system prompt once per player
    system_prompt = agent.generate_system_prompt()
    for player_id in [0, 1]:
        player_conversations[player_id].append({"role": "system", "content": system_prompt})

    conversations = []

    # Replay moves and generate conversations
    for step_idx, step in enumerate(game_data.steps):
        current_player = state.current_player()

        # Get legal actions from OpenSpiel
        legal_actions = state.legal_actions(current_player)

        # Generate user prompt using OpenSpiel agent
        user_prompt = agent.generate_user_prompt(
            state=state,
            player_id=current_player,
            legal_actions=legal_actions
        )

        # Add to conversation
        player_conversations[current_player].append({"role": "user", "content": user_prompt})

        # Convert move to action
        try:
            action = sgf_coord_to_action(step.move_location, board_size)
        except Exception as e:
            print(f"Warning: Failed to convert move {step.move_location}: {e}")
            break

        # Validate action is legal
        if action not in legal_actions:
            print(f"Warning: Action {action} ({step.move_location}) not legal at step {step_idx + 1}")
            break

        # Assistant response (the action number, matching OpenSpiel format)
        player_conversations[current_player].append({"role": "assistant", "content": str(action)})

        # Build conversation entry
        entry = {
            "move_number": step.move_number,
            "player": step.player,
            "move": step.move_location,
            "action": action,
            "conversation": list(player_conversations[current_player]),  # Copy current state
        }

        if include_policy:
            # Top 5 policy moves - convert SGF coords to action numbers
            # sorted_policy = sorted(step.policy.items(), key=lambda x: x[1], reverse=True)[:5]
            sorted_policy = sorted(step.policy.items(), key=lambda x: x[1], reverse=True)
            policy_actions = {}
            for coord, prob in sorted_policy:
                try:
                    action_num = sgf_coord_to_action(coord, board_size)
                    policy_actions[action_num] = prob
                except:
                    pass  # Skip invalid coordinates
            entry["top_policy"] = policy_actions

        if include_value:
            # Value and win_prob should be from current player's perspective
            # step.value is already from the player's perspective who is about to move
            entry["value"] = step.value

            # step.win_prob and step.loss_prob are from White's perspective
            # Convert to current player's perspective
            if step.player == 'B':
                # Black's win probability = White's loss probability
                entry["win_prob"] = step.loss_prob
            else:
                # White's win probability = White's win probability
                entry["win_prob"] = step.win_prob

        conversations.append(entry)

        # Apply action to state
        state.apply_action(action)

    return conversations


def generate_training_trajectory(
    game_data: GameData,
    player_filter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate training trajectory format (matching generate_hex_rollout_dataset.py).
    Uses OpenSpiel for exact compatibility.

    Args:
        game_data: GameData object with steps
        player_filter: 0 for black only, 1 for white only, None for both

    Returns:
        List of trajectory dicts (one per player, filtered as requested)
    """
    if not game_data.steps:
        return []

    board_size = game_data.steps[0].board_size
    metadata = game_data.game_metadata

    # Determine winner from result
    result = metadata.get('result', '?')
    winner_char = 'B' if 'B+' in result else ('W' if 'W+' in result else None)

    # Create OpenSpiel game and agent
    game = pyspiel.load_game("hex", {"board_size": board_size})
    agent = GAME_AGENTS['hex']()

    # Initialize OpenSpiel state
    state = game.new_initial_state()

    # Track conversation history for each player
    player_conversations = {
        0: [],  # Black
        1: []   # White
    }

    # Generate system prompt once per player
    system_prompt = agent.generate_system_prompt()
    for player_id in [0, 1]:
        player_conversations[player_id].append({"role": "system", "content": system_prompt})

    # Replay moves and build conversations
    for step_idx, step in enumerate(game_data.steps):
        current_player = state.current_player()

        # Get legal actions from OpenSpiel
        legal_actions = state.legal_actions(current_player)

        # Generate user prompt using OpenSpiel agent
        user_prompt = agent.generate_user_prompt(
            state=state,
            player_id=current_player,
            legal_actions=legal_actions
        )

        # Add to conversation
        player_conversations[current_player].append({"role": "user", "content": user_prompt})

        # Convert move to action
        try:
            action = sgf_coord_to_action(step.move_location, board_size)
        except Exception as e:
            print(f"Warning: Failed to convert move {step.move_location}: {e}")
            break

        # Validate action is legal
        if action not in legal_actions:
            print(f"Warning: Action {action} ({step.move_location}) not legal at step {step_idx + 1}")
            break

        # Assistant response (action number)
        player_conversations[current_player].append({"role": "assistant", "content": str(action)})

        # Apply action to state
        state.apply_action(action)

    # Get final returns from OpenSpiel
    if state.is_terminal():
        returns = state.returns()
    else:
        # Game not finished, use result from metadata
        if winner_char == 'B':
            returns = [1.0, -1.0]
        elif winner_char == 'W':
            returns = [-1.0, 1.0]
        else:
            returns = [0.0, 0.0]

    # Build trajectories
    trajectories = []
    for player_id in [0, 1]:
        if player_filter is not None and player_id != player_filter:
            continue

        player_return = returns[player_id]

        # Normalize score: 1.0 for win, 0.0 for loss, 0.5 for draw
        if player_return > 0:
            score = 1.0
        elif player_return < 0:
            score = 0.0
        else:
            score = 0.5

        trajectory = {
            'conversation': player_conversations[player_id],
            'score': score,
            'success': score > 0.5,
            'task_id': 600_000_000 + (board_size - 5) // 2,
            'game_name': 'hex',
            'player_id': player_id,
            'returns': list(returns),
            'board_size': board_size,
            'game_id': game_data.game_id,
            'result': result,
        }
        trajectories.append(trajectory)

    return trajectories

