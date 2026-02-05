#!/usr/bin/env python3
"""
Module 2: Neural Network Evaluator
Interfaces with KataHex C++ engine to get policy and value for board positions.
"""

import subprocess
import json
import tempfile
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class BoardState:
    """Represents a board state."""
    board_size: int
    black_stones: List[str]  # List of positions like ['e4', 'f5']
    white_stones: List[str]
    next_player: str  # 'B' or 'W'
    move_history: List[Tuple[str, str]]  # [(player, location), ...]


@dataclass
class NNEvaluation:
    """Neural network evaluation result."""
    policy: Dict[str, float]  # {location: probability}
    value: float  # Win probability for current player (-1 to 1)
    win_prob: float  # White win probability
    loss_prob: float  # White loss probability
    draw_prob: float  # Draw probability


class KataHexEvaluator:
    """
    Interface to KataHex neural network evaluation via GTP protocol.
    """
    
    def __init__(self, 
                 katahex_path: str = "build/katahex-win64-19-eigen.exe",
                 model_path: str = "katahex_model_20220618.bin.gz",
                 config_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            katahex_path: Path to KataHex executable
            model_path: Path to neural network model
            config_path: Optional path to config file
        """
        self.katahex_path = katahex_path
        self.model_path = model_path
        self.config_path = config_path
        self.process = None
        
    def start(self):
        """Start the KataHex GTP engine."""
        cmd = [self.katahex_path, "gtp"]

        # Both config and model can be specified
        if self.config_path:
            cmd.extend(["-config", self.config_path])
        if self.model_path:
            cmd.extend(["-model", self.model_path])

        # Use universal_newlines=True for text mode
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8'
        )

        # Check if process started
        import time
        time.sleep(1.0)

        if self.process.poll() is not None:
            # Process has already terminated
            stderr_output = self.process.stderr.read()
            raise RuntimeError(f"KataHex process terminated immediately. Error: {stderr_output}")

        # Test connection
        try:
            response = self._send_command("name")
        except Exception as e:
            raise RuntimeError(f"Could not connect to KataHex engine: {e}")
        
    def stop(self):
        """Stop the KataHex engine."""
        if self.process:
            try:
                self._send_command("quit")
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            finally:
                self.process = None
    
    def _send_command(self, command: str) -> str:
        """Send a GTP command and get response."""
        if not self.process:
            raise RuntimeError("Engine not started. Call start() first.")

        try:
            # Send command
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

            # Read response until we get a blank line
            response_lines = []
            success = False
            error = False

            while True:
                line = self.process.stdout.readline()
                if not line:  # EOF
                    break

                line = line.strip()

                if not line:  # Empty line marks end of response
                    if success or error:
                        break
                    continue

                if line.startswith('='):
                    success = True
                    # Remove the '=' and any ID
                    content = line[1:].strip()
                    if content:
                        response_lines.append(content)
                elif line.startswith('?'):
                    error = True
                    # Remove the '?' and any ID
                    content = line[1:].strip()
                    if content:
                        response_lines.append(content)
                else:
                    response_lines.append(line)

            if error:
                raise RuntimeError(f"GTP command failed: {command}\nResponse: {' '.join(response_lines)}")

            return '\n'.join(response_lines)

        except Exception as e:
            raise RuntimeError(f"Error sending command '{command}': {e}")
    
    def set_board_size(self, size: int):
        """Set the board size."""
        self._send_command(f"boardsize {size}")
    
    def clear_board(self):
        """Clear the board."""
        self._send_command("clear_board")
    
    def play_move(self, player: str, location: str):
        """
        Play a move on the board.
        
        Args:
            player: 'B' or 'W'
            location: Move location like 'e4' or 'pass'
        """
        self._send_command(f"play {player} {location}")
    
    def set_position(self, board_state: BoardState):
        """
        Set up a board position.
        
        Args:
            board_state: The board state to set up
        """
        self.set_board_size(board_state.board_size)
        self.clear_board()
        
        # Play all moves from history
        for player, location in board_state.move_history:
            self.play_move(player, location)
    
    def evaluate(self, board_state: BoardState) -> NNEvaluation:
        """
        Evaluate a board position using the neural network.

        Args:
            board_state: The board state to evaluate

        Returns:
            NNEvaluation with policy and value
        """
        # Set up the position
        self.set_position(board_state)

        # Get raw neural network output using kata-raw-nn command
        # Symmetry 0 means no symmetry transformation
        response = self._send_command("kata-raw-nn 0")

        # Parse the response
        # Expected format from kata-raw-nn:
        # symmetry 0
        # whiteWin 0.520000
        # whiteLoss 0.480000
        # noResult 0.000000
        # varTimeLeft 12.345
        # shorttermWinlossError 0.123
        # policy
        # <grid of probabilities, one row per board row>
        # policyPass <probability>

        policy = {}
        win_prob = 0.5
        loss_prob = 0.5
        draw_prob = 0.0

        lines = response.strip().split('\n')
        in_policy_section = False
        policy_row = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 1:
                continue

            if parts[0] == 'whiteWin':
                win_prob = float(parts[1])
            elif parts[0] == 'whiteLoss':
                loss_prob = float(parts[1])
            elif parts[0] == 'noResult':
                draw_prob = float(parts[1])
            elif parts[0] == 'policy':
                in_policy_section = True
                policy_row = 0
            elif parts[0] == 'policyPass':
                # Pass move probability
                if len(parts) > 1 and parts[1] != 'NAN':
                    policy['pass'] = float(parts[1])
                in_policy_section = False
            elif in_policy_section:
                # This is a row of the policy grid
                # Convert grid position to Hex coordinates
                for col, prob_str in enumerate(parts):
                    if prob_str != 'NAN':
                        prob = float(prob_str)
                        # Convert (col, row) to Hex notation (e.g., 'e4')
                        # Columns are labeled a, b, c, ...
                        # Rows are labeled 1, 2, 3, ...
                        col_letter = chr(ord('a') + col)
                        row_number = policy_row + 1
                        location = f"{col_letter}{row_number}"
                        policy[location] = prob
                policy_row += 1

        # Calculate value from current player's perspective
        # KataHex outputs are from White's perspective:
        #   whiteWin = probability White wins
        #   whiteLoss = probability White loses (Black wins)
        # If next player is Black: value = whiteLoss - whiteWin (positive when Black winning)
        # If next player is White: value = whiteWin - whiteLoss (positive when White winning)
        if board_state.next_player == 'B':
            value = loss_prob - win_prob  # Black's perspective
        else:
            value = win_prob - loss_prob  # White's perspective

        return NNEvaluation(
            policy=policy,
            value=value,
            win_prob=win_prob,
            loss_prob=loss_prob,
            draw_prob=draw_prob
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

