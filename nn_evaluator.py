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
from threading import Lock, Event
from queue import Queue
import time as time_module
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
                 config_path: Optional[str] = None,
                 use_mcts: bool = False,
                 gpu_id: Optional[int] = None,
                 max_visits: int = 1600):
        """
        Initialize the evaluator.

        Args:
            katahex_path: Path to KataHex executable
            model_path: Path to neural network model
            config_path: Optional path to config file
            use_mcts: If True, use kata-analyze (slow, high quality). If False, use kata-raw-nn (fast, raw NN)
            gpu_id: Optional GPU ID to use (for multi-GPU setups)
            max_visits: Maximum number of MCTS visits (only used when use_mcts=True)
        """
        self.katahex_path = katahex_path
        self.model_path = model_path
        self.config_path = config_path
        self.use_mcts = use_mcts
        self.gpu_id = gpu_id
        self.max_visits = max_visits
        self.process = None
        self._io_lock = Lock()  # For thread-safe batch operations
        self._query_counter = 0  # Unique ID for each query
        
    def start(self):
        """Start the KataHex engine."""
        # Use 'analysis' mode for MCTS, 'gtp' mode for raw NN
        if self.use_mcts:
            cmd = [self.katahex_path, "analysis"]
        else:
            cmd = [self.katahex_path, "gtp"]

        # Both config and model can be specified
        if self.config_path:
            cmd.extend(["-config", self.config_path])
        if self.model_path:
            cmd.extend(["-model", self.model_path])

        # GPU selection via environment variable (most reliable method)
        # CUDA_VISIBLE_DEVICES forces the process to only see specific GPU(s)
        env = os.environ.copy()
        if self.gpu_id is not None:
            # Set CUDA_VISIBLE_DEVICES to make only this GPU visible
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # Use universal_newlines=True for text mode
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            env=env  # Pass modified environment with CUDA_VISIBLE_DEVICES
        )

        # Check if process started
        import time
        time.sleep(0.5)

        if self.process.poll() is not None:
            # Process has already terminated
            stderr_output = self.process.stderr.read()
            raise RuntimeError(f"KataHex process terminated immediately. Error: {stderr_output}")
        
    def stop(self):
        """Stop the KataHex engine."""
        if self.process:
            try:
                if self.use_mcts:
                    # Analysis mode: close stdin to signal end of input
                    self.process.stdin.close()
                else:
                    # GTP mode: send quit command
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
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
            # CRITICAL: Lock I/O to prevent race conditions from multiple threads
            with self._io_lock:
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

    def _send_command_analyze(self, command: str, timeout: float = 30.0) -> str:
        """
        Send kata-analyze command and get response.
        kata-analyze is a continuous command that:
        1. Prints "=" immediately
        2. Starts analysis and outputs "info" lines periodically
        3. Continues until stopped by sending an empty line

        We'll wait for analysis to complete (based on maxVisits from config),
        collect the info lines, then stop the analysis.
        """
        if not self.process:
            raise RuntimeError("Engine not started. Call start() first.")

        try:
            import time

            # Send command
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

            # Read response
            response_lines = []
            start_time = time.time()
            got_first_info = False

            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    # Stop analysis by sending empty line
                    self.process.stdin.write("\n")
                    self.process.stdin.flush()
                    time.sleep(0.2)  # Give it time to stop
                    break

                # Read line (this will block until data is available)
                line = self.process.stdout.readline()
                if not line:  # EOF
                    break

                line = line.strip()

                # First line should be "="
                if line.startswith('='):
                    continue

                # Empty line means analysis stopped
                if not line:
                    if response_lines:  # We have data, we're done
                        break
                    continue

                # Check if this is an error
                if line.startswith('?'):
                    error_msg = line[1:].strip()
                    raise RuntimeError(f"GTP command failed: {command}\nResponse: {error_msg}")

                # Collect info lines
                if line.startswith('info'):
                    response_lines.append(line)
                    if not got_first_info:
                        got_first_info = True

                    # After we get enough info lines, wait a bit more then stop
                    # This ensures we get the final analysis after search completes
                    if len(response_lines) >= 10 and elapsed > 5.0:
                        # Stop analysis by sending empty line
                        self.process.stdin.write("\n")
                        self.process.stdin.flush()
                        # Continue reading until we get the empty line
                        continue

            return '\n'.join(response_lines)

        except Exception as e:
            # Make sure to stop analysis on error
            try:
                self.process.stdin.write("\n")
                self.process.stdin.flush()
            except:
                pass
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
        Evaluate a board position using neural network.

        Args:
            board_state: The board state to evaluate

        Returns:
            NNEvaluation with policy and value
        """
        if self.use_mcts:
            # Analysis mode: send position via JSON, no GTP commands needed
            return self._evaluate_with_mcts(board_state)
        else:
            # GTP mode: set up position using GTP commands, then evaluate
            self.set_position(board_state)
            return self._evaluate_raw_nn(board_state)

    def _evaluate_raw_nn(self, board_state: BoardState) -> NNEvaluation:
        """Fast evaluation using raw NN (no MCTS)."""
        # Get raw neural network output using kata-raw-nn command
        # Symmetry 0 means no symmetry transformation
        response = self._send_command("kata-raw-nn 0")

        # Parse the response
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
                if len(parts) > 1 and parts[1] != 'NAN':
                    policy['pass'] = float(parts[1])
                in_policy_section = False
            elif in_policy_section:
                # This is a row of the policy grid
                for col, prob_str in enumerate(parts):
                    if prob_str != 'NAN':
                        prob = float(prob_str)
                        col_letter = chr(ord('a') + col)
                        row_number = policy_row + 1
                        location = f"{col_letter}{row_number}"
                        policy[location] = prob
                policy_row += 1

        # Calculate value from current player's perspective
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

    def evaluate_batch(self, board_states: List[BoardState]) -> List[NNEvaluation]:
        """
        Evaluate multiple positions in a batch for maximum GPU utilization.
        This is the PROPER way to use KataHex with multiple threads.
        """
        import json

        if not board_states:
            return []

        # Use MCTS or raw NN based on configuration
        if self.use_mcts:
            return self._evaluate_batch_mcts(board_states)
        else:
            return self._evaluate_batch_raw_nn(board_states)

    def _evaluate_batch_mcts(self, board_states: List[BoardState]) -> List[NNEvaluation]:
        """Batch evaluation using MCTS via analysis mode."""
        import json

        # Build queries for all positions
        queries = []
        for idx, board_state in enumerate(board_states):
            moves = []
            for move in board_state.move_history:
                player, location = move
                player_str = "B" if player == 'B' else "W"
                moves.append([player_str, location])

            query = {
                "id": f"batch_{idx}",  # Unique ID for each query
                "moves": moves,
                "rules": "tromp-taylor",
                "boardXSize": board_state.board_size,
                "boardYSize": board_state.board_size,
                "analyzeTurns": [len(moves)],
                "maxVisits": self.max_visits,
                "includePolicy": True,
            }
            queries.append(query)

        # Send all queries at once, then read all responses
        with self._io_lock:
            # Send all queries (non-blocking writes)
            for query in queries:
                self.process.stdin.write(json.dumps(query) + "\n")
            self.process.stdin.flush()

            # Read all responses
            responses = {}
            for _ in range(len(queries)):
                response_line = self.process.stdout.readline()
                if not response_line:
                    raise RuntimeError("No response from KataHex")
                response = json.loads(response_line)
                responses[response["id"]] = response

        # Parse responses in order
        results = []
        for idx, board_state in enumerate(board_states):
            response = responses[f"batch_{idx}"]
            result = self._parse_analysis_response(response, board_state)
            results.append(result)

        return results

    def _evaluate_batch_raw_nn(self, board_states: List[BoardState]) -> List[NNEvaluation]:
        """Batch evaluation using raw NN via GTP mode."""
        # For raw NN, we still need to evaluate one by one in GTP mode
        # But we can send multiple commands before reading responses
        results = []
        for board_state in board_states:
            result = self._evaluate_with_raw_nn(board_state)
            results.append(result)
        return results

    def _evaluate_with_mcts(self, board_state: BoardState) -> NNEvaluation:
        """High-quality evaluation using MCTS search (slow) via JSON analysis mode."""
        # Use batch evaluation with single item for consistency
        return self.evaluate_batch([board_state])[0]

    def _parse_analysis_response(self, response: dict, board_state: BoardState) -> NNEvaluation:
        """Parse KataHex analysis response into NNEvaluation."""
        # Extract policy and value from response
        policy = {}
        win_prob = 0.5
        loss_prob = 0.5
        draw_prob = 0.0

        # Extract policy from moveInfos
        if 'moveInfos' in response and response['moveInfos']:
            for move_info in response['moveInfos']:
                move_str = move_info['move']
                prior = move_info.get('prior', 0.0)
                policy[move_str] = prior

        # Extract value from rootInfo
        if 'rootInfo' in response:
            root_info = response['rootInfo']
            current_player_winrate = root_info.get('winrate', 0.5)

            # KataHex returns winrate from the perspective of the player to move
            # board_state.next_player is the player about to move (current player)
            # We store win_prob/loss_prob from White's perspective for consistency

            if board_state.next_player == 'B':
                # Black to move, so current_player_winrate is Black's winrate
                win_prob = 1.0 - current_player_winrate  # White's win prob
                loss_prob = current_player_winrate        # White's loss prob (= Black's win prob)
            else:
                # White to move, so current_player_winrate is White's winrate
                win_prob = current_player_winrate
                loss_prob = 1.0 - current_player_winrate

        # Calculate value from the perspective of the player to move
        # value should be positive if the current player is winning
        if board_state.next_player == 'B':
            # Black to move: value = Black's advantage = -White's advantage
            value = loss_prob - win_prob
        else:
            # White to move: value = White's advantage
            value = win_prob - loss_prob

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

