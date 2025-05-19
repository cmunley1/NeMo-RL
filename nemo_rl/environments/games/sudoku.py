# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class SudokuConfig(TypedDict):
    difficulty: str  # 'easy', 'medium', 'hard'


class SudokuMetadata(TypedDict):
    game_state: Dict[str, Any]
    num_moves: int
    max_moves: int


class SudokuGameLogic:
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new Sudoku puzzle."""
        difficulty = config.get("difficulty", "medium")
        
        # Create a solved Sudoku
        solution = SudokuGameLogic._generate_solution()
        
        # Create a puzzle by removing numbers from the solution
        if difficulty == "easy":
            cells_to_remove = random.randint(35, 45)
        elif difficulty == "medium":
            cells_to_remove = random.randint(46, 55)
        elif difficulty == "hard":
            cells_to_remove = random.randint(56, 64)
        else:
            cells_to_remove = 45  # Default to medium
        
        # Copy the solution and remove cells to create the puzzle
        puzzle = copy.deepcopy(solution)
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        
        for i, j in cells[:cells_to_remove]:
            puzzle[i][j] = 0
            
        original_puzzle = copy.deepcopy(puzzle)
        
        # Create and return the game state
        return {
            "difficulty": difficulty,
            "puzzle": puzzle,
            "original_puzzle": original_puzzle,  # Keep track of the initial puzzle
            "solution": solution,
            "selected_cell": None,
            "game_over": False,
            "is_complete": False,
            "is_valid": True,
            "commands": {
                "select r c": "Select cell at row r, column c (1-9)",
                "place n": "Place number n (1-9) in selected cell",
                "clear": "Clear selected cell",
                "validate": "Check if the current state is valid",
                "view": "View the current state of the board",
            },
        }

    @staticmethod
    def _generate_solution() -> List[List[int]]:
        """Generate a solved Sudoku board."""
        # Start with an empty 9x9 grid
        grid = [[0 for _ in range(9)] for _ in range(9)]
        
        # Use a backtracking algorithm to fill the grid
        def is_valid(row, col, num):
            # Check row
            for x in range(9):
                if grid[row][x] == num:
                    return False
            
            # Check column
            for x in range(9):
                if grid[x][col] == num:
                    return False
            
            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if grid[i][j] == num:
                        return False
            
            return True
        
        def solve(row=0, col=0):
            # If we've filled the whole grid, we're done
            if row == 9:
                return True
            
            # If we've filled this column, move to the next row
            if col == 9:
                return solve(row + 1, 0)
            
            # If this cell is already filled, move to the next cell
            if grid[row][col] != 0:
                return solve(row, col + 1)
            
            # Try different numbers in this cell
            nums = list(range(1, 10))
            random.shuffle(nums)
            for num in nums:
                if is_valid(row, col, num):
                    grid[row][col] = num
                    if solve(row, col + 1):
                        return True
                    grid[row][col] = 0  # Backtrack
            
            return False
        
        # Generate a solution
        solve()
        return grid

    @staticmethod
    def _is_valid_sudoku(puzzle: List[List[int]]) -> bool:
        """Check if a Sudoku puzzle is valid (no conflicts)."""
        # Check each row
        for row in puzzle:
            # Count occurrences of each number
            counts = [0] * 10
            for num in row:
                if num != 0:  # Skip empty cells
                    counts[num] += 1
                    if counts[num] > 1:
                        return False
        
        # Check each column
        for col in range(9):
            counts = [0] * 10
            for row in range(9):
                num = puzzle[row][col]
                if num != 0:
                    counts[num] += 1
                    if counts[num] > 1:
                        return False
        
        # Check each 3x3 box
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                counts = [0] * 10
                for i in range(box_row, box_row + 3):
                    for j in range(box_col, box_col + 3):
                        num = puzzle[i][j]
                        if num != 0:
                            counts[num] += 1
                            if counts[num] > 1:
                                return False
        
        return True

    @staticmethod
    def _is_complete(puzzle: List[List[int]]) -> bool:
        """Check if a Sudoku puzzle is complete (no empty cells)."""
        for row in puzzle:
            if 0 in row:
                return False
        return True

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """Initialize Sudoku game and return welcome message."""
        return (
            f"\n===== SUDOKU =====\n"
            f"Fill the 9x9 grid so that each row, column, and 3x3 box contains all digits from 1 to 9.\n"
            f"- Difficulty: {game_state['difficulty']}\n"
            f"- Use 'select r c' to select a cell at row r, column c (1-9)\n"
            f"- Use 'place n' to place digit n (1-9) in the selected cell\n"
            f"- Use 'clear' to clear the selected cell\n"
            f"- Use 'validate' to check if your current solution is valid\n"
            f"- Use 'view' to see the current state of the board"
        )

    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the Sudoku game."""
        # Default return values
        response = "Unknown command. Use 'select r c', 'place n', 'clear', 'validate', or 'view'."
        reward = 0.0
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)
        
        # Check if game is already over
        if new_state["game_over"]:
            return "Game is already over.", 0.0, True, new_state
            
        # Parse action
        if action.startswith("select "):
            try:
                _, row, col = action.split()
                row, col = int(row) - 1, int(col) - 1  # Convert to 0-indexed
                
                # Validate input
                if not (0 <= row < 9 and 0 <= col < 9):
                    return "Invalid position. Row/column must be between 1 and 9.", reward, is_terminated, new_state
                
                # Check if this is an original cell that shouldn't be modified
                if new_state["original_puzzle"][row][col] != 0:
                    return f"Cell ({row+1},{col+1}) is part of the original puzzle and cannot be modified.", reward, is_terminated, new_state
                
                new_state["selected_cell"] = (row, col)
                response = f"Selected cell ({row+1},{col+1})."
                
            except ValueError:
                return "Invalid input format. Use: select row col", reward, is_terminated, new_state
                
        elif action.startswith("place "):
            try:
                _, num = action.split()
                num = int(num)
                
                # Validate input
                if not (1 <= num <= 9):
                    return "Invalid number. Must be between 1 and 9.", reward, is_terminated, new_state
                
                # Check if a cell is selected
                if new_state["selected_cell"] is None:
                    return "No cell selected. Use 'select r c' first.", reward, is_terminated, new_state
                
                row, col = new_state["selected_cell"]
                
                # Place the number
                old_value = new_state["puzzle"][row][col]
                new_state["puzzle"][row][col] = num
                
                # Check if the move is valid
                is_valid = SudokuGameLogic._is_valid_sudoku(new_state["puzzle"])
                new_state["is_valid"] = is_valid
                
                if is_valid:
                    response = f"Placed {num} in cell ({row+1},{col+1})."
                    # Small positive reward for valid moves
                    reward = 0.1
                    
                    # Check if correct according to solution
                    if num == new_state["solution"][row][col]:
                        response += " Correct placement!"
                        reward = 0.5  # Larger reward for correct placement
                    else:
                        response += " Valid but may not be correct."
                        reward = -0.1  # Small penalty for incorrect placement
                else:
                    response = f"Placed {num} in cell ({row+1},{col+1}), but the board is now invalid."
                    reward = -0.2  # Penalty for invalid move
                
                # Check if the puzzle is complete
                is_complete = SudokuGameLogic._is_complete(new_state["puzzle"])
                new_state["is_complete"] = is_complete
                
                if is_complete and is_valid:
                    # Check if the solution is correct
                    if new_state["puzzle"] == new_state["solution"]:
                        response = "Congratulations! You've solved the Sudoku puzzle correctly!"
                        reward = 10.0  # Large reward for solving
                    else:
                        response = "The puzzle is complete but the solution is incorrect."
                        reward = -1.0  # Penalty for incorrect solution
                    
                    new_state["game_over"] = True
                    is_terminated = True
                
            except ValueError:
                return "Invalid input format. Use: place number", reward, is_terminated, new_state
                
        elif action == "clear":
            # Check if a cell is selected
            if new_state["selected_cell"] is None:
                return "No cell selected. Use 'select r c' first.", reward, is_terminated, new_state
                
            row, col = new_state["selected_cell"]
            
            # Check if this is an original cell
            if new_state["original_puzzle"][row][col] != 0:
                return f"Cell ({row+1},{col+1}) is part of the original puzzle and cannot be modified.", reward, is_terminated, new_state
                
            # Clear the cell
            new_state["puzzle"][row][col] = 0
            response = f"Cleared cell ({row+1},{col+1})."
            
            # Check if the board is now valid
            new_state["is_valid"] = SudokuGameLogic._is_valid_sudoku(new_state["puzzle"])
            
        elif action == "validate":
            is_valid = SudokuGameLogic._is_valid_sudoku(new_state["puzzle"])
            new_state["is_valid"] = is_valid
            
            if is_valid:
                response = "The current state is valid."
                reward = 0.1  # Small reward for checking
            else:
                response = "The current state is invalid. Check for conflicts in rows, columns, or boxes."
                reward = -0.1  # Small penalty
                
        return response, reward, is_terminated, new_state

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """Render the current Sudoku game state."""
        puzzle = game_state["puzzle"]
        selected_cell = game_state["selected_cell"]
        is_valid = game_state["is_valid"]
        
        # Create the output string
        output = ["\n===== SUDOKU =====\n"]
        
        # Add status indicators
        output.append(f"Difficulty: {game_state['difficulty']}")
        output.append(f"Board state: {'Valid' if is_valid else 'Invalid'}")
        if selected_cell:
            output.append(f"Selected cell: ({selected_cell[0]+1},{selected_cell[1]+1})")
        else:
            output.append("No cell selected")
        
        output.append("\n  " + " ".join(str(i+1) for i in range(9)))  # Column labels
        output.append("  " + "-" * 19)  # Top border
        
        # Draw the grid
        for i, row in enumerate(puzzle):
            row_str = f"{i+1}|"  # Row label
            for j, cell in enumerate(row):
                # Add vertical separators for 3x3 boxes
                if j > 0 and j % 3 == 0:
                    row_str += "|"
                
                # Highlight selected cell and original cells
                if selected_cell and (i, j) == selected_cell:
                    row_str += f"[{cell if cell else ' '}]"
                elif game_state["original_puzzle"][i][j] != 0:
                    row_str += f" {cell} "  # Original cells (given)
                else:
                    row_str += f" {cell if cell else '.'} "  # Empty cells as dots
            
            output.append(row_str)
            
            # Add horizontal separators for 3x3 boxes
            if i < 8 and (i + 1) % 3 == 0:
                output.append("  " + "-" * 19)
                
        # Game status
        if game_state["game_over"]:
            if game_state["is_complete"] and game_state["puzzle"] == game_state["solution"]:
                output.append("\nPuzzle solved correctly!")
            else:
                output.append("\nGame over!")
                
        return "\n".join(output)


class SudokuRunner:
    def __init__(self):
        pass  # No initialization needed as game methods are static

    def _parse_action(self, text: str) -> Optional[str]:
        """Parses the action from '<action></action>'."""
        prefix = "<action>"
        suffix = "</action>"
        # Find the prefix, case-insensitive, and potentially after some thought process
        text_lower = text.lower()
        prefix_lower = prefix.lower()
        suffix_lower = suffix.lower()

        start_idx = text_lower.rfind(prefix_lower)  # Find the last occurrence

        if start_idx != -1:
            # Find the end tag after the start tag
            end_idx = text_lower.find(suffix_lower, start_idx + len(prefix_lower))
            if end_idx != -1:
                # Extract content between tags
                action_content = text[start_idx + len(prefix) : end_idx].strip()
                return action_content
        return None

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: SudokuMetadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[SudokuMetadata],
    ]:
        """Processes a single turn for the Sudoku task."""
        game_state = metadata["game_state"]
        current_moves = metadata["num_moves"]
        max_moves = metadata["max_moves"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</action>"]
        next_metadata = metadata.copy()
        next_observation_content = ""

        # Check if max moves reached
        if current_moves >= max_moves:
            is_terminated = True
            next_observation_content = (
                f"<error>Maximum moves ({max_moves}) reached.</error>"
            )
            next_metadata = None
            return (
                {"role": "environment", "content": next_observation_content},
                0.0,
                is_terminated,
                None,
                next_metadata,
            )

        # Get last assistant message and parse action
        last_assistant_msg_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_assistant_msg_content = message_log[-1]["content"].strip()

        parsed_action = self._parse_action(last_assistant_msg_content)

        if parsed_action is None:
            rendered_game = SudokuGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_game = SudokuGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nViewing the board. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = (
                SudokuGameLogic.step(parsed_action, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            # Render the game state after the action
            rendered_game = SudokuGameLogic.render(next_game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\n{step_response}\n</environment>"

            if is_terminated:
                next_metadata = None  # Clear metadata on termination

        return (
            {"role": "environment", "content": next_observation_content + "\n"},
            turn_reward,
            is_terminated,
            next_stop_strings,
            next_metadata,
        )


@ray.remote
class SudokuEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Sudoku environment (Ray Actor)."""

    def __init__(self, cfg: Optional[SudokuConfig] = None):
        # cfg could contain game generation config like {'difficulty': 'medium'}
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.runner = SudokuRunner()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[SudokuMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of Sudoku interactions."""
        # Since logic is synchronous, process sequentially (can parallelize if logic becomes heavy)
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata_batch)
        ]

        # Unpack results and format according to EnvironmentReturn NamedTuple
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings = []
        all_next_metadata = []

        for obs, rew, term, stops, meta in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        # Calculate metrics
        solved_count = 0
        valid_moves_count = 0
        invalid_moves_count = 0
        total_reward = 0.0
        
        # Extract metrics from the trajectories
        for idx in range(len(batch["idx"])):
            rewards = batch.get("rewards", [])
            if idx < len(rewards) and len(rewards[idx]) > 0:
                episode_rewards = rewards[idx]
                total_reward += episode_rewards.sum().item()
                
                # Check if solved (final reward should be 10.0)
                if episode_rewards[-1].item() == 10.0:
                    solved_count += 1
                    
                # Count valid vs invalid moves
                for reward in episode_rewards:
                    r = reward.item()
                    if r > 0 and r != 10.0:  # Positive reward except for solving
                        valid_moves_count += 1
                    elif r < 0:  # Negative reward for invalid moves
                        invalid_moves_count += 1
        
        # Calculate metrics
        metrics = {}
        if len(batch["idx"]) > 0:
            metrics["sudoku_solve_rate"] = solved_count / len(batch["idx"])
            metrics["sudoku_avg_reward"] = total_reward / len(batch["idx"])
            
            total_moves = valid_moves_count + invalid_moves_count
            if total_moves > 0:
                metrics["sudoku_valid_move_rate"] = valid_moves_count / total_moves
        
        return batch, metrics 