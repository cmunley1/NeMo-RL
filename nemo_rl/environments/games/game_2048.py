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


class Game2048Config(TypedDict):
    grid_size: int
    initial_tiles: int


class Game2048Metadata(TypedDict):
    game_state: Dict[str, Any]
    num_moves: int
    max_moves: int


class Game2048GameLogic:
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new 2048 game."""
        grid_size = config.get("grid_size", 4)
        initial_tiles = config.get("initial_tiles", 2)

        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Add initial tiles
        for _ in range(initial_tiles):
            Game2048GameLogic._add_new_tile(grid)
            
        score = 0
        
        # Create and return the game state
        return {
            "grid_size": grid_size,
            "grid": grid.tolist(),
            "score": score,
            "game_over": False,
            "win": False,
            "max_tile": 2,  # Initial max tile
            "commands": {
                "up": "Swipe up",
                "down": "Swipe down",
                "left": "Swipe left",
                "right": "Swipe right",
                "view": "View the current state of the board",
            },
        }

    @staticmethod
    def _add_new_tile(grid: np.ndarray) -> None:
        """Add a new tile (2 or 4) to a random empty cell."""
        # Find empty cells
        empty_cells = np.argwhere(grid == 0)
        if len(empty_cells) == 0:
            return  # No empty cells
            
        # Choose a random empty cell
        cell_idx = random.randint(0, len(empty_cells) - 1)
        row, col = empty_cells[cell_idx]
        
        # Place a 2 (90% chance) or a 4 (10% chance)
        value = 2 if random.random() < 0.9 else 4
        grid[row, col] = value

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """Initialize 2048 game and return welcome message."""
        return (
            f"\n===== 2048 =====\n"
            f"Combine tiles with the same number to create a tile with the value 2048!\n"
            f"- Grid size: {game_state['grid_size']}x{game_state['grid_size']}\n"
            f"- Use 'up', 'down', 'left', 'right' to move tiles\n"
            f"- Use 'view' to see the current state of the board"
        )

    @staticmethod
    def _move_grid(grid: np.ndarray, direction: str) -> Tuple[np.ndarray, int, bool]:
        """Move the grid in the specified direction and return the new grid, score gained, and whether a move was made."""
        grid_size = grid.shape[0]
        score = 0
        moved = False
        new_grid = grid.copy()
        
        # Rotate the grid to simplify the algorithm
        # We'll always merge left, then rotate back
        if direction == "up":
            new_grid = np.rot90(new_grid, k=1)
        elif direction == "right":
            new_grid = np.rot90(new_grid, k=2)
        elif direction == "down":
            new_grid = np.rot90(new_grid, k=3)
            
        # For each row, move and merge tiles
        for i in range(grid_size):
            # Extract the row
            row = new_grid[i, :].copy()
            
            # Remove zeros (empty cells)
            row = row[row != 0]
            
            # Merge adjacent tiles with the same value
            j = 0
            while j < len(row) - 1:
                if row[j] == row[j+1]:
                    row[j] *= 2  # Double the value
                    score += row[j]  # Add to score
                    row = np.delete(row, j+1)  # Remove the merged tile
                    moved = True
                j += 1
                
            # Pad with zeros to maintain size
            padded_row = np.zeros(grid_size, dtype=int)
            padded_row[:len(row)] = row
            
            # Check if the move changed anything
            if not np.array_equal(new_grid[i, :], padded_row):
                moved = True
                
            # Update the grid
            new_grid[i, :] = padded_row
            
        # Rotate back
        if direction == "up":
            new_grid = np.rot90(new_grid, k=3)
        elif direction == "right":
            new_grid = np.rot90(new_grid, k=2)
        elif direction == "down":
            new_grid = np.rot90(new_grid, k=1)
            
        return new_grid, score, moved

    @staticmethod
    def _check_game_over(grid: np.ndarray) -> bool:
        """Check if the game is over (no moves possible)."""
        # If there are empty cells, game is not over
        if 0 in grid:
            return False
            
        # Check if there are adjacent tiles with the same value
        grid_size = grid.shape[0]
        
        # Check horizontally
        for i in range(grid_size):
            for j in range(grid_size - 1):
                if grid[i, j] == grid[i, j+1]:
                    return False
                    
        # Check vertically
        for i in range(grid_size - 1):
            for j in range(grid_size):
                if grid[i, j] == grid[i+1, j]:
                    return False
                    
        # No moves left
        return True

    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the 2048 game."""
        # Default return values
        response = "Unknown command. Use 'up', 'down', 'left', 'right', or 'view'."
        reward = 0.0
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)
        
        # Check if game is already over
        if new_state["game_over"]:
            return "Game is already over.", 0.0, True, new_state
            
        if action in ["up", "down", "left", "right"]:
            # Convert grid to numpy array for easier manipulation
            grid = np.array(new_state["grid"])
            
            # Move the grid
            new_grid, score_gained, moved = Game2048GameLogic._move_grid(grid, action)
            
            if moved:
                # Add a new tile if the move was successful
                Game2048GameLogic._add_new_tile(new_grid)
                
                # Update the state
                new_state["grid"] = new_grid.tolist()
                new_state["score"] += score_gained
                
                # Update max tile
                new_max_tile = np.max(new_grid)
                new_state["max_tile"] = int(new_max_tile)
                
                # Check for win (2048 tile)
                if new_max_tile >= 2048 and not new_state["win"]:
                    new_state["win"] = True
                    response = f"Congratulations! You've reached 2048! You can continue playing to maximize your score."
                    reward = 10.0  # Significant reward for reaching 2048
                else:
                    response = f"Moved {action}. Score: {new_state['score']}. Max tile: {new_state['max_tile']}."
                    # Reward based on score gained and max tile reached
                    reward = score_gained / 100.0 + (np.log2(new_max_tile) - np.log2(new_state["max_tile"] or 2)) / 10.0
                    
                # Check if game is over
                if Game2048GameLogic._check_game_over(new_grid):
                    new_state["game_over"] = True
                    response += " Game over! No more moves possible."
                    is_terminated = True
            else:
                response = f"Cannot move {action}. Try a different direction."
                
            # Ensure game terminates eventually
            if new_state["max_tile"] >= 8192:  # If someone reaches 8192, consider it a win and terminate
                new_state["game_over"] = True
                response += " Amazing! You've reached 8192! Game complete."
                is_terminated = True
                reward += 20.0  # Extra reward for reaching 8192
                
        return response, reward, is_terminated, new_state

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """Render the current 2048 game state."""
        grid = game_state["grid"]
        grid_size = game_state["grid_size"]
        score = game_state["score"]
        
        # Calculate cell width based on the maximum number
        max_num = game_state["max_tile"]
        cell_width = max(len(str(max_num)), 4)
        
        # Create the output string
        output = [f"\n===== 2048 =====  Score: {score}  Max Tile: {max_num}\n"]
        
        # Top border
        border = "+" + "".join(["-" * (cell_width + 2) + "+" for _ in range(grid_size)])
        output.append(border)
        
        # Rows
        for row in grid:
            row_str = "|"
            for cell in row:
                if cell == 0:
                    row_str += " " * (cell_width + 2) + "|"
                else:
                    # Center the number
                    num_str = str(cell)
                    padding = (cell_width - len(num_str)) // 2
                    extra = (cell_width - len(num_str)) % 2
                    row_str += " " * (padding + 1) + num_str + " " * (padding + 1 + extra) + "|"
            output.append(row_str)
            output.append(border)
            
        # Game status
        if game_state["game_over"]:
            output.append("\nGame Over!")
        if game_state["win"]:
            output.append("\nCongratulations! You've reached 2048!")
            
        return "\n".join(output)


class Game2048Runner:
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
        metadata: Game2048Metadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[Game2048Metadata],
    ]:
        """Processes a single turn for the 2048 game."""
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
            rendered_game = Game2048GameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_game = Game2048GameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nViewing the board. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = (
                Game2048GameLogic.step(parsed_action, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            # Render the game state after the action
            rendered_game = Game2048GameLogic.render(next_game_state)
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
class Game2048Env(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """2048 game environment (Ray Actor)."""

    def __init__(self, cfg: Optional[Game2048Config] = None):
        # cfg could contain game generation config like {'grid_size': 4, 'initial_tiles': 2}
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.runner = Game2048Runner()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[Game2048Metadata],
    ) -> EnvironmentReturn:
        """Processes a batch of 2048 game interactions."""
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
        # Calculate metrics based on final game states
        final_scores = []
        max_tiles = []
        win_count = 0
        
        # Extract metrics from the last game states
        for idx in range(len(batch["idx"])):
            try:
                # Find the last non-None metadata for this trajectory
                trajectory_metadata = [
                    m for m in batch["metadata"][idx] if m is not None
                ]
                if trajectory_metadata:
                    last_metadata = trajectory_metadata[-1]
                    game_state = last_metadata["game_state"]
                    final_scores.append(game_state["score"])
                    max_tiles.append(game_state["max_tile"])
                    win_count += 1 if game_state["win"] else 0
            except (KeyError, IndexError, TypeError):
                # Skip if data is missing or malformed
                pass
                
        # Calculate metrics
        metrics = {}
        if final_scores:
            metrics["game_2048_avg_score"] = sum(final_scores) / len(final_scores)
            metrics["game_2048_max_score"] = max(final_scores)
        if max_tiles:
            metrics["game_2048_avg_max_tile"] = sum(max_tiles) / len(max_tiles)
            metrics["game_2048_highest_tile"] = max(max_tiles)
        if len(batch["idx"]) > 0:
            metrics["game_2048_win_rate"] = win_count / len(batch["idx"])
        
        return batch, metrics 