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
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set

import ray
import torch
import numpy as np

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class ScrabbleConfig(TypedDict):
    dictionary_path: str
    num_players: int


class ScrabbleMetadata(TypedDict):
    game_state: Dict[str, Any]
    num_moves: int
    max_moves: int


class ScrabbleGameLogic:
    # Tile distribution and point values based on standard English Scrabble
    TILE_DISTRIBUTION = {
        'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9,
        'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6,
        'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, '*': 2  # * is blank
    }
    
    LETTER_VALUES = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
        'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
        'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, '*': 0  # * is blank
    }
    
    # Board premium squares (DL = double letter, TL = triple letter, DW = double word, TW = triple word)
    BOARD_PREMIUMS = {
        # Format: (row, col): premium_type
        (0, 0): 'TW', (0, 7): 'TW', (0, 14): 'TW',
        (7, 0): 'TW', (7, 14): 'TW',
        (14, 0): 'TW', (14, 7): 'TW', (14, 14): 'TW',
        
        (1, 1): 'DW', (1, 13): 'DW',
        (2, 2): 'DW', (2, 12): 'DW',
        (3, 3): 'DW', (3, 11): 'DW',
        (4, 4): 'DW', (4, 10): 'DW',
        (10, 4): 'DW', (10, 10): 'DW',
        (11, 3): 'DW', (11, 11): 'DW',
        (12, 2): 'DW', (12, 12): 'DW',
        (13, 1): 'DW', (13, 13): 'DW',
        
        (1, 5): 'TL', (1, 9): 'TL',
        (5, 1): 'TL', (5, 5): 'TL', (5, 9): 'TL', (5, 13): 'TL',
        (9, 1): 'TL', (9, 5): 'TL', (9, 9): 'TL', (9, 13): 'TL',
        (13, 5): 'TL', (13, 9): 'TL',
        
        (0, 3): 'DL', (0, 11): 'DL',
        (2, 6): 'DL', (2, 8): 'DL',
        (3, 0): 'DL', (3, 7): 'DL', (3, 14): 'DL',
        (6, 2): 'DL', (6, 6): 'DL', (6, 8): 'DL', (6, 12): 'DL',
        (7, 3): 'DL', (7, 11): 'DL',
        (8, 2): 'DL', (8, 6): 'DL', (8, 8): 'DL', (8, 12): 'DL',
        (11, 0): 'DL', (11, 7): 'DL', (11, 14): 'DL',
        (12, 6): 'DL', (12, 8): 'DL',
        (14, 3): 'DL', (14, 11): 'DL',
    }
    
    @staticmethod
    def _load_dictionary(dictionary_path: str) -> Set[str]:
        """Load the dictionary from the specified path."""
        try:
            with open(dictionary_path, 'r') as f:
                # Filter out words with apostrophes, hyphens, and proper nouns (capitalized)
                words = {word.strip().upper() for word in f 
                         if word.strip() and "'" not in word and "-" not in word
                         and not word[0].isupper() and len(word.strip()) > 1}
            return words
        except FileNotFoundError:
            # Fall back to a smaller set of common words if dictionary file not found
            print(f"Warning: Dictionary file {dictionary_path} not found. Using default word list.")
            common_words = {"THE", "OF", "AND", "TO", "IN", "IS", "YOU", "THAT", "IT", "HE",
                          "WAS", "FOR", "ON", "ARE", "AS", "WITH", "HIS", "THEY", "AT", "BE",
                          "THIS", "HAVE", "FROM", "OR", "ONE", "HAD", "BY", "WORD", "BUT", "NOT",
                          "WHAT", "ALL", "WERE", "WE", "WHEN", "YOUR", "CAN", "SAID", "THERE",
                          "USE", "AN", "EACH", "WHICH", "SHE", "DO", "HOW", "THEIR", "IF", "WILL",
                          "UP", "OTHER", "ABOUT", "OUT", "MANY", "THEN", "THEM", "THESE", "SO",
                          "SOME", "HER", "WOULD", "MAKE", "LIKE", "HIM", "INTO", "TIME", "HAS",
                          "LOOK", "TWO", "MORE", "WRITE", "GO", "SEE", "NUMBER", "NO", "WAY",
                          "COULD", "PEOPLE", "MY", "THAN", "FIRST", "WATER", "BEEN", "CALL",
                          "WHO", "OIL", "ITS", "NOW", "FIND", "LONG", "DOWN", "DAY", "DID", "GET",
                          "COME", "MADE", "MAY", "PART"}
            return common_words
            
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new Scrabble game."""
        dictionary_path = config.get("dictionary_path", "/usr/share/dict/words")
        num_players = config.get("num_players", 2)
        
        # Load the dictionary
        dictionary = ScrabbleGameLogic._load_dictionary(dictionary_path)
        
        # Create the tile bag
        tile_bag = []
        for letter, count in ScrabbleGameLogic.TILE_DISTRIBUTION.items():
            tile_bag.extend([letter] * count)
        random.shuffle(tile_bag)
        
        # Create the board (15x15 grid)
        board = [[None for _ in range(15)] for _ in range(15)]
        
        # Initialize players
        players = []
        for i in range(num_players):
            # Draw 7 tiles for each player
            rack = []
            for _ in range(min(7, len(tile_bag))):
                if tile_bag:
                    rack.append(tile_bag.pop())
            
            players.append({
                "id": i,
                "rack": rack,
                "score": 0,
                "passes": 0,  # Count consecutive passes (game ends when all players pass twice consecutively)
            })
        
        # Create and return the game state
        return {
            "board": board,
            "tile_bag": tile_bag,
            "players": players,
            "current_player": 0,
            "consecutive_passes": 0,
            "game_over": False,
            "winner": None,
            "dictionary": dictionary,
            "moves_history": [],
            "commands": {
                "play row col direction word": "Place a word on the board (direction: 'across' or 'down')",
                "exchange letters": "Exchange tiles from your rack (e.g., 'exchange ABC')",
                "pass": "Pass your turn",
                "view": "View the current state of the board",
                "help": "Show available commands",
            },
        }

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """Initialize Scrabble game and return welcome message."""
        return (
            f"\n===== SCRABBLE =====\n"
            f"Form words on a 15x15 grid, connecting to existing words.\n"
            f"- {len(game_state['players'])} players\n"
            f"- {len(game_state['tile_bag'])} tiles remaining in bag\n"
            f"- Use 'play row col direction word' to place a word (e.g., 'play 8 8 across HELLO')\n"
            f"- Use 'exchange letters' to exchange tiles (e.g., 'exchange ABC')\n"
            f"- Use 'pass' to skip your turn\n"
            f"- Use 'view' to see the current board state\n"
            f"- Use 'help' to see available commands\n"
            f"- You win by having the highest score when the game ends\n"
        )
        
    @staticmethod
    def _is_valid_play(game_state: Dict[str, Any], row: int, col: int, direction: str, word: str) -> Tuple[bool, str, List[Tuple[int, int, str]]]:
        """
        Check if a play is valid and return tiles placement if it is.
        
        Returns:
        - is_valid: boolean indicating if the play is valid
        - reason: string explaining why the play is invalid (if applicable)
        - tile_placements: list of (row, col, letter) tuples for the new tiles
        """
        # Check if word is in dictionary
        if word.upper() not in game_state["dictionary"]:
            return False, f"'{word}' is not in the dictionary.", []
            
        # Check if word fits on the board
        board = game_state["board"]
        if direction not in ["across", "down"]:
            return False, f"Direction must be 'across' or 'down', not '{direction}'.", []
            
        if direction == "across":
            if col + len(word) > 15:
                return False, f"Word '{word}' won't fit at position ({row},{col}) across.", []
        else:  # down
            if row + len(word) > 15:
                return False, f"Word '{word}' won't fit at position ({row},{col}) down.", []
                
        # Check if the word connects to existing tiles correctly
        connects_to_existing = False
        tile_placements = []
        current_player_rack = game_state["players"][game_state["current_player"]]["rack"].copy()
        
        # First word must go through the center
        center_used = board[7][7] is not None
        will_use_center = False
        
        # Check each letter in the word
        for i, letter in enumerate(word.upper()):
            if direction == "across":
                r, c = row, col + i
            else:  # down
                r, c = row + i, col
                
            # Check if position is the center
            if r == 7 and c == 7:
                will_use_center = True
                
            # If there's already a tile at this position
            if board[r][c] is not None:
                # The existing tile must match the word letter
                if board[r][c] != letter:
                    return False, f"Letter at position ({r},{c}) is '{board[r][c]}', which conflicts with '{letter}' in '{word}'.", []
                connects_to_existing = True
            else:
                # This position needs a new tile
                tile_placements.append((r, c, letter))
                
                # Check if player has this letter in their rack
                if letter in current_player_rack:
                    current_player_rack.remove(letter)
                elif '*' in current_player_rack:  # Use a blank
                    current_player_rack.remove('*')
                else:
                    return False, f"You don't have the letter '{letter}' in your rack.", []
                    
                # Check if the placement connects to any existing tiles (adjacent)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 15 and 0 <= nc < 15 and board[nr][nc] is not None:
                        if (nr, nc) not in [(r, c+1 if direction == "across" else c), 
                                            (r+1 if direction == "down" else r, c)]:
                            connects_to_existing = True
                            
        # If this is the first play, must use the center
        if not center_used and not will_use_center:
            return False, "The first word must go through the center square (8,8).", []
            
        # If not the first play, must connect to existing tiles
        if center_used and not connects_to_existing and not tile_placements:
            return False, "Your word must connect to existing tiles on the board.", []
            
        # Must place at least one new tile
        if not tile_placements:
            return False, "You must place at least one new tile from your rack.", []
            
        # Check if all formed words are valid
        all_words = ScrabbleGameLogic._get_all_formed_words(game_state, tile_placements)
        for formed_word in all_words:
            word_str = ''.join(letter for _, _, letter in formed_word)
            if word_str not in game_state["dictionary"] and len(word_str) > 1:
                return False, f"Would form invalid word: '{word_str}'.", []
                
        return True, "", tile_placements
        
    @staticmethod
    def _get_all_formed_words(game_state: Dict[str, Any], new_tiles: List[Tuple[int, int, str]]) -> List[List[Tuple[int, int, str]]]:
        """
        Get all words that would be formed by placing the new tiles.
        Returns a list of words, where each word is a list of (row, col, letter) tuples.
        """
        board = copy.deepcopy(game_state["board"])
        # Place the new tiles temporarily
        for r, c, letter in new_tiles:
            board[r][c] = letter
            
        # Find all words formed
        words = []
        
        # Check each new tile for words formed
        for row, col, _ in new_tiles:
            # Check for horizontal word
            h_word = []
            # Find the start of the horizontal word
            c = col
            while c > 0 and board[row][c-1] is not None:
                c -= 1
            # Collect the horizontal word
            start_col = c
            while c < 15 and board[row][c] is not None:
                h_word.append((row, c, board[row][c]))
                c += 1
            if len(h_word) > 1:
                words.append(h_word)
                
            # Check for vertical word
            v_word = []
            # Find the start of the vertical word
            r = row
            while r > 0 and board[r-1][col] is not None:
                r -= 1
            # Collect the vertical word
            start_row = r
            while r < 15 and board[r][col] is not None:
                v_word.append((r, col, board[r][col]))
                r += 1
            if len(v_word) > 1:
                words.append(v_word)
                
        return words
        
    @staticmethod
    def _calculate_score(game_state: Dict[str, Any], tile_placements: List[Tuple[int, int, str]]) -> int:
        """Calculate the score for a play."""
        # Get all words formed by this play
        all_words = ScrabbleGameLogic._get_all_formed_words(game_state, tile_placements)
        total_score = 0
        
        for word in all_words:
            word_score = 0
            word_multiplier = 1
            
            for r, c, letter in word:
                letter_score = ScrabbleGameLogic.LETTER_VALUES.get(letter, 0)
                letter_multiplier = 1
                
                # Check if this is a newly placed tile (eligible for premium squares)
                is_new_tile = (r, c, letter) in tile_placements
                
                # Apply premium squares for new tiles only
                if is_new_tile and (r, c) in ScrabbleGameLogic.BOARD_PREMIUMS:
                    premium = ScrabbleGameLogic.BOARD_PREMIUMS[(r, c)]
                    if premium == 'DL':
                        letter_multiplier = 2
                    elif premium == 'TL':
                        letter_multiplier = 3
                    elif premium == 'DW':
                        word_multiplier *= 2
                    elif premium == 'TW':
                        word_multiplier *= 3
                
                # Add the letter score with its multiplier
                word_score += letter_score * letter_multiplier
            
            # Apply word multiplier
            word_score *= word_multiplier
            total_score += word_score
        
        # Bonus for using all 7 tiles (50 points)
        if len(tile_placements) == 7:
            total_score += 50
            
        return total_score
    
    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the Scrabble game."""
        # Default return values
        response = "Unknown command. Type 'help' to see available commands."
        reward = 0.0
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)
        
        # Check if game is already over
        if new_state["game_over"]:
            return "Game is already over.", 0.0, True, new_state
        
        # Get current player
        current_player = new_state["current_player"]
        
        # Parse the action
        if action.startswith("play "):
            try:
                # Format: "play row col direction word"
                parts = action.split()
                if len(parts) < 5:
                    return "Invalid play format. Use: play row col direction word", 0.0, False, new_state
                
                row = int(parts[1]) - 1  # Convert to 0-indexed
                col = int(parts[2]) - 1  # Convert to 0-indexed
                direction = parts[3].lower()
                word = ''.join(parts[4:]).upper()  # Join remaining parts as the word
                
                # Validate the play
                is_valid, reason, tile_placements = ScrabbleGameLogic._is_valid_play(new_state, row, col, direction, word)
                
                if not is_valid:
                    return reason, -0.5, False, new_state  # Small penalty for invalid play
                
                # Calculate score
                score = ScrabbleGameLogic._calculate_score(new_state, tile_placements)
                
                # Update the board
                for r, c, letter in tile_placements:
                    new_state["board"][r][c] = letter
                    
                # Update player score
                new_state["players"][current_player]["score"] += score
                
                # Remove used tiles from rack
                player_rack = new_state["players"][current_player]["rack"]
                for _, _, letter in tile_placements:
                    if letter in player_rack:
                        player_rack.remove(letter)
                    elif '*' in player_rack:  # Use a blank
                        player_rack.remove('*')
                
                # Draw new tiles
                tiles_needed = min(len(tile_placements), len(new_state["tile_bag"]))
                for _ in range(tiles_needed):
                    if new_state["tile_bag"]:
                        player_rack.append(new_state["tile_bag"].pop())
                
                # Reset passes counter
                new_state["players"][current_player]["passes"] = 0
                new_state["consecutive_passes"] = 0
                
                # Record the move
                new_state["moves_history"].append({
                    "player": current_player,
                    "action": action,
                    "score": score,
                })
                
                # Check game ending conditions
                if len(player_rack) == 0 and not new_state["tile_bag"]:
                    # Player used all tiles and bag is empty - game ends
                    new_state["game_over"] = True
                    is_terminated = True
                    
                    # Find the winner
                    max_score = -1
                    winner = -1
                    for i, player in enumerate(new_state["players"]):
                        if player["score"] > max_score:
                            max_score = player["score"]
                            winner = i
                    new_state["winner"] = winner
                    
                    if winner == current_player:
                        response = f"You played '{word}' for {score} points. Game over! You win with {max_score} points!"
                        reward = 5.0  # Big reward for winning
                    else:
                        response = f"You played '{word}' for {score} points. Game over! Player {winner+1} wins with {max_score} points!"
                        reward = score / 20.0  # Small reward proportional to score
                else:
                    # Game continues
                    response = f"You played '{word}' for {score} points. Your score is now {new_state['players'][current_player]['score']}."
                    reward = score / 20.0  # Reward proportional to score
                    
                    # Move to next player
                    new_state["current_player"] = (current_player + 1) % len(new_state["players"])
                
            except (ValueError, IndexError) as e:
                return f"Invalid play format: {str(e)}. Use: play row col direction word", -0.1, False, new_state
                
        elif action.startswith("exchange "):
            # Format: "exchange ABC" to exchange tiles A, B, and C
            if len(new_state["tile_bag"]) < 7:
                return "Cannot exchange tiles when fewer than 7 tiles remain in the bag.", -0.1, False, new_state
                
            letters_to_exchange = action[9:].strip().upper()
            if not letters_to_exchange:
                return "You must specify which tiles to exchange.", -0.1, False, new_state
                
            player_rack = new_state["players"][current_player]["rack"]
            
            # Check if player has these tiles
            temp_rack = player_rack.copy()
            for letter in letters_to_exchange:
                if letter in temp_rack:
                    temp_rack.remove(letter)
                else:
                    return f"You don't have enough '{letter}' tiles in your rack.", -0.1, False, new_state
            
            # Perform the exchange
            for letter in letters_to_exchange:
                player_rack.remove(letter)
                new_state["tile_bag"].append(letter)
                
            # Shuffle the bag
            random.shuffle(new_state["tile_bag"])
            
            # Draw new tiles
            for _ in range(len(letters_to_exchange)):
                player_rack.append(new_state["tile_bag"].pop())
                
            # Reset passes counter
            new_state["players"][current_player]["passes"] = 0
            new_state["consecutive_passes"] = 0
            
            # Record the move
            new_state["moves_history"].append({
                "player": current_player,
                "action": f"exchange {letters_to_exchange}",
                "score": 0,
            })
            
            # Move to next player
            new_state["current_player"] = (current_player + 1) % len(new_state["players"])
            
            response = f"Exchanged {len(letters_to_exchange)} tiles. Your new rack: {', '.join(player_rack)}"
            reward = -0.5  # Small penalty for exchanging instead of playing
            
        elif action == "pass":
            # Increment pass counters
            new_state["players"][current_player]["passes"] += 1
            new_state["consecutive_passes"] += 1
            
            # Record the move
            new_state["moves_history"].append({
                "player": current_player,
                "action": "pass",
                "score": 0,
            })
            
            # Check if all players have passed twice consecutively
            if new_state["consecutive_passes"] >= len(new_state["players"]) * 2:
                # Game ends
                new_state["game_over"] = True
                is_terminated = True
                
                # Find the winner
                max_score = -1
                winner = -1
                for i, player in enumerate(new_state["players"]):
                    if player["score"] > max_score:
                        max_score = player["score"]
                        winner = i
                new_state["winner"] = winner
                
                if winner == current_player:
                    response = f"You passed. Game over due to consecutive passes! You win with {max_score} points!"
                    reward = 2.0  # Reward for winning, but less than playing words
                else:
                    response = f"You passed. Game over due to consecutive passes! Player {winner+1} wins with {max_score} points!"
                    reward = -1.0  # Penalty for losing
            else:
                # Move to next player
                new_state["current_player"] = (current_player + 1) % len(new_state["players"])
                response = f"You passed your turn. Next player's turn."
                reward = -1.0  # Penalty for passing
                
        elif action == "help":
            commands = new_state["commands"]
            help_text = "Available commands:\n"
            for cmd, desc in commands.items():
                help_text += f"- {cmd}: {desc}\n"
            response = help_text
                
        return response, reward, is_terminated, new_state 

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """Render the current Scrabble game state."""
        board = game_state["board"]
        current_player = game_state["current_player"]
        players = game_state["players"]
        
        # Create the output string
        output = ["\n===== SCRABBLE =====\n"]
        
        # Add game info
        output.append(f"Player {current_player + 1}'s turn")
        output.append(f"Tiles in bag: {len(game_state['tile_bag'])}")
        
        # Show scores
        scores = []
        for i, player in enumerate(players):
            scores.append(f"Player {i + 1}: {player['score']} pts")
        output.append("Scores: " + ", ".join(scores))
        
        # Show current player's rack
        rack = players[current_player]["rack"]
        output.append(f"\nYour rack: {' '.join(rack)}")
        
        # Column headers (A-O)
        output.append("\n   " + " ".join([chr(65 + i) for i in range(15)]))
        
        # Top border
        output.append("  +" + "-" * 31 + "+")
        
        # Board rows
        for i in range(15):
            row = [f"{i + 1:2d}|"]
            for j in range(15):
                if board[i][j] is not None:
                    # Show placed tile
                    row.append(f"{board[i][j]}")
                else:
                    # Show premium square or empty space
                    if (i, j) in ScrabbleGameLogic.BOARD_PREMIUMS:
                        premium = ScrabbleGameLogic.BOARD_PREMIUMS[(i, j)]
                        row.append(premium[0] + premium[1])  # e.g., "TW", "DL", etc.
                    else:
                        row.append("__")
                row.append(" ")
            row.append("|")
            output.append("".join(row))
        
        # Bottom border
        output.append("  +" + "-" * 31 + "+")
        
        # Show last few moves
        if game_state["moves_history"]:
            output.append("\nLast move:")
            last_move = game_state["moves_history"][-1]
            output.append(f"Player {last_move['player'] + 1}: {last_move['action']} ({last_move['score']} pts)")
        
        # Game status
        if game_state["game_over"]:
            winner = game_state["winner"]
            output.append(f"\nGame over! Player {winner + 1} wins with {players[winner]['score']} points!")
            
        return "\n".join(output)


class ScrabbleRunner:
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
        metadata: ScrabbleMetadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[ScrabbleMetadata],
    ]:
        """Processes a single turn for the Scrabble task."""
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
            rendered_game = ScrabbleGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_game = ScrabbleGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nViewing the board. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = (
                ScrabbleGameLogic.step(parsed_action, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            # Render the game state after the action
            rendered_game = ScrabbleGameLogic.render(next_game_state)
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
class ScrabbleEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Scrabble environment (Ray Actor)."""

    def __init__(self, cfg: Optional[ScrabbleConfig] = None):
        # cfg could contain game generation config
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.runner = ScrabbleRunner()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[ScrabbleMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of Scrabble interactions."""
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
        total_scores = []
        win_count = 0
        total_words_played = 0
        word_lengths = []
        
        # Extract metrics from the trajectories
        for idx in range(len(batch["idx"])):
            try:
                # Find the last non-None metadata for this trajectory
                trajectory_metadata = [
                    m for m in batch["metadata"][idx] if m is not None
                ]
                if trajectory_metadata:
                    last_metadata = trajectory_metadata[-1]
                    game_state = last_metadata["game_state"]
                    
                    # Get player scores
                    player_score = game_state["players"][0]["score"]  # Player is always index 0
                    total_scores.append(player_score)
                    
                    # Check if player won
                    if game_state["game_over"] and game_state["winner"] == 0:
                        win_count += 1
                        
                    # Count words played and their lengths
                    for move in game_state["moves_history"]:
                        if move["player"] == 0 and move["action"].startswith("play "):
                            total_words_played += 1
                            parts = move["action"].split()
                            if len(parts) >= 5:
                                word = ''.join(parts[4:])
                                word_lengths.append(len(word))
            except (KeyError, IndexError, TypeError):
                # Skip if data is missing or malformed
                pass
                
        # Calculate metrics
        metrics = {}
        if len(batch["idx"]) > 0:
            metrics["scrabble_win_rate"] = win_count / len(batch["idx"])
            
        if total_scores:
            metrics["scrabble_avg_score"] = sum(total_scores) / len(total_scores)
            metrics["scrabble_max_score"] = max(total_scores)
            
        if total_words_played > 0:
            metrics["scrabble_words_per_game"] = total_words_played / len(batch["idx"])
            
        if word_lengths:
            metrics["scrabble_avg_word_length"] = sum(word_lengths) / len(word_lengths)
        
        return batch, metrics 