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


class BlackjackConfig(TypedDict):
    num_decks: int
    dealer_hit_soft_17: bool


class BlackjackMetadata(TypedDict):
    game_state: Dict[str, Any]
    num_moves: int
    max_moves: int


class BlackjackGameLogic:
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new Blackjack game."""
        num_decks = config.get("num_decks", 1)
        dealer_hit_soft_17 = config.get("dealer_hit_soft_17", True)

        # Create a deck 
        suits = ["♠", "♥", "♦", "♣"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        card_values = {"A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
        
        # Create a deck with multiple decks
        deck = []
        for _ in range(num_decks):
            for suit in suits:
                for rank in ranks:
                    deck.append({"rank": rank, "suit": suit, "value": card_values[rank]})
        
        # Shuffle 
        random.shuffle(deck)
        
        # Deal 
        player_hand = [deck.pop(), deck.pop()]
        dealer_hand = [deck.pop(), deck.pop()]
        
        # Return the game state
        return {
            "num_decks": num_decks,
            "dealer_hit_soft_17": dealer_hit_soft_17,
            "deck": deck,
            "player_hand": player_hand,
            "dealer_hand": dealer_hand,
            "player_stand": False,
            "game_over": False,
            "result": None,
            "commands": {
                "hit": "Draw another card",
                "stand": "End your turn, dealer plays",
                "view": "View the current game state",
            },
        }

    @staticmethod
    def init(game_state: Dict[str, Any]) -> str:
        """Initialize Blackjack game and return welcome message."""
        return (
            f"\n===== BLACKJACK =====\n"
            f"Try to get as close to 21 as possible without going over.\n"
            f"- Dealer hits on soft 17: {game_state['dealer_hit_soft_17']}\n"
            f"- Number of decks: {game_state['num_decks']}\n"
            f"- Type 'hit' to draw another card\n"
            f"- Type 'stand' to end your turn\n"
            f"- Type 'view' to see the current state of the game"
        )

    @staticmethod
    def calculate_hand_value(hand: List[Dict[str, Any]]) -> Tuple[int, bool]:
        """Calculate the value of a hand, handling aces appropriately."""
        value = sum(card["value"] for card in hand)
        num_aces = sum(1 for card in hand if card["rank"] == "A")
        
        soft_hand = False
        while value > 21 and num_aces > 0:
            value -= 10  # Convert ace from 11 to 1
            num_aces -= 1
            soft_hand = True
            
        return value, soft_hand

    @staticmethod
    def dealer_play(game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Dealer plays their turn."""
        new_state = copy.deepcopy(game_state)
        dealer_hand = new_state["dealer_hand"]
        
        # Dealer plays until they reach 17 or more
        while True:
            value, soft = BlackjackGameLogic.calculate_hand_value(dealer_hand)
            
            if value >= 17 and not (value == 17 and soft and new_state["dealer_hit_soft_17"]):
                break
                
            if new_state["deck"]:
                dealer_hand.append(new_state["deck"].pop())
            else:
                break
        
        return new_state

    @staticmethod
    def determine_winner(game_state: Dict[str, Any]) -> Tuple[str, float]:
        """Determine the winner and the reward."""
        player_value, _ = BlackjackGameLogic.calculate_hand_value(game_state["player_hand"])
        dealer_value, _ = BlackjackGameLogic.calculate_hand_value(game_state["dealer_hand"])
        
        if player_value > 21:
            return "Bust! You went over 21. Dealer wins.", -1.0
            
        if dealer_value > 21:
            return "Dealer busts! You win!", 1.0
            
        if player_value > dealer_value:
            return "You win! Your hand is higher than the dealer's.", 1.0
        elif player_value < dealer_value:
            return "Dealer wins. Their hand is higher than yours.", -1.0
        else:
            return "Push! It's a tie.", 0.0

    @staticmethod
    def step(
        action: str, game_state: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the Blackjack game."""
        # Defaults
        response = "Unknown command. Type 'hit', 'stand', or 'view'."
        reward = 0.0
        is_terminated = False

        # Deep copy game state to avoid modifying the original
        new_state = copy.deepcopy(game_state)
        
        # Check if game is over
        if new_state["game_over"]:
            return "Game is already over. " + new_state["result"], 0.0, True, new_state

        if action == "hit":
            # Player draws a card
            if new_state["deck"]:
                new_state["player_hand"].append(new_state["deck"].pop())
                
                # Check if player busts
                player_value, _ = BlackjackGameLogic.calculate_hand_value(new_state["player_hand"])
                if player_value > 21:
                    result, reward = BlackjackGameLogic.determine_winner(new_state)
                    new_state["game_over"] = True
                    new_state["result"] = result
                    is_terminated = True
                    response = f"You drew a {new_state['player_hand'][-1]['rank']}{new_state['player_hand'][-1]['suit']}. {result}"
                else:
                    response = f"You drew a {new_state['player_hand'][-1]['rank']}{new_state['player_hand'][-1]['suit']}."
            else:
                response = "No more cards in the deck!"
                
        elif action == "stand":
            # Player stands, dealer plays
            new_state["player_stand"] = True
            new_state = BlackjackGameLogic.dealer_play(new_state)
            
            # Determine winner
            result, reward = BlackjackGameLogic.determine_winner(new_state)
            new_state["game_over"] = True
            new_state["result"] = result
            is_terminated = True
            response = f"You stand. Dealer reveals cards. {result}"
            
        return response, reward, is_terminated, new_state

    @staticmethod
    def render(game_state: Dict[str, Any]) -> str:
        """Render the current Blackjack game state."""
        player_hand = game_state["player_hand"]
        dealer_hand = game_state["dealer_hand"]
        player_value, player_soft = BlackjackGameLogic.calculate_hand_value(player_hand)
        dealer_value, dealer_soft = BlackjackGameLogic.calculate_hand_value(dealer_hand)
        
        show_dealer = game_state["player_stand"] or game_state["game_over"]
        
        output = ["\n===== BLACKJACK =====\n"]
        
        output.append("Dealer's hand:")
        if show_dealer:
            dealer_cards = " ".join(f"{card['rank']}{card['suit']}" for card in dealer_hand)
            output.append(f"  {dealer_cards} (Value: {dealer_value}{'s' if dealer_soft else ''})")
        else:
            output.append(f"  {dealer_hand[0]['rank']}{dealer_hand[0]['suit']} ?")
        
        output.append("\nYour hand:")
        player_cards = " ".join(f"{card['rank']}{card['suit']}" for card in player_hand)
        output.append(f"  {player_cards} (Value: {player_value}{'s' if player_soft else ''})")
        
        if game_state["game_over"] and game_state["result"]:
            output.append(f"\nResult: {game_state['result']}")
            
        return "\n".join(output)


class BlackjackRunner:
    def __init__(self):
        pass  

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
        metadata: BlackjackMetadata,
    ) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[BlackjackMetadata],
    ]:
        """Processes a single turn for the blackjack task."""
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
            rendered_game = BlackjackGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nInvalid response format no move made. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        elif parsed_action == "view":
            rendered_game = BlackjackGameLogic.render(game_state)
            next_observation_content = f"<environment>\n{rendered_game}\n\nViewing the game. No move made.</environment>"
        else:
            # Execute the game step
            step_response, reward, game_over, next_game_state = (
                BlackjackGameLogic.step(parsed_action, game_state)
            )

            turn_reward = reward
            is_terminated = game_over
            next_metadata["game_state"] = next_game_state
            next_metadata["num_moves"] = current_moves + 1

            # Render the game state after the action
            rendered_game = BlackjackGameLogic.render(next_game_state)
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
class BlackjackEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """Blackjack environment (Ray Actor)."""

    def __init__(self, cfg: Optional[BlackjackConfig] = None):
        # cfg can contain game generation config like {'num_decks': 1, 'dealer_hit_soft_17': True}
        self.game_config = cfg.get("game_config", {}) if cfg else {}
        self.runner = BlackjackRunner()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[BlackjackMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of blackjack interactions."""
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
        # Calculate win rate based on final reward > 0
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        win_rate = (
            (final_rewards > 0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        # Calculate push rate (tie)
        push_rate = (
            (final_rewards == 0.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        # Calculate loss rate
        loss_rate = (
            (final_rewards < 0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        
        return batch, {
            "blackjack_win_rate": win_rate,
            "blackjack_push_rate": push_rate,
            "blackjack_loss_rate": loss_rate,
        } 