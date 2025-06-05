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

import os
import re
import shutil
import tempfile
from typing import Any, Optional, TypedDict
import json

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.tools.bash_tool import BashTool
from nemo_rl.environments.tools.file_tool import FileTool


class MathToolsConfig(TypedDict):
    max_turns: int
    max_bash_timeout: int  # seconds
    max_file_size: int  # bytes
    enable_network: bool
    memory_limit: int  # MB
    cpu_limit: float  # cores


class MathToolsMetadata(TypedDict):
    problem_id: str
    problem_text: str
    ground_truth: str  
    working_dir: str
    current_turn: int
    max_turns: int
    files_created: list[str]
    bash_history: list[tuple[str, str]]  
    tool_calls_count: dict[str, int]  


@ray.remote
class MathToolsEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MathToolsConfig):
        self.cfg = cfg
        self.bash_tool = BashTool(
            timeout=cfg.get("max_bash_timeout", 30),
            memory_limit=cfg.get("memory_limit", 512),
            cpu_limit=cfg.get("cpu_limit", 1.0),
            enable_network=cfg.get("enable_network", False),
        )
        self.file_tool = FileTool(
            max_file_size=cfg.get("max_file_size", 1024 * 1024)  # 1MB 
        )

    @property
    def TOOLS(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute bash commands in a controlled environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_create",
                    "description": "Create a new file with specified content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path where the file should be created."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file."
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_edit",
                    "description": "Edit an existing file by replacing old content with new content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to edit."
                            },
                            "old_content": {
                                "type": "string",
                                "description": "The content to replace. Leave empty to append to file."
                            },
                            "new_content": {
                                "type": "string",
                                "description": "The new content to replace the old content with."
                            }
                        },
                        "required": ["path", "old_content", "new_content"]
                    }
                }
            }
        ]

    def shutdown(self) -> None:
        pass

    def get_tools_definition(self) -> list[dict[str, Any]]:
        return self.TOOLS

    def _parse_tool_calls(self, content: str) -> list[tuple[str, dict[str, Any]]]:
        tool_calls = []
        
        # Tool call format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        for match in re.finditer(tool_call_pattern, content, re.DOTALL):
            tool_call_json = match.group(1).strip()
            
            try:
                tool_call_data = json.loads(tool_call_json)
                tool_name = tool_call_data.get("name")
                arguments = tool_call_data.get("arguments", {})
                
                if tool_name:
                    tool_calls.append((tool_name, arguments))
                    
            except json.JSONDecodeError as e:
                continue
        
        return tool_calls

    def _extract_final_answer(self, content: str) -> Optional[str]:
        # TODO: use math grader
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None

    def _normalize_answer(self, answer: str) -> str:
        # TODO: use math grader
        answer = answer.strip()
        
        try:
            if answer.replace('.', '').replace('-', '').replace('+', '').replace('/', '').replace('*', '').replace('(', '').replace(')', '').replace(' ', '').isdigit():
                result = eval(answer)
                if isinstance(result, (int, float)):
                    if isinstance(result, float) and result.is_integer():
                        return str(int(result))
                    else:
                        return f"{result:.6f}".rstrip('0').rstrip('.')
        except:
            pass
        
        return answer

    def _check_answer(self, submitted: str, ground_truth: str) -> bool:
        """Check if submitted answer matches ground truth."""
        norm_submitted = self._normalize_answer(submitted)
        norm_ground_truth = self._normalize_answer(ground_truth)
        
        if norm_submitted == norm_ground_truth:
            return True
        
        try:
            sub_val = float(norm_submitted)
            gt_val = float(norm_ground_truth)
            return abs(sub_val - gt_val) < 1e-6
        except:
            pass
        
        return False

    def _execute_tool(
        self, tool_name: str, tool_args: dict[str, Any], working_dir: str
    ) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "bash":
                return self.bash_tool.execute(
                    command=tool_args["command"],
                    working_dir=working_dir
                )
            elif tool_name == "file_create":
                return self.file_tool.create(
                    path=tool_args["path"],
                    content=tool_args["content"],
                    working_dir=working_dir
                )
            elif tool_name == "file_edit":
                file_path = os.path.join(working_dir, tool_args["path"])
                if not os.path.exists(file_path):
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write("")
                
                try:
                    with open(file_path, 'r') as f:
                        current_content = f.read()
                    
                    old_content = tool_args["old_content"]
                    new_content = tool_args["new_content"]
                    
                    if old_content == "":
                        updated_content = current_content + new_content
                    elif old_content in current_content:
                        updated_content = current_content.replace(old_content, new_content)
                    else:
                        return f"Error: Content to replace not found in file '{tool_args['path']}'"
                    
                    with open(file_path, 'w') as f:
                        f.write(updated_content)
                    
                    return f"File '{tool_args['path']}' updated successfully"
                    
                except Exception as e:
                    return f"Error editing file '{tool_args['path']}': {str(e)}"
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata_batch: list[MathToolsMetadata],
    ) -> EnvironmentReturn:
        """Process a batch of interactions."""
        observations = []
        rewards = []
        terminateds = []
        next_metadata = []
        next_stop_strings = []
        
        for message_log, metadata in zip(message_log_batch, metadata_batch):
            last_assistant_msg = ""
            if message_log and message_log[-1]["role"] == "assistant":
                last_assistant_msg = message_log[-1]["content"]
            
            if metadata["current_turn"] >= metadata["max_turns"]:
                observations.append({
                    "role": "environment",
                    "content": f"Error: Maximum turns ({metadata['max_turns']}) reached."
                })
                rewards.append(0.0)
                terminateds.append(True)
                next_metadata.append(None)
                next_stop_strings.append(None)
                continue
            
            final_answer = self._extract_final_answer(last_assistant_msg)
            if final_answer is not None:
                is_correct = self._check_answer(final_answer, metadata["ground_truth"])
                observations.append({
                    "role": "environment",
                    "content": f"Final answer submitted: {final_answer}. "
                              f"{'Correct!' if is_correct else f'Incorrect. Expected: {metadata['ground_truth']}'}"
                })
                rewards.append(1.0 if is_correct else 0.0)
                terminateds.append(True)
                next_metadata.append(None)
                next_stop_strings.append(None)
                continue
            
            tool_calls = self._parse_tool_calls(last_assistant_msg)
            
            if not tool_calls:
                observations.append({
                    "role": "environment",
                    "content": "No valid tool calls or final answer found. Please use the provided tools or submit your final answer."
                })
                rewards.append(0.0)
                terminateds.append(False)
                updated_metadata = metadata.copy()
                updated_metadata["current_turn"] += 1
                next_metadata.append(updated_metadata)
                next_stop_strings.append(["</tool_call>", "</answer>"])
                continue
            
            tool_results = []
            updated_metadata = metadata.copy()
            
            for tool_name, tool_args in tool_calls:
                result = self._execute_tool(tool_name, tool_args, metadata["working_dir"])
                tool_results.append(f"[{tool_name}]\n{result}")
                
                updated_metadata["tool_calls_count"][tool_name] = (
                    updated_metadata["tool_calls_count"].get(tool_name, 0) + 1
                )
                
                if tool_name == "bash":
                    updated_metadata["bash_history"].append((tool_args["command"], result))
                elif tool_name in ["file_create", "file_edit"]:
                    if tool_args["path"] not in updated_metadata["files_created"]:
                        updated_metadata["files_created"].append(tool_args["path"])
            
            observation_content = "\n\n".join(tool_results)
            observations.append({
                "role": "environment",
                "content": observation_content
            })
            rewards.append(0.0)  # only outcome reward 
            terminateds.append(False)
            updated_metadata["current_turn"] += 1
            next_metadata.append(updated_metadata)
            next_stop_strings.append(["</tool_call>", "</answer>"])
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminateds_tensor = torch.tensor(terminateds, dtype=torch.bool)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        accuracy = batch["rewards"].mean().item()
        
        total_tool_calls = 0
        tool_usage = {"bash": 0, "file_create": 0, "file_edit": 0}
        
        if "extra_env_info" in batch:
            for info in batch["extra_env_info"]:
                if info and "tool_calls_count" in info:
                    for tool, count in info["tool_calls_count"].items():
                        if tool in tool_usage:
                            tool_usage[tool] += count
                        total_tool_calls += count
        
        metrics = {
            "accuracy": accuracy,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch.get("text", []), batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "avg_turns_per_problem": batch.get("generation_lengths", torch.zeros(1)).float().mean().item(),
            "total_tool_calls": total_tool_calls,
            **{f"tool_usage_{k}": v for k, v in tool_usage.items()},
        }
        
        return batch, metrics


def create_working_directory() -> str:
    """Create a temporary working directory for a problem."""
    return tempfile.mkdtemp(prefix="math_tools_") 
