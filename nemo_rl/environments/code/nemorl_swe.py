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
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch
import os
import subprocess
from openai import OpenAI #FIXME: use policy
import ast
import re
import tempfile

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

# insert bug with llm
# FIXME: use policy
def generate_bug_with_llm(file_content, bug_type):
    """
    Generate a bug in the provided code using NVDev endpoint. 
    
    Args:
        file_content (str): The content of the file to inject a bug into
        bug_type (str): The type of bug to inject
        
    Returns:
        str: Git diff format of the bug change
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVAPI_KEY")
    )
    
    prompt = f"Inject a {bug_type} bug into the following code: {file_content}. Return the git diff only."
    
    completion = client.chat.completions.create(
        model="nvdev/meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024
    )
    
    return completion.choices[0].message.content

# view file tool 
def view_file_folded(filename):
    """
    cats a file with collapsed/folded functions 

    Args:
        filename (str): The name of the Python file to process.
    """

    with open(filename, 'r') as file:
        content = file.read()

    tree = ast.parse(content)
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    collapsed_content = content
    for func in function_defs:
        func_name = func.name
        start_line = func.lineno
        end_line = func.body[-1].end_lineno if func.body else func.lineno

        function_body = '\n'.join(content.splitlines()[start_line - 1:end_line])

        params = []
        for arg in func.args.args:
            params.append(arg.arg)
        
        if func.args.defaults:
            non_default_count = len(func.args.args) - len(func.args.defaults)
            for i, default in enumerate(func.args.defaults):
                arg_idx = non_default_count + i
                if arg_idx < len(params):
                    params[arg_idx] = f"{params[arg_idx]}={ast.unparse(default)}"
        
        params_str = ", ".join(params)

        collapsed_representation = f"def {func_name}({params_str}): # collapsed body"

        function_pattern = re.compile(re.escape(function_body), re.DOTALL)
        collapsed_content = function_pattern.sub(collapsed_representation, collapsed_content, count=1)

    print(collapsed_content)
    
# view function tool
def view_function(filename, function_name):
    """
    Extracts and prints a specific function from a Python file.
    
    Args:
        filename (str): The path to the Python file.
        function_name (str): The name of the function to extract and print.
    """
    with open(filename, 'r') as file:
        content = file.read()
    
    tree = ast.parse(content)
    
    function_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_node = node
            break
    
    if function_node is None:
        print(f"Function '{function_name}' not found in {filename}")
        return
    
    start_line = function_node.lineno
    end_line = function_node.body[-1].end_lineno if function_node.body else function_node.lineno
    
    function_code = '\n'.join(content.splitlines()[start_line - 1:end_line])
    
    print(f"Function '{function_name}' from {filename}:")
    print("-" * 50)
    print(function_code)
    print("-" * 50)
    

# apply patch tool
def apply_patch(file_path: str, patch_content: str) -> bool:
    """
    Apply a git-style patch to a file.
    
    Args:
        file_path (str): Path to the file to be patched
        patch_content (str): The git-style patch content
        
    Returns:
        bool: True if patch was applied successfully, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False
    
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        original_lines = original_content.splitlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    
    patch_lines = patch_content.splitlines()
    
    hunks = []
    current_hunk = None
    
    for line in patch_lines:
        # Look for hunk headers like @@ -1,7 +1,6 @@
        hunk_header_match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        
        if hunk_header_match:
            if current_hunk is not None:
                hunks.append(current_hunk)
            
            start_line = int(hunk_header_match.group(1))
            start_count = int(hunk_header_match.group(2) or 1)
            target_line = int(hunk_header_match.group(3))
            target_count = int(hunk_header_match.group(4) or 1)
            
            current_hunk = {
                'start_line': start_line,
                'start_count': start_count,
                'target_line': target_line,
                'target_count': target_count,
                'content': [],
                'header': line
            }
        elif current_hunk is not None:
            current_hunk['content'].append(line)
    
    if current_hunk is not None:
        hunks.append(current_hunk)
    
    new_lines = original_lines.copy()
    offset = 0  
    
    for hunk in hunks:
        # Adjust target line with offset
        target_line = hunk['target_line'] + offset - 1  # 0-indexed
        
        # Parse hunk content
        added_lines = []
        removed_line_count = 0
        
        for line in hunk['content']:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                removed_line_count += 1
        
        # Remove the lines that are being replaced
        del new_lines[target_line:target_line + removed_line_count]
        
        # Insert the new lines
        for i, line in enumerate(added_lines):
            new_lines.insert(target_line + i, line)
        
        # Update offset for next hunk
        offset += len(added_lines) - removed_line_count
    
    try:
        with open(file_path, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Successfully applied patch to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False


class NemoRLSWEDatasetConfig(TypedDict):
    repo_url: str
    
class NemoRLSWEDatasetMetadata(TypedDict):
    task_state: Dict[str, Any]
    num_tool_calls: int
    max_tool_calls: int
    

class NemoRLSWETaskLogic:
    @staticmethod
    def generate(config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new task state including an injected bug, failing tests."""
        repo_url = config["repo_url"]
        
        unique_id = str(uuid.uuid4())[:8]
        repo_path = os.path.join(tempfile.gettempdir(), f"nemo_rl_swe_{unique_id}")
        clone_command = f"git clone {repo_url} {repo_path}"
        subprocess.run(clone_command, shell=True, check=True)
        
        test_cases_and_files = { "tests/unit/algorithms/test_grpo.py": ["nemo_rl/experience/rollouts.py"]}
        random_test_case = random.choice(list(test_cases_and_files.keys()))
        random_file = random.choice(test_cases_and_files[random_test_case])
        
        # Run the test case
        test_command = f"cd {repo_path} && export HF_TOKEN={os.environ.get('HF_TOKEN')} && pytest {random_test_case} -v"
        result = subprocess.run(test_command, shell=True, check=True)
        
        # Make sure the test passed
        # TODO: instead of assert, we should try a different test case if this one fails
        assert result.returncode == 0
        
        # Inject bug with llm  
        bug_types = ['off_by_one', 'null_pointer', 'type_error', 'logic_error', 'api_misuse', 'syntax_error']
        bug_type = random.choice(bug_types)
        file_content = open(os.path.join(repo_path, random_file), "r").read()
        
        # Use the new abstracted function to generate the bug
        git_diff = generate_bug_with_llm(file_content, bug_type)
        
        # validate and apply the git diff
        apply_patch(os.path.join(repo_path, random_file), git_diff)
        
        # Run the test case again
        test_command = f"cd {repo_path} && export HF_TOKEN={os.environ.get('HF_TOKEN')} && pytest {random_test_case} -v"
        result = subprocess.run(test_command, shell=True, check=True)
        
        # Make sure the test failed
        # TODO: instead of assert, we should try inserting a different bug 
        assert result.returncode != 0
        
        available_tools = ["view_directory()", "view_file(file_path)", "view_function(file_path, function_name)", "apply_patch(file_path, patch_content)"]
        
        # Create and return the task state
        return {
            "repo_path": repo_path,
            "test_case": random_test_case,
            "test_cases_and_files": test_cases_and_files,
            "available_tools": available_tools
        }
        
    @staticmethod
    def init(task_state: Dict[str, Any]) -> str:
        """Initialize the task state and return welcome message."""
        repo_path = task_state["repo_path"]
        test_case = task_state["test_case"]
        test_cases_and_files = task_state["test_cases_and_files"]
        available_tools = task_state["available_tools"]
        

        return (
            f"You are a software engineer tasked with fixing a bug in the codebase. "
            f"The test case is failing: {test_case}. "
            f"You should fix one or more of the following files: {test_cases_and_files[test_case]}. "
            f"You can use the following tools to help you: {available_tools}"
            f"You must format the tool as <action> view_directory() </action> and similar for other tools."
            f"Do not output anything after the action. Keep reasoning before the action to a few sentences max."
        )
    
    @staticmethod
    def step(action: str, task_state: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Process an action in the task state."""
        repo_path = task_state["repo_path"]
        test_case = task_state["test_case"]
        
        # Deep copy task state to avoid modifying the original
        new_state = copy.deepcopy(task_state)
        
        response = "Unknown command."
        reward = 0.0
        is_terminated = False
        
        # possible actions (maybe change these as we go, not sure): 
        # view_directory(): Show directory structure
        # view_file(file_path): View file content with line numbers and collapsed functions
        # view_function(file_path, function_name): View a specific function
        # apply_patch(file_path, patch_content): Apply a git-style patch
        
        if action.startswith("view_directory"):
            path = repo_path
            if "(" in action and ")" in action:
                path_arg = action.split("(")[1].split(")")[0].strip()
                if path_arg:
                    path = os.path.join(repo_path, path_arg)
            
            command = f"ls {path}/**"
            contents = subprocess.run(command, shell=True, capture_output=True, text=True)
            response = f"Directory contents: {contents.stdout}"
        
        elif action.startswith("view_file"):
            if "(" in action and ")" in action:
                file_path = action.split("(")[1].split(")")[0].strip()
                file_path = os.path.join(repo_path, file_path)
                
                if os.path.exists(file_path):
                    contents = view_file_folded(file_path)
                    response = f"File contents of {file_path}"
                else:
                    response = f"Error: File {file_path} does not exist"
            else:
                response = "Error: Invalid view_file command format"
        
        elif action.startswith("view_function"):
            if "(" in action and ")" in action:
                params = action.split("(")[1].split(")")[0].strip()
                if "," in params:
                    file_path, function_name = [p.strip() for p in params.split(",", 1)]
                    file_path = os.path.join(repo_path, file_path)
                    
                    if os.path.exists(file_path):
                        view_function(file_path, function_name)
                        response = f"Function {function_name} from {file_path}"
                    else:
                        response = f"Error: File {file_path} does not exist"
                else:
                    response = "Error: Missing function name parameter"
            else:
                response = "Error: Invalid view_function command format"
            
        elif action.startswith("apply_patch"):
            if "(" in action and ")" in action:
                params = action.split("(")[1].split(")")[0].strip()
                if "," in params:
                    file_path, patch_content = [p.strip() for p in params.split(",", 1)]
                    file_path = os.path.join(repo_path, file_path)
                    
                    success = apply_patch(file_path, patch_content)
                    if success:
                        # Run the test case to see if the patch fixed the issue
                        test_command = f"cd {repo_path} && export HF_TOKEN={os.environ.get('HF_TOKEN')} && pytest {test_case} -v"
                        result = subprocess.run(test_command, shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            response = f"Success! The patch fixed the failing test {test_case}."
                            reward = 1.0
                            is_terminated = True
                        else:
                            response = f"Patch applied to {file_path}, but test {test_case} still fails:\n{result.stdout}"
                    else:
                        response = f"Error: Failed to apply patch to {file_path}"
                else:
                    response = "Error: Missing patch content parameter"
            else:
                response = "Error: Invalid apply_patch command format"

        return response, reward, is_terminated, new_state
    
    @staticmethod
    def render(task_state: Dict[str, Any]) -> str:
        """Render the task state."""
        #FIXME: Do we need this? We have a repo which is always same, a failing test and a list of files to fix 
        # we could just render the failing test and potential files to fix
        failing_test = task_state["test_case"]
        potential_files_to_fix = task_state["test_cases_and_files"][failing_test]
        return f"Failing test: {failing_test}\nPotential files to fix: {potential_files_to_fix}"

class NemoRLSWERunner:
    def __init__(self):
        pass

    def _parse_action(self, text: str) -> Optional[str]:
        """Parses the action from tool calling format."""
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
    
    def process_turn(self, message_log: LLMMessageLogType, metadata: NemoRLSWEDatasetConfig) -> Tuple[
        Dict[str, str],
        float,
        bool,
        Optional[List[str]],
        Optional[NemoRLSWEDatasetMetadata],]:
        """Processes a single turn for the NemoRLSWEDataset, where a turn is a single tool call or action from LLM."""
        task_state = metadata["task_state"]
        num_tool_calls = metadata["num_tool_calls"]
        max_tool_calls = metadata["max_tool_calls"]

        turn_reward = 0.0
        is_terminated = False
        next_stop_strings = ["</action>"]
        next_metadata = metadata.copy()
        next_observation_content = ""   

        # Check if max tool calls reached
        if num_tool_calls >= max_tool_calls:
            is_terminated = True
            next_observation_content = (
                f"<error>Maximum tool calls ({max_tool_calls}) reached.</error>"
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
            next_observation_content = f"<environment>\n{task_state}\n\nInvalid response format no action taken. Try <action></action> like this: <action>your_action</action></environment>"
            next_metadata = None
        else:
            # Execute the action
            step_response, reward, is_terminated, next_task_state = NemoRLSWETaskLogic.step(parsed_action, task_state)
            turn_reward = reward
            is_terminated = is_terminated
            next_metadata["task_state"] = next_task_state
            next_metadata["num_tool_calls"] = num_tool_calls + 1
            
            next_observation_content = f"<environment>\n{step_response}\n</environment>"
            
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
class NemoRLSWEEnv(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    """NemoRLSWE environment (Ray Actor)."""

    def __init__(self, cfg: Optional[NemoRLSWEDatasetConfig] = None):
        self.task_config = cfg.get("task_config", {}) if cfg else {}
        self.runner = NemoRLSWERunner()

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[NemoRLSWEDatasetMetadata],
    ) -> EnvironmentReturn:
        """Processes a batch of NemoRLSWE interactions."""
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
        # Calculate success rate based on final reward == 1.0
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        # Could also calculate average number of moves for successful episodes, etc.
        return batch, {"nemo_rl_swe_success_rate": success_rate}    

