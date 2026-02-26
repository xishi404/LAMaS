import asyncio
import json
import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple
from pydantic import BaseModel, Field
from lamas.actions.action_node import ActionNode
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from lamas.configs.models_config import ModelsConfig
from lamas.provider.llm_provider_registry import create_llm_instance
from lamas.logs import logger
from lamas.utils.common import write_json_file
from lamas.ext.lamas.scripts.utils import extract_random_prompt, update_prompt_in_file
from lamas.ext.lamas.scripts.textgrad.textual_gradient import TEXT_GRAD_PROMPT

class TextGrad(BaseModel):
    prompt: str = Field(default="", description="prompt")

class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
        token_weight: float = 0.00001,
        use_tokens: bool = False,
        virtual_token_rate: float = 50.0,
        use_critical_path: bool = True,
        parallel_execution: bool = True,
        normalize_rewards: bool = False,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.optimizer = optimizer
        self.token_weight = token_weight
        self.use_tokens = use_tokens
        self.virtual_token_rate = virtual_token_rate
        self.use_critical_path = use_critical_path
        self.parallel_execution = parallel_execution
        self.normalize_rewards = normalize_rewards

        # EMA statistics for reward normalization (running mean/std across batches)
        self.reward_ema_mean = None  # Running mean of utilities
        self.reward_ema_std = None   # Running std of utilities
        self.ema_momentum = 0.95     # Momentum for EMA updates (0.99 = slow adaptation)

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        avg_cost = df["cost"].mean() if "cost" in df.columns else 0.0
        avg_cp_token = df["cp_token"].mean() if "cp_token" in df.columns else 0.0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include penalty type in CSV filename
        penalty_suffix = ""
        if hasattr(self, 'use_tokens') and self.use_tokens:
            penalty_suffix = f"_tok{self.token_weight:.4f}".replace(".", "_")

        # Include parallel execution mode in CSV filename
        parallel_suffix = ""
        if hasattr(self, 'parallel_execution'):
            parallel_suffix = "_parallel" if self.parallel_execution else "_sequential"

        # Include critical path mode in CSV filename
        critical_path_suffix = ""
        if hasattr(self, 'use_critical_path') and self.use_critical_path:
            critical_path_suffix = "_cp"

        # Include normalization mode in CSV filename
        norm_suffix = ""
        if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
            norm_suffix = "_norm"

        filename = f"{avg_score:.5f}_{current_time}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{norm_suffix}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, avg_cost, avg_cp_token

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    def _compute_critical_path_loss_tokens(
        self,
        batch_layer_operator_infos: list,
        scores_tensor: torch.Tensor,
        costs_tensor: torch.Tensor
    ) -> tuple:
        """
        Critical path loss using ACTUAL TOKENS from critical path operators only.

        For each layer:
        - Identify max token operator (critical path) using cp_token_count
        - Only critical path operator gets penalized
        - Penalty uses sum of critical path tokens across all layers (not global token sum)
        """
        all_log_probs = []
        all_utilities = []

        for problem_idx in range(len(batch_layer_operator_infos)):
            layer_info = batch_layer_operator_infos[problem_idx]
            problem_score = scores_tensor[problem_idx].item()
            problem_cost = costs_tensor[problem_idx].item()

            log_probs_per_layer = layer_info['log_probs_per_layer']
            operator_token_counts_per_layer = layer_info['operator_token_counts_per_layer']

            # Calculate problem_tokens by summing only critical path tokens
            problem_tokens = 0.0
            critical_path_indices = []  # Track which operators are on critical path

            for layer_tokens in operator_token_counts_per_layer:
                if len(layer_tokens) == 0:
                    critical_path_indices.append(None)
                    continue
                # Find critical path operator (max tokens in this layer)
                max_tokens = max(layer_tokens)
                max_token_idx = layer_tokens.index(max_tokens)
                critical_path_indices.append(max_token_idx)
                problem_tokens += max_tokens

            # Assign utilities to each operator
            for layer_idx, (layer_log_probs, layer_tokens) in enumerate(zip(
                log_probs_per_layer, operator_token_counts_per_layer
            )):
                if len(layer_tokens) == 0:
                    continue

                max_token_idx = critical_path_indices[layer_idx]

                for op_idx in range(len(layer_tokens)):
                    if op_idx == max_token_idx:
                        # Critical path operator gets penalized by total critical path tokens
                        # Divide token_weight by 50 to align scale
                        utility = problem_score - 3 * problem_cost - (self.token_weight / 50.0) * problem_tokens
                    else:
                        # Non-critical operators don't get penalty
                        utility = problem_score - 3 * problem_cost

                    all_log_probs.append(layer_log_probs[op_idx])
                    all_utilities.append(utility)

        # Apply normalization and compute loss
        if len(all_log_probs) > 0:
            log_probs_tensor = torch.stack(all_log_probs)
            utilities_tensor = torch.tensor(all_utilities, dtype=torch.float32, device=self.device)

            # Store raw average utility before normalization for logging
            avg_utility = utilities_tensor.mean().item()

            # Apply EMA-based reward normalization if enabled
            if self.normalize_rewards:
                # Compute batch statistics
                batch_mean = utilities_tensor.mean().item()
                batch_std = utilities_tensor.std(unbiased=False).item()

                # Initialize EMA on first batch
                if self.reward_ema_mean is None:
                    self.reward_ema_mean = batch_mean
                    self.reward_ema_std = max(batch_std, 1e-8)
                else:
                    # Update EMA statistics
                    self.reward_ema_mean = self.ema_momentum * self.reward_ema_mean + (1 - self.ema_momentum) * batch_mean
                    self.reward_ema_std = self.ema_momentum * self.reward_ema_std + (1 - self.ema_momentum) * max(batch_std, 1e-8)

                utilities_tensor = (utilities_tensor - self.reward_ema_mean) / max(self.reward_ema_std, 1e-8)

            loss = -(log_probs_tensor * utilities_tensor).mean()
            return loss, avg_utility
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

    @abstractmethod
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 30, repetitions: int = 4, is_textgrad: bool = False):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []
        previous_cost = 0.0
        textgrad = False
        prev_rep_score = None

        # Track total tokens for training
        self.total_training_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        async def sem_evaluate(problem):
            async with semaphore:
                try:
                    return await self.evaluate_problem(problem, graph)
                except Exception as e:
                    logger.error(f"Error evaluating problem: {e}")
                    return ("", "", "", 0.0, 0.0, 0.0, 0.0)

        for rep in range(1, repetitions + 1):
            logger.info(f"Starting training repetition {rep}/{repetitions}")
            rep_scores = []
            total_batches_count = 0

            if textgrad and is_textgrad:
                prompt_name, prompt_content = extract_random_prompt(self.log_path)
                textgrad_prompt = TEXT_GRAD_PROMPT.format(dataset = self.name, prompt_name = prompt_name, prompt_content = prompt_content)
                textgrad_llm_config = ModelsConfig.default().get("gpt-4o-mini")
                textgrad_llm = create_llm_instance(textgrad_llm_config)
                textgrad_node = await ActionNode.from_pydantic(TextGrad).fill(context=textgrad_prompt, mode="xml_fill", llm=textgrad_llm)
                response = textgrad_node.instruct_content.model_dump()
                update_prompt_in_file(prompt_name, response["prompt"])
                is_textgrad = False

            for batch_start in range(0, len(data), self.batch_size):
                batch = data[batch_start:batch_start + self.batch_size]

                tasks = [sem_evaluate(problem) for problem in batch]
                batch_results = await tqdm_asyncio.gather(
                    *tasks,
                    desc=f"Repetition {rep}: Executing batch {batch_start // self.batch_size + 1}",
                    total=len(batch)
                )
                results.extend(batch_results)

                # Extract data from results
                batch_layer_operator_infos = []
                per_problem_logprobs = []
                scores = []
                costs = []
                cp_tokens = []
                output_tokens_list = []

                for r in batch_results:
                    score = float(r[3]) if r[3] is not None else 0.0
                    cost = float(r[4]) if r[4] is not None else 0.0
                    logprob = r[5]
                    cp_token = float(r[6]) if r[6] is not None else 0.0
                    layer_operator_info = r[7] if len(r) > 7 else None

                    # Calculate output tokens for this problem from layer_operator_info
                    if layer_operator_info is not None and 'operator_token_counts_per_layer' in layer_operator_info:
                        problem_tokens = sum(
                            sum(layer_tokens) for layer_tokens in layer_operator_info['operator_token_counts_per_layer']
                        )
                    else:
                        problem_tokens = 0.0

                    per_problem_logprobs.append(logprob)
                    scores.append(score)
                    costs.append(cost - previous_cost)
                    cp_tokens.append(cp_token)
                    output_tokens_list.append(problem_tokens)
                    batch_layer_operator_infos.append(layer_operator_info)
                    previous_cost = cost
                    rep_scores.append(score)

                # Accumulate tokens from the graph's cost manager after each batch
                if hasattr(graph, 'llm') and hasattr(graph.llm, 'cost_manager'):
                    cost_manager = graph.llm.cost_manager
                    self.total_prompt_tokens = cost_manager.get_total_prompt_tokens()
                    self.total_completion_tokens = cost_manager.get_total_completion_tokens()
                    self.total_training_cost = previous_cost

                if len(per_problem_logprobs) > 0:
                    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                    costs_tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
                    output_tokens_tensor = torch.tensor(output_tokens_list, dtype=torch.float32, device=self.device)

                    # Critical path is ONLY applicable for parallel execution mode
                    if self.use_tokens and self.use_critical_path and self.parallel_execution and all(info is not None for info in batch_layer_operator_infos):
                        loss, avg_utility = self._compute_critical_path_loss_tokens(
                            batch_layer_operator_infos,
                            scores_tensor,
                            costs_tensor
                        )
                    elif self.use_tokens:
                        # Standard virtual token penalty (no critical path)
                        logprobs = torch.stack(per_problem_logprobs).to(self.device)
                        cp_tokens_tensor = torch.tensor(cp_tokens, dtype=torch.float32, device=self.device)
                        utilities = scores_tensor - 3 * costs_tensor - (self.token_weight / 50.0) * cp_tokens_tensor

                        avg_utility = utilities.mean().item()

                        if self.normalize_rewards:
                            std = utilities.std(unbiased=False)
                            if std > 1e-8:
                                utilities = (utilities - utilities.mean()) / std
                            else:
                                utilities = utilities - utilities.mean()

                        loss = -(logprobs * utilities).mean()
                    else:
                        # No token penalty
                        logprobs = torch.stack(per_problem_logprobs).to(self.device)
                        utilities = scores_tensor - 3 * costs_tensor

                        avg_utility = utilities.mean().item()

                        if self.normalize_rewards:
                            std = utilities.std(unbiased=False)
                            if std > 1e-8:
                                utilities = (utilities - utilities.mean()) / std
                            else:
                                utilities = utilities - utilities.mean()

                        loss = -(logprobs * utilities).mean()

                    avg_batch_cost = costs_tensor.mean().item()
                    avg_batch_score = scores_tensor.mean().item()

                    if self.use_tokens and self.use_critical_path and self.parallel_execution:
                        cp_tokens_tensor = torch.tensor(cp_tokens, dtype=torch.float32, device=self.device)
                        avg_batch_cp_tokens = cp_tokens_tensor.mean().item()
                    else:
                        avg_batch_cp_tokens = output_tokens_tensor.mean().item()

                    if isinstance(per_problem_logprobs[0], torch.Tensor):
                        logprobs_tensor = torch.stack(per_problem_logprobs)
                    else:
                        logprobs_tensor = torch.tensor(per_problem_logprobs, device=self.device)
                    sum_logprobs = logprobs_tensor.sum().item()

                    if loss.requires_grad:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.use_tokens:
                            logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, CP_Tokens: {avg_batch_cp_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                        else:
                            logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss: {loss.item():.4f}, Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                    else:
                        if self.use_tokens:
                            logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, CP_Tokens: {avg_batch_cp_tokens:.1f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                        else:
                            logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss does not require grad and was skipped. Score: {avg_batch_score:.4f}, Cost: {avg_batch_cost:.6f}, Utility: {avg_utility:.4f}, SumLogProbs: {sum_logprobs:.4f}")
                else:
                    logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} skipped due to invalid logprobs.")

            if rep_scores:
                current_rep_score = sum(rep_scores) / len(rep_scores)
            else:
                current_rep_score = 0.0

            if not textgrad:
                if prev_rep_score is not None and current_rep_score < prev_rep_score:
                    textgrad = True
                prev_rep_score = current_rep_score

        return results

    async def evaluate_all_problems_test(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 10):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Initialize probability aggregation structures
        self.prob_aggregator = {}  # {layer_idx: {operator_name: [probabilities]}}

        # Initialize operator iteration aggregation structures
        self.operator_iteration_aggregator = {}  # {operator_name: [iteration_counts]}

        async def sem_evaluate(problem):
            async with semaphore:
                result = await self.evaluate_problem(problem, graph)

                # Collect probabilities from the graph if available
                if hasattr(graph, 'last_probs_layers'):
                    for layer_idx, probs in enumerate(graph.last_probs_layers):
                        if layer_idx not in self.prob_aggregator:
                            self.prob_aggregator[layer_idx] = {}

                        if hasattr(graph, 'selection_operator_names'):
                            probs_cpu = probs.detach().cpu().numpy()
                            for op_idx, op_name in enumerate(graph.selection_operator_names):
                                if op_name not in self.prob_aggregator[layer_idx]:
                                    self.prob_aggregator[layer_idx][op_name] = []
                                self.prob_aggregator[layer_idx][op_name].append(float(probs_cpu[op_idx]))

                # Collect operator iterations from the graph if available
                if hasattr(graph, 'last_operator_iterations'):
                    for op_name, iterations in graph.last_operator_iterations.items():
                        if op_name not in self.operator_iteration_aggregator:
                            self.operator_iteration_aggregator[op_name] = []
                        self.operator_iteration_aggregator[op_name].extend(iterations)

                return result

        tasks = [sem_evaluate(problem) for problem in data]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

        # Convert results to format expected by CSV
        # CSV format: (input_text, prediction, expected_output, score, cost, logprob, cp_token) - 7 columns
        processed_results = []
        for r in results:
            # r format: (input_text, prediction, expected_output, score, cost, logprob, cp_token, layer_operator_info)
            if len(r) >= 8:
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
            elif len(r) >= 7:
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
            else:
                processed_result = (r[0], r[1], r[2], r[3], r[4], r[5], 0.0)
            processed_results.append(processed_result)

        return processed_results

    async def run_evaluation(self, graph: Callable, va_list: List[int], is_test: bool, sample: int, is_textgrad: bool = False, max_concurrent_tasks: int = 30):
        data = await self.load_data(va_list)

        if is_test == True:
            results = await self.evaluate_all_problems_test(data, graph, max_concurrent_tasks)
            columns = self.get_result_columns()
            average_score, average_cost, average_cp_token = self.save_results_to_csv(results, columns)
            logger.info(f"Average score on {self.name} dataset: {average_score:.5f}, Average cost: {average_cost:.6f}, Avg cp_token: {average_cp_token:.2f}")

            # Track token usage and cost for testing
            if hasattr(graph, 'llm') and hasattr(graph.llm, 'cost_manager'):
                cost_manager = graph.llm.cost_manager
                self.test_prompt_tokens = cost_manager.get_total_prompt_tokens()
                self.test_completion_tokens = cost_manager.get_total_completion_tokens()
                self.test_total_tokens = self.test_prompt_tokens + self.test_completion_tokens
                self.test_total_cost = cost_manager.get_total_cost()
                logger.info(f"Test token usage - Prompt: {self.test_prompt_tokens:,}, Completion: {self.test_completion_tokens:,}, Total: {self.test_total_tokens:,}")
                logger.info(f"Test total cost: ${self.test_total_cost:.6f}")

            return average_score

        results = await self.evaluate_all_problems(data, graph, max_concurrent_tasks, sample, is_textgrad)

        columns = self.get_result_columns()
        # Filter out layer_operator_info before saving to CSV
        # Keep first 7 elements: problem, prediction, expected_output, score, cost, logprob, cp_token
        results_for_csv = [result[:7] if len(result) >= 7 else result for result in results]
        average_score, average_cost, average_cp_token = self.save_results_to_csv(results_for_csv, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}, Average cost: {average_cost:.6f}, Avg cp_token: {average_cp_token:.2f}")

        try:
            os.makedirs(self.log_path, exist_ok=True)
            penalty_suffix = ""
            if hasattr(self, 'use_tokens') and self.use_tokens:
                penalty_suffix = f"_tok{self.token_weight:.4f}".replace(".", "_")
            parallel_suffix = ""
            if hasattr(self, 'parallel_execution'):
                parallel_suffix = "_parallel" if self.parallel_execution else "_sequential"
            critical_path_suffix = ""
            if hasattr(self, 'use_critical_path') and self.use_critical_path:
                critical_path_suffix = "_cp"
            norm_suffix = ""
            if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
                norm_suffix = "_norm"
            controller_path = os.path.join(self.log_path, f"{self.name}_controller_sample{sample}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{norm_suffix}.pth")
            torch.save(self.controller.state_dict(), controller_path)
            logger.info(f"Saved controller parameters to {controller_path}")
            logger.info("Successfully Finish Training")
        except Exception as e:
            logger.error(f"Failed to save controller parameters: {e}")

        return average_score
