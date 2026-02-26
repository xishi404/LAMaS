import asyncio
import torch
import lamas.ext.lamas.scripts.optimized.HumanEval.train.template.prompt as prompt_custom
import lamas.ext.lamas.scripts.optimized.HumanEval.train.template.operator as operator
from lamas.ext.lamas.scripts.optimized.HumanEval.train.template.operator_registry import operator_mapping, operator_names
from lamas.provider.llm_provider_registry import create_llm_instance
from lamas.utils.cost_manager import CostManager
from lamas.logs import logger

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
        parallel_execution: bool = True,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.test_operator = operator.Test(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
        self.parallel_execution = parallel_execution

    def _log_operator_probabilities(self, probs_layers, selected_names_layers, log_path, problem):
        """Log the probability distribution for each operator in each layer"""
        logger.info(f"\n{'='*80}")
        logger.info("OPERATOR SELECTION PROBABILITIES")
        logger.info(f"{'='*80}")

        for layer_idx, probs in enumerate(probs_layers):
            logger.info(f"\nLayer {layer_idx + 1}:")
            logger.info(f"Selected operators: {', '.join(selected_names_layers[layer_idx])}")
            logger.info(f"{'Operator':<25} {'Probability':<15} {'Selected':<10}")
            logger.info("-" * 50)

            probs_cpu = probs.detach().cpu().numpy()
            for op_idx, op_name in enumerate(self.selection_operator_names):
                prob_value = probs_cpu[op_idx]
                is_selected = "✓" if op_name in selected_names_layers[layer_idx] else ""
                logger.info(f"{op_name:<25} {prob_value:>6.4f} ({prob_value*100:>5.2f}%)  {is_selected:<10}")

        logger.info(f"{'='*80}\n")

    async def __call__(self, problem: str, entry_point: str, log_path: str):
        # Run controller to select operators for all layers (single forward pass)
        log_probs_layers, selected_names_layers, probs_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)

        # Store probabilities for aggregation (will be used by benchmark)
        self.last_probs_layers = probs_layers

        # Log probability distributions for each layer (only during testing, not training to reduce noise)
        # Uncomment the following lines if you want to see probabilities during training too
        # if log_path:
        #     self._log_operator_probabilities(probs_layers, selected_names_layers, log_path, problem)

        # Decompose layer log probs into per-operator log probs for critical path tracking
        log_probs_per_layer = []  # List[torch.Tensor] - individual operator log probs
        operator_names_per_layer = []  # List[List[str]]

        for layer_idx, (selected_names, probs) in enumerate(zip(selected_names_layers, probs_layers)):
            if not selected_names:
                log_probs_per_layer.append(torch.tensor([], device=self.device))
                operator_names_per_layer.append([])
                continue

            # Extract individual log probs from the probability distribution
            log_probs_full = torch.log(probs + 1e-10)
            selected_indices = [self.selection_operator_names.index(name) for name in selected_names]
            individual_log_probs = log_probs_full[selected_indices]

            log_probs_per_layer.append(individual_log_probs)
            operator_names_per_layer.append(selected_names)

        current_solution = ""
        solutions = []
        sum_log_prob = 0.0

        # Track per-operator iteration and token counts
        operator_iterations = {}  # {operator_name: [iteration_counts]}
        operator_token_counts = {}  # {operator_name: [token_counts]}

        # Track per-layer operator token counts for virtual token calculation
        operator_token_counts_per_layer = []  # List[List[float]] - [num_layers][num_ops_in_layer]

        for layer_idx, selected_names in enumerate(selected_names_layers):
            # All operators in the same layer are independent and can run in parallel
            # They all receive the same state from previous layers
            if not selected_names:
                continue

            if self.parallel_execution:
                # PARALLEL MODE: Run all operators in the layer in parallel
                async def execute_operator(op_name):
                    """Execute a single operator with the current state and track tokens"""
                    selected_operator = self.selection_operator_instances[op_name]

                    if op_name in ["Generate", "GenerateCoT"]:
                        result = await selected_operator(problem=problem, entry_point=entry_point, instruction="", return_usage=True)
                        cp_token_count = result.get('_usage_tokens', 0)
                        return {"type": "generate", "solution": result.get('response', ""), "operator": op_name, "iterations": 1, "cp_token_count": cp_token_count}
                    elif op_name == "MultiGenerateCoT":
                        result = await selected_operator(problem=problem, entry_point=entry_point, instruction="", return_usage=True)
                        raw_tokens = result.get('_usage_tokens', 0)
                        if isinstance(result, dict) and 'response' in result:
                            num_iterations = len(result['response']) if isinstance(result['response'], list) else 1
                            # For parallel CoT runs, divide by num_iterations to get effective tokens
                            cp_token_count = raw_tokens / num_iterations if num_iterations > 0 else raw_tokens
                            return {"type": "multi_generate", "solutions": [res.get('response', "") for res in result['response']], "operator": op_name, "iterations": num_iterations, "cp_token_count": cp_token_count}
                        else:
                            logger.error(f"Expected dict with 'response' from MultiGenerateCoT, got {type(result)}")
                            return {"type": "multi_generate", "solutions": [], "operator": op_name, "iterations": 0, "cp_token_count": raw_tokens}
                    elif op_name == "SelfRefine":
                        result = await selected_operator(problem=problem, solution=current_solution, return_usage=True)
                        cp_token_count = result.get('_usage_tokens', 0)
                        return {"type": "refine", "solution": result.get('response', ""), "operator": op_name, "iterations": 1, "cp_token_count": cp_token_count}
                    elif op_name == "Test":
                        result = await selected_operator(problem=problem, solution=current_solution, entry_point=entry_point, return_usage=True)
                        # For Test: LLM tokens + virtual tokens for tool execution
                        llm_tokens = result.get('_usage_tokens', 0)
                        tool_exec_time = result.get('tool_exec_time', 0.0)
                        cp_token_count = llm_tokens + tool_exec_time * 50.0  # 50 tokens/sec as virtual cost
                        return {"type": "test", "solution": result.get('solution', ""), "operator": op_name, "iterations": 1, "cp_token_count": cp_token_count}
                    elif op_name == "ScEnsemble":
                        result = await selected_operator(problem=problem, solutions=solutions, return_usage=True)
                        cp_token_count = result.get('_usage_tokens', 0)
                        return {"type": "ensemble", "solution": result.get('response', ""), "operator": op_name, "iterations": 1, "cp_token_count": cp_token_count}
                    else:
                        # EarlyStop or other operators
                        cp_token_count = 0  # No tokens for non-LLM operators
                        return {"type": "noop", "solution": current_solution, "operator": op_name, "iterations": 0, "cp_token_count": cp_token_count}

                # Run all operators in the layer in parallel
                results = await asyncio.gather(*[execute_operator(op_name) for op_name in selected_names])

                # Track token counts for this layer in the same order as selected_names
                layer_token_counts = []
                for op_name in selected_names:
                    # Find the result for this operator
                    op_result = next((r for r in results if r.get("operator") == op_name), None)
                    if op_result:
                        layer_token_counts.append(op_result.get("cp_token_count", 0))
                    else:
                        layer_token_counts.append(0)
                operator_token_counts_per_layer.append(layer_token_counts)

                # Process results and update state
                for result in results:
                    op_name = result.get("operator", "Unknown")
                    op_iterations = result.get("iterations", 1)
                    op_tokens = result.get("cp_token_count", 0)

                    if op_name not in operator_iterations:
                        operator_iterations[op_name] = []
                        operator_token_counts[op_name] = []
                    operator_iterations[op_name].append(op_iterations)
                    operator_token_counts[op_name].append(op_tokens)

                    if result["type"] == "generate":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "multi_generate":
                        solutions.extend(result["solutions"])
                        if result["solutions"]:
                            current_solution = result["solutions"][-1]
                    elif result["type"] == "refine":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "test":
                        solutions.append(result["solution"])
                        current_solution = result["solution"]
                    elif result["type"] == "ensemble":
                        solutions = [result["solution"]]
                        current_solution = result["solution"]

            else:
                # SEQUENTIAL MODE: Run operators one by one (original behavior)
                layer_token_counts = []

                for op_name in selected_names:
                    selected_operator = self.selection_operator_instances[op_name]

                    if op_name in ["Generate", "GenerateCoT"]:
                        result = await selected_operator(problem=problem, entry_point=entry_point, instruction="", return_usage=True)
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                    elif op_name == "SelfRefine":
                        result = await selected_operator(problem=problem, solution=current_solution, return_usage=True)
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                    elif op_name == "Test":
                        result = await selected_operator(problem=problem, solution=current_solution, entry_point=entry_point, return_usage=True)
                        new_solution = result.get('solution', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                    elif op_name == "ScEnsemble":
                        result = await selected_operator(problem=problem, solutions=solutions, return_usage=True)
                        solutions = []
                        new_solution = result.get('response', "")
                        solutions.append(new_solution)
                        current_solution = new_solution
                    elif op_name == "MultiGenerateCoT":
                        result = await selected_operator(problem=problem, entry_point=entry_point, instruction="", return_usage=True)
                        if isinstance(result, dict) and 'response' in result:
                            num_iterations = len(result['response']) if isinstance(result['response'], list) else 1
                            for res in result['response']:
                                new_solution = res.get('response', "")
                                solutions.append(new_solution)
                            current_solution = new_solution
                        else:
                            logger.error(f"Expected dict with 'response' from MultiGenerateCoT, got {type(result)}")
                            new_solution = current_solution
                    else:
                        new_solution = current_solution

                    # Track tokens for this operator
                    raw_tokens = result.get('_usage_tokens', 0) if isinstance(result, dict) else 0

                    op_iterations = 1
                    if op_name == "MultiGenerateCoT" and isinstance(result, dict) and 'response' in result:
                        op_iterations = len(result['response']) if isinstance(result['response'], list) else 1

                    # For MultiGenerateCoT, divide tokens by iterations to get effective tokens
                    cp_token_count = raw_tokens / op_iterations if op_iterations > 0 else raw_tokens

                    # For Test operator, add virtual tokens for tool execution time
                    if op_name == "Test" and isinstance(result, dict):
                        tool_exec_time = result.get('tool_exec_time', 0.0)
                        cp_token_count += tool_exec_time * 50.0  # 50 tokens/sec as virtual cost

                    if op_name not in operator_iterations:
                        operator_iterations[op_name] = []
                        operator_token_counts[op_name] = []
                    operator_iterations[op_name].append(op_iterations)
                    operator_token_counts[op_name].append(cp_token_count)

                    # Track tokens for this layer
                    layer_token_counts.append(cp_token_count)

                # Store layer token counts for critical path tracking
                operator_token_counts_per_layer.append(layer_token_counts)

            sum_log_prob += log_probs_layers[layer_idx]

        # Post-processing: Test and potentially improve
        test_result = await self.test_operator(problem=problem, solution=current_solution, entry_point=entry_point, return_usage=True)

        # Track test operator tokens
        if "Test" not in operator_iterations:
            operator_iterations["Test"] = []
            operator_token_counts["Test"] = []
        operator_iterations["Test"].append(1)

        # Calculate tokens for Test: LLM tokens + virtual tokens for tool execution
        llm_tokens = test_result.get('_usage_tokens', 0)
        tool_exec_time = test_result.get('tool_exec_time', 0.0)
        test_tokens = llm_tokens + tool_exec_time * 50.0  # 50 tokens/sec as virtual cost
        operator_token_counts["Test"].append(test_tokens)

        if test_result['result']:
            final_solution = test_result['solution']
        else:
            new_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.IMPROVE_CODE_PROMPT, return_usage=True)
            final_solution = new_solution['response']

            # Track improve operator tokens
            if "ImproveCode" not in operator_iterations:
                operator_iterations["ImproveCode"] = []
                operator_token_counts["ImproveCode"] = []
            operator_iterations["ImproveCode"].append(1)
            improve_tokens = new_solution.get('_usage_tokens', 0)
            operator_token_counts["ImproveCode"].append(improve_tokens)

        # Store operator iteration statistics in graph for later aggregation
        self.last_operator_iterations = operator_iterations

        # Build layer operator info structure for critical path tracking
        layer_operator_info = {
            'log_probs_per_layer': log_probs_per_layer,
            'operator_names_per_layer': operator_names_per_layer,
            'operator_token_counts_per_layer': operator_token_counts_per_layer,
        }

        # Calculate total virtual tokens across all layers
        total_virtual_tokens = 0.0
        for layer_tokens in operator_token_counts_per_layer:
            if layer_tokens:
                if self.parallel_execution:
                    # In parallel mode, count max tokens per layer (critical path)
                    total_virtual_tokens += max(layer_tokens)
                else:
                    # In sequential mode, sum all tokens in layer
                    total_virtual_tokens += sum(layer_tokens)

        # Add post-processing tokens (Test and ImproveCode are not in operator_token_counts_per_layer)
        # The post-processing Test is always the last entry in operator_token_counts["Test"]
        if "Test" in operator_token_counts and operator_token_counts["Test"]:
            total_virtual_tokens += operator_token_counts["Test"][-1]
        # ImproveCode only runs if test failed
        if "ImproveCode" in operator_token_counts and operator_token_counts["ImproveCode"]:
            total_virtual_tokens += operator_token_counts["ImproveCode"][-1]

        return final_solution, self.llm.cost_manager.total_cost, sum_log_prob, total_virtual_tokens, layer_operator_info
