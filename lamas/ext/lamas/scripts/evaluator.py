from typing import Dict, Literal, Tuple
from lamas.ext.lamas.benchmark.benchmark import BaseBenchmark
from lamas.ext.lamas.benchmark.gsm8k import GSM8KBenchmark
from lamas.ext.lamas.benchmark.humaneval import HumanEvalBenchmark
from lamas.ext.lamas.benchmark.math import MATHBenchmark

DatasetType = Literal["HumanEval", "GSM8K", "MATH", "GAIA"]


class Evaluator:
    def __init__(self, eval_path: str, batch_size: int):
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ):
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]

        benchmark = benchmark_class(
            name=dataset,
            file_path=data_path,
            log_path=path,
            batch_size=self.batch_size,
            controller=params["controller"],
            operator_embeddings=params["operator_embeddings"],
            optimizer=params["optimizer"],
            token_weight=params.get("token_weight", 0.00001),
            use_tokens=params.get("use_tokens", False),
            virtual_token_rate=params.get("virtual_token_rate", 50.0),
            use_critical_path=params.get("use_critical_path", True),
            parallel_execution=params.get("parallel_execution", True),
            normalize_rewards=params.get("normalize_rewards", False),
        )
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None
        else:
            va_list = None
        score = await benchmark.run_evaluation(configured_graph, va_list, is_test, params["sample"], params["is_textgrad"])

        # Return both score and benchmark instance so caller can access training metrics
        return score, benchmark

    async def _configure_graph(self, dataset, graph, params: dict):
        controller = params.get("controller")
        operator_embeddings = params.get("operator_embeddings")
        llm_config = params.get("execute_llm_config")
        dataset_config = params.get("dataset")
        parallel_execution = params.get("parallel_execution", True)
        configured_graph = graph(
            name=dataset,
            llm_config=llm_config,
            dataset=dataset_config,
            controller=controller,
            operator_embeddings=operator_embeddings,
            parallel_execution=parallel_execution,
        )
        return configured_graph

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"lamas/ext/lamas/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_train.jsonl" 
