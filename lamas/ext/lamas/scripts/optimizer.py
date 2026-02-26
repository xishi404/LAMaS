import asyncio
import glob
import os
import time
import torch
import numpy as np
from typing import List, Literal

from pydantic import BaseModel, Field
from lamas.ext.lamas.scripts.evaluator import DatasetType
from lamas.ext.lamas.scripts.optimizer_utils.data_utils import DataUtils               
from lamas.ext.lamas.scripts.optimizer_utils.experience_utils import ExperienceUtils
from lamas.ext.lamas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from lamas.ext.lamas.scripts.optimizer_utils.graph_utils import GraphUtils           
from lamas.logs import logger
from lamas.ext.lamas.models.utils import get_sentence_embedding
from lamas.ext.lamas.models.controller import MultiLayerController

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]

class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str = None,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        is_textgrad: bool = False,
        token_weight: float = 0.00001,
        use_tokens: bool = False,
        virtual_token_rate: float = 50.0,
        use_critical_path: bool = True,
        code_path: str = None,
        parallel_execution: bool = True,
        normalize_rewards: bool = False,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.dataset = dataset
        self.type = question_type
        self.graph = None
        self.operators = operators
        self.root_path = f"{optimized_path}/{self.dataset}"
        # Separate code path for templates/graph files (defaults to same as root_path for backward compatibility)
        self.code_path = f"{code_path}/{self.dataset}" if code_path else self.root_path
        self.sample = sample
        self.top_scores = []
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.is_textgrad = is_textgrad
        self.token_weight = token_weight
        self.use_tokens = use_tokens
        self.virtual_token_rate = virtual_token_rate
        self.use_critical_path = use_critical_path
        self.parallel_execution = parallel_execution
        self.normalize_rewards = normalize_rewards
        self.graph_utils = GraphUtils(self.root_path, self.code_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.controller = MultiLayerController(device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)          

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(self.test())
            return None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        retry_count = 0
        max_retries = 1
        round = 1

        while retry_count < max_retries:
            try:
                score = loop.run_until_complete(self._optimize_graph_lamas())
                break
            except Exception as e:
                import traceback
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                logger.info(f"Full traceback:\n{traceback.format_exc()}")
                if retry_count == max_retries:
                    logger.info("Max retries reached. Moving to next round.")
                    score = None

                wait_time = 5 * retry_count
                time.sleep(wait_time)

            if retry_count < max_retries: 
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        logger.info(f"Score for round {round}: {score}")
        round += 1
        
        time.sleep(5)

    async def _optimize_graph_lamas(self):
        graph_path = f"{self.root_path}/train"
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_lamas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        logger.info(directory)

        self.graph = self.graph_utils.load_graph_lamas(graph_path)

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": self.is_textgrad,
            "token_weight": self.token_weight,
            "use_tokens": self.use_tokens,
            "virtual_token_rate": self.virtual_token_rate,
            "use_critical_path": self.use_critical_path,
            "parallel_execution": self.parallel_execution,
            "normalize_rewards": self.normalize_rewards,
        }

        avg_score = await self.evaluation_utils.evaluate_graph_lamas(self, directory, data, initial=False, params=params)

        return avg_score

    async def test(self):
        data = []
        graph_path = f"{self.root_path}/test"
        
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_lamas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)

        self.graph = self.graph_utils.load_graph_lamas(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        pth_path = f"{self.root_path}/train"
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)
        
        # Try to find checkpoint with penalty, parallel execution, critical path, and normalization suffixes
        controller_path = None
        base_name = f"{self.dataset}_controller_sample{self.sample}"

        # Build penalty suffix based on token settings
        penalty_suffix = ""
        if hasattr(self, 'use_tokens') and self.use_tokens:
            penalty_suffix = f"_tok{self.token_weight:.6f}".replace(".", "_")

        parallel_suffix = ""
        if hasattr(self, 'parallel_execution'):
            parallel_suffix = "_parallel" if self.parallel_execution else "_sequential"

        critical_path_suffix = ""
        if hasattr(self, 'use_critical_path') and self.use_critical_path:
            critical_path_suffix = "_cp"

        norm_suffix = ""
        if hasattr(self, 'normalize_rewards') and self.normalize_rewards:
            norm_suffix = "_norm"

        # Check for checkpoint with all suffixes (penalty, parallel execution, critical path, normalization)
        if penalty_suffix or parallel_suffix or critical_path_suffix or norm_suffix:
            full_suffix_path = os.path.join(pth_directory, f"{base_name}{penalty_suffix}{parallel_suffix}{critical_path_suffix}{norm_suffix}.pth")
            if os.path.exists(full_suffix_path):
                controller_path = full_suffix_path

        # Fallback to checkpoint with only penalty suffix (for backward compatibility)
        if controller_path is None and penalty_suffix:
            penalty_only_path = os.path.join(pth_directory, f"{base_name}{penalty_suffix}.pth")
            if os.path.exists(penalty_only_path):
                controller_path = penalty_only_path

        # Fallback to default checkpoint name
        if controller_path is None:
            default_path = os.path.join(pth_directory, f"{base_name}.pth")
            if os.path.exists(default_path):
                controller_path = default_path

        # If still not found, try to find any matching checkpoint file
        if controller_path is None:
            pattern = os.path.join(pth_directory, f"{base_name}*.pth")
            matching_files = glob.glob(pattern)
            if matching_files:
                controller_path = matching_files[0]  # Use the first match
                logger.info(f"Found checkpoint with pattern matching: {controller_path}")

        if controller_path and os.path.exists(controller_path):
            logger.info(f"Loading checkpoint from: {controller_path}")
            checkpoint = torch.load(controller_path, map_location=self.device)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller model file not found. Searched in: {pth_directory}")         

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": False,
            "token_weight": self.token_weight,
            "use_tokens": self.use_tokens,
            "virtual_token_rate": self.virtual_token_rate,
            "use_critical_path": self.use_critical_path,
            "parallel_execution": self.parallel_execution,
            "normalize_rewards": self.normalize_rewards,
        }

        score = await self.evaluation_utils.evaluate_graph_test_lamas(self, directory, is_test=True, params=params)

        new_data = self.data_utils.create_result_data(self.round, score)
        data.append(new_data)

        self.data_utils.save_results(json_file_path, data)

        return score
