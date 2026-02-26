import inspect
import re
import asyncio
from math import isclose
from typing import Any, Callable, List, Tuple, Literal
import torch
import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from lamas.ext.lamas.benchmark.benchmark import BaseBenchmark
from lamas.logs import logger
from lamas.utils.sanitize import sanitize

class MATHBenchmark(BaseBenchmark):
    def __init__(self,
                name: str,
                file_path: str,
                log_path: str,
                batch_size: int,
                controller: torch.nn.Module,
                operator_embeddings: List[List[float]],
                optimizer: torch.optim.Optimizer,
                token_weight: float = 0.00001,
                use_tokens: bool = False,
                virtual_token_rate: float = 50.0,
                use_critical_path: bool = True,
                parallel_execution: bool = True,
                normalize_rewards: bool = False,):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer, token_weight, use_tokens, virtual_token_rate, use_critical_path, parallel_execution, normalize_rewards)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            result = await self._generate_output(graph, input_text)
            # Unpack result: (output, cost, logprob, total_virtual_tokens, layer_operator_info)
            if len(result) == 5:
                output, cost, logprob, cp_token, layer_operator_info = result
            elif len(result) == 4:
                # Backward compatibility: old graphs return 4 values
                output, cost, logprob, layer_operator_info = result
                cp_token = 0.0
            else:
                output, cost, logprob = result[0], result[1], result[2]
                cp_token = 0.0
                layer_operator_info = None

            if not output:
                raise ValueError("output is empty")

            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost, logprob, cp_token, layer_operator_info

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), 0.0, None

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob", "cp_token"]
