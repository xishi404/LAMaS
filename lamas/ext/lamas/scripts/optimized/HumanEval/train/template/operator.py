import asyncio
import ast
import random
import sys
import threading
import traceback
from collections import Counter
from typing import Dict, List, Tuple

from lamas.ext.lamas.scripts.optimized.HumanEval.train.template.operator_an import *
from lamas.ext.lamas.scripts.optimized.HumanEval.train.template.op_prompt import *
from lamas.ext.lamas.scripts.utils import extract_test_cases_from_jsonl, test_case_2_test_function
from lamas.actions.action_node import ActionNode
from lamas.llm import LLM
from lamas.logs import logger
import re


class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, return_usage=False, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm, "return_usage": return_usage}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        result = node.instruct_content.model_dump()
        # Get usage tokens from node attribute (stored separately from Pydantic model)
        usage_tokens = getattr(node, '_usage_tokens', 0) if return_usage else 0
        return (result, usage_tokens) if return_usage else result

class CustomCodeGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction, return_usage=False):
        prompt = instruction + problem
        response, tokens = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True)
        response["_usage_tokens"] = tokens
        return response

class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction, return_usage=False):
        prompt = instruction + problem
        response, tokens = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True)
        response["_usage_tokens"] = tokens
        return response
    
class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction, return_usage=False):
        # Use Chain-of-Thought prompt to encourage step-by-step reasoning
        prompt = GENERATE_COT_PROMPT.format(instruction=instruction, problem=problem)
        response, tokens = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True)
        response["_usage_tokens"] = tokens
        return response

class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction, return_usage=False):
        # Use Chain-of-Thought prompt to encourage step-by-step reasoning
        prompt = GENERATE_COT_PROMPT.format(instruction=instruction, problem=problem)

        # Run all three generations in parallel
        results = await asyncio.gather(
            self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True),
            self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True),
            self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point, return_usage=True)
        )

        response1, tokens1 = results[0]
        response2, tokens2 = results[1]
        response3, tokens3 = results[2]
        total_tokens = tokens1 + tokens2 + tokens3

        return {"response": [response1, response2, response3], "_usage_tokens": total_tokens}
    
class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str, return_usage=False):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response, tokens = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill", return_usage=True)

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]], "_usage_tokens": tokens}

class Test(Operator):
    def __init__(self, llm: LLM, name: str = "Test"):
        super().__init__(llm, name)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout=15):
        """Execute a function with timeout protection"""
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def exec_code(self, solution, entry_point, timeout=15):
        """Execute test code with timeout protection and track execution time"""
        import time
        test_cases = extract_test_cases_from_jsonl(entry_point, dataset="HumanEval")

        fail_cases = []
        total_exec_time = 0.0

        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                # Execute test code with timeout protection and timing
                def execute_test():
                    exec(test_code, globals())

                exec_start = time.time()
                self.run_with_timeout(execute_test, (), timeout=timeout)
                total_exec_time += time.time() - exec_start
            except self.TimeoutError:
                with open("tester.txt", "a") as f:
                    f.write(f"Timeout error for {entry_point}\n")
                return {"exec_fail_case": f"Execution timed out after {timeout} seconds", "exec_time": total_exec_time}
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
                total_exec_time += time.time() - exec_start
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e), "exec_time": total_exec_time}
        if fail_cases != []:
            return {"fail_cases": fail_cases, "exec_time": total_exec_time}
        else:
            return {"result": "no error", "exec_time": total_exec_time}

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3, return_usage=False
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        total_tool_exec_time = 0.0
        total_usage_tokens = 0

        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            total_tool_exec_time += result.get("exec_time", 0.0)

            if result.get("result") == "no error":
                return {"result": True, "solution": solution, "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_usage_tokens}
            elif "exec_fail_case" in result:
                error_msg = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {error_msg}",
                    test_fail="executed unsucessfully",
                )
                response, tokens = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill", return_usage=True)
                total_usage_tokens += tokens
                solution = response["reflection_and_solution"]
            else:
                fail_cases = result.get("fail_cases", result)
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=fail_cases,
                )
                response, tokens = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill", return_usage=True)
                total_usage_tokens += tokens
                solution = response["reflection_and_solution"]

        result = self.exec_code(solution, entry_point)
        total_tool_exec_time += result.get("exec_time", 0.0)

        if result.get("result") == "no error":
            return {"result": True, "solution": solution, "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_usage_tokens}
        else:
            return {"result": False, "solution": solution, "tool_exec_time": total_tool_exec_time, "_usage_tokens": total_usage_tokens}
        
class SelfRefine(Operator):
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, return_usage=False):
        prompt = SELFREFINE_PROMPT.format(problem=problem, solution=solution)
        response, tokens = await self._fill_node(SelfRefineOp, prompt, mode="code_fill", return_usage=True)
        response["_usage_tokens"] = tokens
        return response
    
class EarlyStop(Operator):
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        return NotImplementedError
