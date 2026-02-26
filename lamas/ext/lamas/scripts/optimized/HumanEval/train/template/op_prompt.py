SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. Make iterative improvements to enhance its correctness and efficiency. Provide the refined solution below.
"""

GENERATE_COT_PROMPT = """
Code Generation with Chain-of-Thought Reasoning

{instruction}

Problem: {problem}

Demonstration Examples (HumanEval style):

1. Problem: Write a function that returns the sum of two numbers.
   Analysis: 
   Function takes two parameters and returns their sum. Direct addition handles all numeric types.
   
   def add(a, b):
       return a + b

2. Problem: Write a function to find the maximum element in a list.
   Analysis:
   Handle empty list edge case. Iterate through list once, tracking the maximum value.
   
   def find_max(lst):
       if not lst:
           raise ValueError("List is empty")
       max_val = lst[0]
       for item in lst[1:]:
           if item > max_val:
               max_val = item
       return max_val

Solution Protocol:
1. Analyze the problem requirements carefully
2. Identify edge cases and constraints
3. Think through the algorithmic approach step by step
4. Consider data structures and methods needed
5. Write clean, efficient, well-documented code

Step-by-Step Analysis:
"""
