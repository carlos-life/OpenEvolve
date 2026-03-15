"""LLM integration for generating and mutating code."""

from __future__ import annotations

import re
from typing import List

from openai import AsyncOpenAI


def _extract_code(text: str) -> str:
    """Extract Python code from an LLM response.

    Looks for ```python ... ``` blocks first, then ``` ... ``` blocks,
    and falls back to the raw text.
    """
    # Try ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class LLMClient:
    """Client for generating/mutating code via an OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_initial(
        self, problem_description: str, function_signature: str, n: int
    ) -> List[str]:
        """Generate *n* initial candidate solutions."""
        prompt = (
            "You are an expert Python programmer.\n\n"
            f"Problem: {problem_description}\n\n"
            f"Write a Python function with this exact signature:\n{function_signature}\n\n"
            "Output ONLY the function code inside a ```python``` block. "
            "No explanations, no test code."
        )
        results: list[str] = []
        for _ in range(n):
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            code = _extract_code(resp.choices[0].message.content or "")
            results.append(code)
        return results

    async def mutate(
        self,
        code: str,
        fitness: float,
        problem_description: str,
        hints: str = "",
    ) -> str:
        """Mutate an existing solution to potentially improve it."""
        prompt = (
            "You are an expert Python programmer.\n\n"
            f"Problem: {problem_description}\n\n"
            f"Here is a current solution (fitness={fitness:.3f}):\n```python\n{code}\n```\n\n"
            "Modify this solution to improve it. Try a different approach or fix bugs.\n"
        )
        if hints:
            prompt += f"Hints: {hints}\n"
        prompt += (
            "\nOutput ONLY the improved function code inside a ```python``` block. "
            "Keep the same function name and signature."
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return _extract_code(resp.choices[0].message.content or "")

    async def crossover(
        self,
        code1: str,
        fitness1: float,
        code2: str,
        fitness2: float,
        problem_description: str,
    ) -> str:
        """Combine two solutions to create a new one."""
        prompt = (
            "You are an expert Python programmer.\n\n"
            f"Problem: {problem_description}\n\n"
            f"Solution A (fitness={fitness1:.3f}):\n```python\n{code1}\n```\n\n"
            f"Solution B (fitness={fitness2:.3f}):\n```python\n{code2}\n```\n\n"
            "Combine the best ideas from both solutions into a single improved solution.\n"
            "Output ONLY the combined function code inside a ```python``` block. "
            "Keep the same function name and signature."
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return _extract_code(resp.choices[0].message.content or "")


class MockLLMClient:
    """A deterministic mock LLM client for testing without API keys."""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        self.model = model
        self._call_count = 0

    async def generate_initial(
        self, problem_description: str, function_signature: str, n: int
    ) -> List[str]:
        results: list[str] = []
        for i in range(n):
            self._call_count += 1
            # Generate simple sorting implementations with variation
            if "sort" in problem_description.lower():
                if i % 3 == 0:
                    code = "def solution(arr: list) -> list:\n    return sorted(arr)"
                elif i % 3 == 1:
                    code = (
                        "def solution(arr: list) -> list:\n"
                        "    result = list(arr)\n"
                        "    for i in range(len(result)):\n"
                        "        for j in range(i + 1, len(result)):\n"
                        "            if result[i] > result[j]:\n"
                        "                result[i], result[j] = result[j], result[i]\n"
                        "    return result"
                    )
                else:
                    code = (
                        "def solution(arr: list) -> list:\n"
                        "    if len(arr) <= 1:\n"
                        "        return list(arr)\n"
                        "    pivot = arr[0]\n"
                        "    left = [x for x in arr[1:] if x <= pivot]\n"
                        "    right = [x for x in arr[1:] if x > pivot]\n"
                        "    return solution(left) + [pivot] + solution(right)"
                    )
            else:
                # Generic: identity-like function
                code = "def solution(*args):\n    return args[0] if args else None"
            results.append(code)
        return results

    async def mutate(
        self,
        code: str,
        fitness: float,
        problem_description: str,
        hints: str = "",
    ) -> str:
        self._call_count += 1
        # If the code already works perfectly, return it unchanged
        if fitness >= 1.0:
            return code
        # Otherwise return the canonical correct solution
        if "sort" in problem_description.lower():
            return "def solution(arr: list) -> list:\n    return sorted(arr)"
        return code

    async def crossover(
        self,
        code1: str,
        fitness1: float,
        code2: str,
        fitness2: float,
        problem_description: str,
    ) -> str:
        self._call_count += 1
        # Return the fitter parent's code
        return code1 if fitness1 >= fitness2 else code2
