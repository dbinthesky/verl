"""
Agentic Task Synthesis - Parser Node

Parses raw LLM responses into structured AgenticTaskSample data.
"""

from __future__ import annotations

from typing import Any, Dict

from ...core import NodeConfig
from ..base import MapNode
from .data import AgenticTaskSample


__all__ = ['AgenticTaskParserNode']


class AgenticTaskParserNode(MapNode[AgenticTaskSample]):
    """Parser node: parses raw LLM response into AgenticTaskSample.

    Processing:
        1. Remove EOS markers
        2. Extract JSON from <think> and ```json``` blocks
        3. Validate schema
        4. Populate AgenticTaskSample fields
    """

    # 显式声明处理的数据类型
    data_type = AgenticTaskSample

    def _postprocess_solution(self, solution_str: str) -> str:
        """Remove common LLM end-of-sequence markers."""
        markers = [
            "<|im_end|>",
            "<｜end▁of▁sentence｜>",
            "<|endoftext|>"
        ]

        for marker in markers:
            if marker in solution_str:
                return solution_str[:solution_str.index(marker)].strip()

        return solution_str

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response.

        Steps:
            1. Remove EOS markers
            2. Cut from </think> onwards
            3. Extract between ```json and ```

        Returns:
            JSON string

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import re

        # Step 1: Remove EOS markers
        cleaned = self._postprocess_solution(response)

        # Step 2: Cut from </think> onwards
        if "</think>" in cleaned:
            cleaned = cleaned[cleaned.index("</think>") + len("</think>"):].strip()

        # Step 3: Extract from ```json ... ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, cleaned, re.DOTALL)

        if not match:
            raise ValueError("No JSON block found (expected ```json...```)")

        return match.group(1).strip()

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate parsed JSON schema.

        Expected schema:
        {
            "task_description": str,
            "verify_rubrics": {
                "category": [
                    {
                        "rubric_name": str,
                        "binary_statement": str,
                        "justification": List[str],
                        "traceability": str
                    }
                ]
            }
        }
        """
        if not isinstance(data, dict):
            return False

        if "task_description" not in data or "verify_rubrics" not in data:
            return False

        if not isinstance(data["task_description"], str):
            return False

        if not isinstance(data["verify_rubrics"], dict):
            return False

        # Validate rubrics structure
        for category, rubrics in data["verify_rubrics"].items():
            if not isinstance(rubrics, list):
                return False

            for rubric in rubrics:
                if not isinstance(rubric, dict):
                    return False

                required_fields = ["rubric_name", "binary_statement", "justification", "traceability"]
                if not all(field in rubric for field in required_fields):
                    return False

                if not isinstance(rubric["rubric_name"], str):
                    return False
                if not isinstance(rubric["binary_statement"], str):
                    return False
                if not isinstance(rubric["justification"], list):
                    return False
                if not isinstance(rubric["traceability"], str):
                    return False

        return True

    async def map_one(self, data: AgenticTaskSample, context: Dict[str, Any]) -> None:
        """Parse single raw response (in-place modification).

        Modifies data.task_description and data.parsed_json.
        Marks as skipped if parsing fails.
        """
        try:
            # Extract JSON
            json_str = self._extract_json_from_response(data.raw_response)

            # Parse JSON
            import json
            parsed = json.loads(json_str)

            # Validate schema
            if not self._validate_schema(parsed):
                raise ValueError("Schema validation failed")

            # Populate fields (in-place)
            data.task_description = parsed["task_description"]
            data.parsed_json = parsed

        except Exception as e:
            # Mark as skipped on parse failure
            data.mark_skipped(f"parse_error: {e}", self.name)
