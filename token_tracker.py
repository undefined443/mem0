"""Token usage tracker for LLM API calls."""

from typing import Any


class TokenTracker:
    """Track token usage from NVIDIA API responses."""

    def __init__(self) -> None:
        """Initialize the token tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def callback(self, llm_instance: Any, response: Any, params: dict[str, Any]) -> None:
        """
        Callback to capture usage info from API response.

        Args:
            llm_instance: The LLM instance (unused).
            response: The API response object.
            params: The request parameters (unused).
        """
        if hasattr(response, "usage") and response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens
            self.call_count += 1

            print(
                f"  [API Call #{self.call_count}] "
                f"Input: {response.usage.prompt_tokens}, "
                f"Output: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

    def summary(self) -> None:
        """Print summary of token usage."""
        print("\n=== Token Usage Summary ===")
        print(f"Total API calls: {self.call_count}")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total tokens: {self.total_tokens}")
