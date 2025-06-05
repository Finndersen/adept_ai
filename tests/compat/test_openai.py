import json

import pytest
from openai.types.responses import ResponseFunctionToolCall

from adept_ai.compat.openai import OpenAITools
from adept_ai.tool import Tool


class TestOpenAITools:
    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""

        # Create a simple tool with a mock function
        async def add_numbers(a: int, b: int = 5) -> str:
            return str(a + b)

        add_tool = Tool.from_function(
            add_numbers, name_prefix="Test", name="add", description="Add two numbers together"
        )

        # Create another tool
        async def echo(message: str) -> str:
            return f"Echo: {message}"

        echo_tool = Tool.from_function(echo, name_prefix="Test", name="echo", description="Echo a message back")

        return [add_tool, echo_tool]

    @pytest.fixture
    def openai_tools(self, sample_tools):
        """Create an OpenAITools instance with sample tools."""
        return OpenAITools(sample_tools)

    @pytest.mark.asyncio
    async def test_get_tool_params(self, openai_tools, sample_tools):
        """Test converting tools to OpenAI format."""
        tool_params = openai_tools.get_responses_tools()

        # Check tool params match expected data structure
        expected_params = [
            {
                "name": "Test-add",
                "type": "function",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "Test-echo",
                "type": "function",
                "description": "Echo a message back",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        ]

        assert tool_params == expected_params

    @pytest.mark.asyncio
    async def test_call_tool(self, openai_tools, sample_tools):
        """Test calling a tool."""
        # Call the tool
        result = await openai_tools.call_tool("Test-add", a=5, b=3)

        # Verify the result
        assert result == "8"

    @pytest.mark.asyncio
    async def test_handle_function_call_output(self, openai_tools, sample_tools):
        """Test handling a function call output from OpenAI."""
        # Handle the function call output
        function_tool_call = ResponseFunctionToolCall(
            call_id="call_123", type="function_call", name="Test-add", arguments=json.dumps({"a": 5, "b": 3})
        )
        result = await openai_tools.handle_function_call_output(function_tool_call)

        # Verify the result is a FunctionCallOutput
        assert result == {"call_id": "call_123", "output": "8", "type": "function_call_output"}
