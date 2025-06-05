import inspect
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession
from mcp import Tool as MCPTool
from pydantic_ai import RunContext
from pydantic_ai.messages import SystemPromptPart

from adept_ai.capabilities.mcp.main import MCPCapability
from adept_ai.compat.pydantic_ai import wrap_tool_func_for_pydantic
from adept_ai.tool import Tool, ToolError

# Define a type variable for testing generic RunContext
T = TypeVar("T")


class TestWrapToolFuncForPydantic:
    @pytest.fixture
    def mock_ctx(self):
        """Create a mock RunContext with messages."""
        ctx = MagicMock(spec=RunContext)
        system_prompt_part = MagicMock(spec=SystemPromptPart)
        ctx.messages = [MagicMock()]
        ctx.messages[0].parts = [system_prompt_part]
        return ctx

    @pytest.mark.asyncio
    async def test_wrap_tool_func_without_ctx(self, mock_ctx):
        """Test wrapping a tool function without ctx parameter."""

        # Define a simple tool function without ctx
        async def sample_function(param1: str, param2: float = 0) -> str:
            return f"Result: {param1}, {param2}"

        # Create a tool from the function
        tool = Tool.from_function(sample_function, name_prefix="Test")

        # Wrap the tool function
        wrapped_func = wrap_tool_func_for_pydantic(tool)

        # Check the signature
        sig = inspect.signature(wrapped_func)
        assert list(sig.parameters.keys()) == ["ctx", "param1", "param2"]
        assert sig.parameters["ctx"].annotation == RunContext

        # Call the wrapped function
        result = await wrapped_func(mock_ctx, param1="test", param2=42)

        # Verify the original function was called with the correct arguments by checking the return value
        assert result == "Result: test, 42"

    @pytest.mark.asyncio
    async def test_wrap_tool_func_with_ctx(self, mock_ctx):
        """Test wrapping a tool function with ctx parameter."""

        # Define a tool function with ctx
        async def sample_function_with_ctx(ctx: RunContext, param1: str) -> str:
            # We can verify ctx was passed by using it in the return value
            return f"Result with ctx: {param1} (ctx: {id(ctx)})"

        # Create a tool from the function
        tool = Tool.from_function(sample_function_with_ctx, name_prefix="Test")

        # Wrap the tool function
        wrapped_func = wrap_tool_func_for_pydantic(tool)

        # Check the signature
        sig = inspect.signature(wrapped_func)
        assert list(sig.parameters.keys()) == ["ctx", "param1"]
        assert sig.parameters["ctx"].annotation == RunContext

        # Call the wrapped function
        result = await wrapped_func(mock_ctx, param1="test")

        # Verify the original function was called with the correct arguments
        # The result should contain the id of the mock_ctx object
        assert f"Result with ctx: test (ctx: {id(mock_ctx)})" == result

    async def test_wrap_tool_func_handles_errors(self, mock_ctx):
        """Test that the wrapped function properly handles errors from the tool function."""

        # Define a tool function that raises an error
        async def failing_function(param1: str) -> str:
            raise ToolError("Test error message")

        # Create a tool from the function
        tool = Tool.from_function(failing_function, name_prefix="Test")

        # Wrap the tool function
        wrapped_func = wrap_tool_func_for_pydantic(tool)

        # Call the wrapped function - it should not raise an exception
        result = await wrapped_func(mock_ctx, param1="test")

        # Verify the error was properly handled and returned as a string
        assert "Error:" in result
        assert "Test error message" in result

    @pytest.mark.asyncio
    async def test_wrap_tool_func_with_mcptool(self, mock_ctx):
        """Test wrapping a Tool instance created with mcptool_to_tool()."""

        # Create a mock MCP tool
        mcp_tool = MCPTool(
            name="test_mcp_tool",
            description="Test MCP tool description",
            inputSchema={
                "type": "object",
                "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}},
                "required": ["param1"],
            },
        )

        # Create a mock MCP session with call_tool() method
        tool_result = MagicMock(content=[MagicMock(text="MCP tool result")], isError=False)
        mock_mcp_session = MagicMock(spec=ClientSession, call_tool=AsyncMock(return_value=tool_result))

        # Create a Tool instance from the MCPTool definition
        mcp_capability = MCPCapability(name="test", description="test", mcp_client=None)
        mcp_capability._mcp_lifecycle_manager = MagicMock(mcp_session=mock_mcp_session, active=True)
        tool = mcp_capability.mcptool_to_tool(mcp_tool)

        # Wrap the tool function
        wrapped_func = wrap_tool_func_for_pydantic(tool)

        # Check the signature
        sig = inspect.signature(wrapped_func)
        assert list(sig.parameters.keys()) == ["ctx", "param1", "param2"]
        assert sig.parameters["ctx"].annotation == RunContext

        # Call the wrapped function
        result = await wrapped_func(mock_ctx, param1="test_value", param2=42)

        # Verify the mcp_session.call_tool was called with the correct arguments
        mock_mcp_session.call_tool.assert_called_once_with(
            "test_mcp_tool", arguments={"param1": "test_value", "param2": 42}
        )

        # Verify the result
        assert result == "MCP tool result"

    @pytest.mark.asyncio
    async def test_wrap_tool_func_with_mcptool_error(self, mock_ctx):
        """Test wrapping a Tool instance created with mcptool_to_tool() that returns an error."""

        # Create a mock MCP tool
        mcp_tool = MagicMock(spec=MCPTool)
        mcp_tool.name = "test_mcp_tool_error"
        mcp_tool.description = "Test MCP tool description"
        mcp_tool.inputSchema = {"type": "object", "properties": {"param1": {"type": "string"}}, "required": ["param1"]}

        # Create a mock MCP session with call_tool() method that returns error result
        tool_result = MagicMock(content=[MagicMock(text="Something went wrong")], isError=True)
        mock_mcp_session = MagicMock(spec=ClientSession, call_tool=AsyncMock(return_value=tool_result))

        # Create a Tool instance from the MCPTool definition
        mcp_capability = MCPCapability(name="test", description="test", mcp_client=None)
        mcp_capability._mcp_lifecycle_manager = MagicMock(mcp_session=mock_mcp_session, active=True)
        tool = mcp_capability.mcptool_to_tool(mcp_tool)

        # Wrap the tool function
        wrapped_func = wrap_tool_func_for_pydantic(tool)

        # Call the wrapped function
        result = await wrapped_func(mock_ctx, param1="test_value")

        # Verify the mcp_session.call_tool was called with the correct arguments
        mock_mcp_session.call_tool.assert_called_once_with("test_mcp_tool_error", arguments={"param1": "test_value"})

        # Verify the result contains the error message
        assert "Error calling tool: Something went wrong" == result
