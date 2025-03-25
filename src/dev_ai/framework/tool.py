import inspect
from typing import Awaitable, Callable, Literal, cast

from decorator import decorate
from pydantic import BaseModel
from pydantic_ai._pydantic import function_schema
from pydantic_ai._utils import run_in_executor
from pydantic_ai.tools import Tool as PydanticTool

from dev_ai.console import console


ToolFunction = Callable[..., Awaitable[str]] | Callable[..., str]

class ParameterSpec(BaseModel):
    type: Literal["string", "integer", "boolean", "array", "object"]
    description: str | None = None


class ToolCallError(Exception):
    """
    Exception for when a tool call fails, and information should be returned to the LLM so it can retry.
    """

    pass


class Tool(BaseModel):
    """
    Data structure to represent a tool that can be used by an agent.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    function: Callable[..., Awaitable[str]]

    @classmethod
    def from_function(
        cls, function: ToolFunction, name: str | None = None, description: str | None = None
    ) -> "Tool":
        """
        Creates a Tool instance from a function.
        """

        schema = function_schema(
            function=function, takes_ctx=False, docstring_format="auto", require_parameter_descriptions=False
        )

        description = description or schema["description"]
        name = name or function.__name__

        return cls(name=name, description=description, parameters=schema["json_schema"], function=wrap_tool(function))
    
    def to_pydantic_ai_tool(self) -> PydanticTool:
        # This will re-process the function to determine the schema...
        return PydanticTool(
            function=self.function,
            name=self.name,
            description=self.description,
        )


class ToolError(Exception):
    """Exception raised when a tool function fails."""

    pass


async def _wrapper(func: ToolFunction, *args, **kwargs) -> str:
    """Wrap a tool function to handle errors and log to the console."""
    console.print(f"[bold blue]Running tool: {func.__name__} with args: {args} and kwargs: {kwargs}[/bold blue]")
    try:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await run_in_executor(cast(Callable[..., str], func), *args, **kwargs)
    except ToolError as e:
        error_msg = f"Error: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        return error_msg


def wrap_tool(tool_func: ToolFunction) -> Callable[..., Awaitable[str]]:
    """Wrap a tool function to handle errors and log to the console."""
    return cast(Callable[..., Awaitable[str]], decorate(tool_func, _wrapper))
