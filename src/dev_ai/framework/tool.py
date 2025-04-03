import inspect
from typing import Any, Awaitable, Callable, Literal, cast

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai._pydantic import function_schema
from pydantic_ai._utils import run_in_executor
from pydantic_ai.messages import SystemPromptPart

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
    parameters: dict[str, Any]
    function: ToolFunction
    updates_system_prompt: bool = False

    @classmethod
    def from_function(
        cls,
        function: ToolFunction,
        name: str | None = None,
        description: str | None = None,
        updates_system_prompt: bool = False,
    ) -> "Tool":
        """
        Creates a Tool instance from a function.
        """

        schema = function_schema(
            function=function, takes_ctx=False, docstring_format="auto", require_parameter_descriptions=False
        )

        description = description or schema["description"]
        name = name or function.__name__

        return cls(
            name=name,
            description=description,
            parameters=schema["json_schema"],
            function=function,
            updates_system_prompt=updates_system_prompt,
        )


class ToolError(Exception):
    """Exception raised when a tool function fails."""

    pass


def wrap_tool_for_pydantic(
    tool_func: ToolFunction, system_prompt_builder: Callable[[], Awaitable[str]], refresh_system_prompt: bool = False
) -> Callable[..., Awaitable[str]]:
    """Decorate the tool function with a wrapper that will have the same signature as the tool function."""

    async def _wrapper(ctx: RunContext, *args, **kwargs) -> str:
        """Wrap a tool function to handle errors and log to the console."""
        console.print(
            f"[bold blue]Running tool: {tool_func.__name__} with args: {args} and kwargs: {kwargs}[/bold blue]"
        )
        try:
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(*args, **kwargs)
            else:
                result = await run_in_executor(cast(Callable[..., str], tool_func), *args, **kwargs)

            if refresh_system_prompt:
                system_prompt_part = ctx.messages[0].parts[0]
                assert isinstance(system_prompt_part, SystemPromptPart)
                print("UPDATING SYSTEM PROMPT")
                system_prompt_part.content = await system_prompt_builder()

            return result
        except ToolError as e:
            error_msg = f"Error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    # Set the signature of the wrapper to match the tool function, but with the RunContext first argument
    tool_func_sig = inspect.signature(tool_func)
    wrapper_sig = tool_func_sig.replace(
        parameters=[
            inspect.Parameter(name="ctx", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *tool_func_sig.parameters.values(),
        ]
    )
    _wrapper.__signature__ = wrapper_sig
    # Set the wrapper annotations to match the tool function, but with the RunContext first argument
    tool_func_annotations = tool_func.__annotations__.copy()
    tool_func_annotations["ctx"] = RunContext
    _wrapper.__annotations__ = tool_func_annotations

    _wrapper.__name__ = tool_func.__name__
    _wrapper.__doc__ = tool_func.__doc__
    _wrapper.__wrapped__ = tool_func
    _wrapper.__qualname__ = tool_func.__qualname__

    return _wrapper
