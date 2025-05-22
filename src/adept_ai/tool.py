import inspect
import logging
from functools import partial
from typing import Awaitable, Callable, Literal, TypedDict, cast

from anyio.to_thread import run_sync
from pydantic import BaseModel, Field
from pydantic_ai.tools import Tool as PydanticTool

ToolFunction = Callable[..., Awaitable[str]] | Callable[..., str]

JSONType = Literal["string", "number", "integer", "boolean", "array", "object"]

logger = logging.getLogger(__name__)


class RequiredParams(TypedDict):
    type: JSONType


class OptionalParams(TypedDict, total=False):
    description: str
    # Only relevant to specific types
    items: dict | list
    enum: list[str] | list[int]


class ParameterSpec(RequiredParams, OptionalParams):
    pass


class ToolInputSchema(TypedDict):
    type: Literal["object"]
    properties: dict[str, ParameterSpec]
    required: list[str]


class ToolCallError(Exception):
    """
    Exception for when a tool call fails, and information should be returned to the LLM so it can retry.
    """

    pass


class Tool(BaseModel):
    """
    Data structure to represent a tool that can be used by an agent.
    """

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str
    input_schema: ToolInputSchema
    function: ToolFunction
    updates_context_data: bool = False

    @classmethod
    def from_function(
        cls,
        function: ToolFunction,
        name_prefix: str,
        name: str | None = None,
        description: str | None = None,
        updates_context_data: bool = False,
    ) -> "Tool":
        """
        Creates a Tool instance from a function, automatically constructing the input schema
        """
        # Use PydanticAI Tool to do the work of constructing JSON schema from function signature
        pydantic_ai_tool = PydanticTool(function=function, name=name, description=description)

        return cls(
            name=f"{name_prefix}-{pydantic_ai_tool.name}",
            description=pydantic_ai_tool.description,
            input_schema=cast(ToolInputSchema, pydantic_ai_tool._base_parameters_json_schema),
            function=function,
            updates_context_data=updates_context_data,
        )

    async def call(self, **kwargs) -> str:
        """
        Call the tool function, with logging and error handling.
        Calls function asynchronously regardless of whether it originally was or not.
        This method could be overridden for custom tool call behaviour.
        :param kwargs:
        :return:
        """
        logger.info(f"Running tool: '{self.name}' with args: {kwargs}")
        try:
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                wrapped_func = partial(cast(Callable[..., str], self.function), **kwargs)
                result = await run_sync(wrapped_func)

        except ToolError as e:
            result = f"Error: {str(e)}"

        return result


class ToolError(Exception):
    """Exception raised when a tool function fails."""

    pass
