from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel
from pydantic_ai._pydantic import function_schema
from pydantic_ai.tools import GenerateToolJsonSchema

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
            function=function,
            takes_ctx=False,
            docstring_format="auto",
            require_parameter_descriptions=False,
            schema_generator=GenerateToolJsonSchema,
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


