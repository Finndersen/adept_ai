import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union, cast

from pydantic_ai import RunContext
from pydantic_ai import Tool as PydanticTool
from pydantic_ai._pydantic import _is_call_ctx
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from dev_ai.framework.tool import ParameterSpec, Tool, ToolError, ToolInputSchema


def wrap_tool_func_for_pydantic(
    tool: Tool, system_prompt_builder: Callable[[], Awaitable[str]]
) -> Callable[..., Awaitable[str]]:
    """
    Decorate the tool function with a wrapper that:
    - Has a signature defined by the tool input schema
    - Forces a rebuild of the system prompt if updates_system_prompt is True
    This function can then be provided to PydanticAI as a tool function
    """

    # Check if the tool function has a `ctx: RunContext` arg
    has_ctx_param = tool_func_has_ctx_arg(tool.function)

    async def _wrapper(ctx: RunContext, *args, **kwargs) -> str:
        if has_ctx_param:
            result = await tool.call(ctx, *args, **kwargs)
        else:
            result = await tool.call(*args, **kwargs)

        if tool.updates_system_prompt:
            system_prompt_part = ctx.messages[0].parts[0]
            assert isinstance(system_prompt_part, SystemPromptPart)
            system_prompt_part.content = await system_prompt_builder()

        return result

    # Set the signature of the wrapper to match the tool function, but with the RunContext first argument
    tool_func_sig = json_schema_to_signature(tool.input_schema)

    if has_ctx_param:
        wrapper_sig = tool_func_sig
    else:
        wrapper_sig = tool_func_sig.replace(
            parameters=[
                inspect.Parameter(name="ctx", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext),
                *tool_func_sig.parameters.values(),
            ]
        )
    _wrapper.__signature__ = wrapper_sig
    # Set the wrapper annotations to match the tool function, but with the RunContext first argument
    tool_func_annotations = {name: param.annotation for name, param in wrapper_sig.parameters.items()}
    _wrapper.__annotations__ = tool_func_annotations

    _wrapper.__name__ = tool.name
    _wrapper.__doc__ = tool.description
    _wrapper.__wrapped__ = tool.function
    _wrapper.__qualname__ = tool.function.__qualname__

    return _wrapper


def to_pydanticai_tool(
    tool: Tool, system_prompt_builder: Callable[[], Awaitable[str]], enabled: bool | Callable[[], bool] = True
) -> PydanticTool:
    # Tool preparation function to only enable the tool when its capability is enabled
    async def enable_tool(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
        # Need to dynamically evaluate the enabler function each time
        if callable(enabled):
            _enabled = enabled()
        else:
            _enabled = enabled

        if _enabled:
            return tool_def
        else:
            return None

    return PydanticTool(
        function=wrap_tool_func_for_pydantic(tool, system_prompt_builder),
        name=tool.name,
        description=tool.description,
        takes_ctx=True,
        prepare=enable_tool,
    )


def tool_func_has_ctx_arg(function: Callable) -> bool:
    """
    Check whether the tool function has a `ctx: RunContext` argument.
    If it has a `ctx` argument, it must be of type `RunContext`, otherwise error will be raised.

    Args:
        function: The tool function to validate.

    Returns:
        bool: True if the function has a first parameter named 'ctx' with RunContext type, False otherwise.

    Raises:
        ToolError: If the function has a first parameter named 'ctx' that doesn't have the RunContext type.
    """
    # Get the function signature
    sig = inspect.signature(function)
    parameters = list(sig.parameters.values())

    # Check if there are any parameters
    if not parameters:
        return False

    # Get the first parameter
    first_param = parameters[0]

    # Check if the first parameter is named 'ctx'
    if first_param.name == "ctx":
        # Check if the parameter has a type annotation
        if first_param.annotation == inspect.Parameter.empty:
            raise ToolError(
                "Tool function has a 'ctx' parameter without type annotation. "
                "If you include a 'ctx' parameter, it must be of type RunContext."
            )

        # Check if the type is RunContext or a generic version of RunContext
        param_type = first_param.annotation
        if not _is_call_ctx(param_type):
            raise ToolError(
                f"Tool function has a 'ctx' parameter with type {param_type}, not RunContext. "
                f"If you include a 'ctx' parameter, it must be of type RunContext."
            )

        return True

    return False

def map_json_type_to_python(prop_schema: ParameterSpec) -> Type[Any] | Any:
    """Maps a JSON Schema property definition to a Python type hint."""
    json_type = prop_schema.get("type")

    if isinstance(json_type, list):
        # Handle union types (like ["string", "null"])
        py_types = []
        has_null = False
        for t in json_type:
            if t == "null":
                has_null = True
            else:
                # Create a temporary schema for the single type
                single_type_schema = prop_schema.copy()
                single_type_schema["type"] = t
                py_types.append(map_json_type_to_python(single_type_schema))

        # Filter out NoneType if it resulted from 'null'
        py_types = [t for t in py_types if t is not type(None)]

        # Create Union if multiple non-null types
        union_type = Union[tuple(py_types)] if len(py_types) > 1 else py_types[0]

        if has_null:
            return Optional[union_type]
        else:
            return union_type

    elif json_type == "string":
        # Could potentially check 'format' here (e.g., 'date-time' -> datetime.datetime)
        return str
    elif json_type == "integer":
        return int
    elif json_type == "number":
        return float  # Or potentially Decimal, depending on needs
    elif json_type == "boolean":
        return bool
    elif json_type == "array":
        items_schema = prop_schema.get("items")
        if isinstance(items_schema, dict):
            item_type = map_json_type_to_python(cast(ParameterSpec, items_schema))
            return List[item_type]
        else:
            # Array with no specific item type or mixed types
            return List[Any]
    elif json_type == "object":
        # Could potentially generate a TypedDict if 'properties' is detailed,
        # but Dict[str, Any] is a safe default.
        return Dict[str, Any]
    elif json_type == "null":
        return type(None)  # NoneType
    else:
        # Unknown or missing type
        return Any


def json_schema_to_signature(schema: ToolInputSchema) -> inspect.Signature:
    """
    Construct a function signature from a JSON Schema.

    Args:
        schema: The JSON Schema dictionary (expecting 'properties' and optionally 'required').

    Returns:
        The modified function.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    required_parameters = []
    optional_parameters = []
    for name, prop_schema in properties.items():
        # 1. Determine Annotation
        annotation = map_json_type_to_python(prop_schema)

        # 2. Determine Default value
        if name in required:
            default = inspect.Parameter.empty  # No default = required
            add_to = required_parameters
        else:
            # Optional parameter - provide a default. None is common.
            default = None
            add_to = optional_parameters

        # 3. Create Parameter
        param = inspect.Parameter(
            name=name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,  # Most common kind
            default=default,
            annotation=annotation,
        )
        add_to.append(param)

    # 4. Create new Signature
    return inspect.Signature(required_parameters + optional_parameters)
