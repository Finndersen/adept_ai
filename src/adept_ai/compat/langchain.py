from typing import Any, cast

from langchain_core.tools.structured import StructuredTool as LangChainTool

from adept_ai.tool import Tool


def tool_to_langchain_tool(tool: Tool) -> LangChainTool:
    return LangChainTool(
        name=tool.name,
        func=None,
        description=tool.description,
        coroutine=tool.call,
        args_schema=cast(dict[str, Any], tool.input_schema),
    )
