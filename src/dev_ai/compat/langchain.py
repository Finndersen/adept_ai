from langchain_core.tools.structured import StructuredTool as LangChainTool

from dev_ai.tool import Tool


def tool_to_langchain_tool(tool: Tool) -> LangChainTool:
    return LangChainTool(
        name=tool.name,
        func=None,
        description=tool.description,
        coroutine=tool.call,
        args_schema=tool.input_schema,
    )
