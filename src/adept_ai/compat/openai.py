import json
from typing import Any, Iterable

from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput

from adept_ai.tool import Tool


class OpenAITools:
    """
    Helper class to convert tools to OpenAI tool format, and handle tool calling
    """

    _tools: dict[str, Tool] = {}

    def __init__(self, tools: Iterable[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def get_tool_params(self) -> list[FunctionToolParam]:
        return [self._tool_to_function_tool_param(tool) for tool in self._tools.values()]

    @staticmethod
    def _tool_to_function_tool_param(tool: Tool) -> FunctionToolParam:
        params: dict[str, Any] = tool.input_schema.copy()
        params["additionalProperties"] = False
        params["required"] = list(params["properties"].keys())  # OpenAI needs all properties to be required
        return FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=params,
        )

    async def call_tool(self, tool_name: str, **args) -> str:
        return await self._tools[tool_name].call(**args)

    async def handle_function_call_output(self, function_call_output: ResponseFunctionToolCall) -> FunctionCallOutput:
        """
        Parse a ResponseFunctionToolCall, call the tool, and return a FunctionCallOutput
        :param function_call_output:
        :return:
        """
        args = json.loads(function_call_output.arguments)
        result = await self.call_tool(function_call_output.name, **args)
        return FunctionCallOutput(call_id=function_call_output.call_id, output=result, type="function_call_output")
