import os
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseFunctionToolCallParam
from openai.types.responses.response_input_param import ResponseInputParam

from adept_ai.compat.openai import OpenAITools
from examples.agent_builder import get_agent_builder


async def run_openai(prompt: str, model_name: str | None, api_key: str | None = None) -> str:
    """
    Run an OpenAI model directly using the OpenAI SDK.

    Args:
        prompt: The user's prompt to send to the model
        model_name: The OpenAI model name to use (defaults to gpt-4o if not specified)
        api_key: The OpenAI API key (defaults to OPENAI_API_KEY environment variable)
    """
    client, model_name = get_openai_client_and_model(api_key, model_name)

    async with get_agent_builder() as agent_builder:
        message_history: ResponseInputParam = [EasyInputMessageParam(role="user", content=prompt)]
        # Agent tool calling loop
        while True:
            # Retrieve tools within loop so they can be dynamically updated
            tools = await agent_builder.get_tools()
            # Helper class for translating to OpenAI tool format, and handling tool calling
            openai_tools = OpenAITools(tools)
            system_prompt = await agent_builder.get_system_prompt()

            response = await client.responses.create(
                model=model_name,
                input=[EasyInputMessageParam(role="system", content=system_prompt)] + message_history,
                tools=openai_tools.get_tool_params(),
            )

            for output in response.output:
                if output.type == "function_call":
                    # Call the tool
                    result = await openai_tools.handle_function_call_output(output)
                    # Add the tool response to the message history
                    message_history.append(cast(ResponseFunctionToolCallParam, output.model_dump()))
                    message_history.append(result)
                else:
                    return response.output_text


def get_openai_client_and_model(api_key: str | None = None, model_name: str | None = None) -> tuple[AsyncOpenAI, str]:
    """
    Get an async OpenAI client using the OpenAI SDK.
    Use Gemini if GEMINI_API_KEY is set in the environment.
    """
    # Set default model if not provided

    # Use provided API key or get from environment
    api_url = None
    if not api_key:
        if os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not model_name:
                model_name = "gpt-4o"
        elif os.environ.get("GEMINI_API_KEY"):
            api_key = os.environ.get("GEMINI_API_KEY")
            api_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not model_name:
                model_name = "gemini-2.5-flash-preview-04-17"
        else:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
    elif not model_name:
        raise ValueError("Must provide model name if API key is provided.")

    return AsyncOpenAI(api_key=api_key, base_url=api_url), model_name
