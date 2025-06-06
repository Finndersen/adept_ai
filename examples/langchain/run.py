from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from adept_ai.compat.langchain import tool_to_langchain_tool
from examples.agent_builder import get_agent_builder
from examples.langchain.models import get_model_from_name_and_api_key


async def run_langchain(prompt: str, model_name: str | None, api_key: str | None = None) -> str:
    model = get_model_from_name_and_api_key(model_name=model_name, api_key=api_key)

    async with get_agent_builder() as builder:
        messages = [HumanMessage(content=prompt)]

        while True:
            agent = create_react_agent(
                model,
                tools=[tool_to_langchain_tool(tool) for tool in await builder.get_tools()],
                prompt=await builder.get_system_prompt(),
                # Interrupt after tool calls to dynamically rebuild agent with new tools and system prompt
                interrupt_after=["tools"],
            )
            response = await agent.ainvoke({"messages": messages})

            if isinstance(response["messages"][-1], ToolMessage):
                messages = response["messages"]
                # Continue agent tool calling loop with refreshed system prompt and tools
                continue
            else:
                message = response["messages"][-1].content
                return message
