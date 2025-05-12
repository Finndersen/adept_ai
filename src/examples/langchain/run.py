import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from dev_ai.compat.langchain import tool_to_langchain_tool
from examples.agent_builder import get_agent_builder

AGENT_CONFIG: RunnableConfig = {"configurable": {"thread_id": "abc123"}}


async def run_langchain(prompt: str, model_name: str | None, api_key: str | None = None):
    # Only support Gemini models for this example
    if not model_name:
        model_name = "gemini-2.0-flash"

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

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
            response = await agent.ainvoke({"messages": messages}, config=AGENT_CONFIG)

            if isinstance(response["messages"][-1], ToolMessage):
                messages = response["messages"]
                # Continue agent tool calling loop
                continue
            else:
                message = response["messages"][-1].content
                print(f"AI Agent: {message}")
                break
