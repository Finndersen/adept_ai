import os

from langchain_core.messages import HumanMessage, MessageLikeRepresentation, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.utils.runnable import RunnableCallable

from dev_ai.compat.langchain import tool_to_langchain_tool
from examples.agent_builder import get_agent_builder

AGENT_CONFIG = {"configurable": {"thread_id": "abc123"}}


async def run_langchain(prompt: str, model_name: str | None, api_key: str | None = None):
    # Only support Gemini models for this example
    if not model_name:
        model_name = "gemini-2.0-flash"

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    memory = MemorySaver()

    # The rebuilt create_react_agent() does not support dynamic tool updates, so must enable all capabilities
    async with get_agent_builder(enable_all_capabilities=True) as builder:

        async def get_prompt(state: AgentState) -> list[MessageLikeRepresentation]:
            # System prompt gets dynamically regenerated for each message sent to LLM (even within a run)
            system_prompt = await builder.get_system_prompt()
            return [SystemMessage(content=system_prompt)] + state["messages"]

        agent = create_react_agent(
            model,
            tools=[tool_to_langchain_tool(tool) for tool in await builder.get_tools()],
            prompt=RunnableCallable(func=None, afunc=get_prompt),
            checkpointer=memory,
        )

        response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=AGENT_CONFIG)
        message = response["messages"][-1].content
        print(f"AI Agent: {message}")
