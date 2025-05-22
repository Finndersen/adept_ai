from pydantic_ai import Agent

from adept_ai.compat.pydantic_ai import get_pydantic_ai_tools
from examples.agent_builder import get_agent_builder
from examples.pydantic_ai.models import build_model_from_name_and_api_key


async def run_pydantic_ai(prompt: str, model_name: str | None, api_key: str | None = None):
    # Build the model from name and API key
    model = build_model_from_name_and_api_key(model_name, api_key)

    async with get_agent_builder() as builder:
        agent = Agent(model=model, tools=await get_pydantic_ai_tools(builder), instrument=True)

        # Configure the dynamic system prompt (evaluated for each new agent.run(), but not within it)
        @agent.system_prompt(dynamic=True)
        async def system_prompt() -> str:
            return await builder.get_system_prompt()

        response = await agent.run(prompt)

        return response.output
