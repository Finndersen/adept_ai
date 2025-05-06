from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from rich.prompt import Prompt

from examples.agent_builder import get_agent_builder
from examples.console import console

from .llm import build_model_from_name_and_api_key

EXIT_COMMANDS = ["/quit", "/exit", "/q"]


async def run_pydantic_ai(model_name: str | None, api_key: str | None = None):
    # Build the model from name and API key
    model = build_model_from_name_and_api_key(model_name, api_key)

    async with get_agent_builder() as builder:
        agent = Agent(model=model, tools=await builder.get_pydantic_ai_tools(), instrument=True)

        # Configure dynamic system prompt
        @agent.system_prompt(dynamic=True)
        async def system_prompt() -> str:
            return await builder.get_system_prompt()

        message_history: list[ModelMessage] = []

        while True:
            prompt = Prompt.ask("You").strip()

            if not prompt:
                continue

            if prompt.startswith("/"):
                if prompt.lower() in EXIT_COMMANDS:
                    break
                console.print(f"Unknown command: {prompt}")
                continue

            response = await agent.run(prompt, message_history=message_history)

            console.print(f"AI Agent: {response.output}")
            message_history = response.all_messages()
