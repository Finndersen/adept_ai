from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from rich.prompt import Prompt

from dev_ai.console import console
from dev_ai.framework.agent_builder import AgentBuilder
from dev_ai.framework.capabilities.filesystem import FileSystemCapability

EXIT_COMMANDS = ["/quit", "/exit", "/q"]
ROLE = """
You are a helpful assistant with strong software development and engineering skills,
whose purpose is to help the user with their software development or general computer use needs."""


async def run(model: Model, prompt: str):
    """Initialise services and run agent conversation loop."""
    current_working_directory = Path.cwd()

    builder = AgentBuilder(
        role=ROLE, capabilities=[FileSystemCapability(root_directory=current_working_directory, enabled=False)]
    )

    agent = Agent(
        model=model,
        tools=await builder.get_pydantic_ai_tools(),
    )

    @agent.system_prompt(dynamic=True)
    async def system_prompt() -> str:
        sys_prompt = await builder.get_system_prompt()
        return sys_prompt

    message_history: list[ModelMessage] = []

    while True:
        # Need to dynamically re-build the agent for each run because the capabilities can change

        if not prompt:
            continue

        if prompt.startswith("/"):
            if prompt.lower() in EXIT_COMMANDS:
                break
            console.print(f"Unknown command: {prompt}")
            continue

        response = await agent.run(prompt, message_history=message_history)

        assert response is not None
        console.print(f"Dev AI: {response.data}")
        message_history = response.all_messages()

        prompt = Prompt.ask("You").strip()
