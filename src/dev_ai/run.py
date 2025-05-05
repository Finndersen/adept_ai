import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from rich.prompt import Prompt

from dev_ai.console import console
from dev_ai.framework.agent_builder import AgentBuilder
from dev_ai.framework.capabilities.filesystem import FileSystemCapability
from dev_ai.framework.capabilities.mcp import StdioMCPCapability

EXIT_COMMANDS = ["/quit", "/exit", "/q"]
ROLE = """You are a helpful assistant with strong software development and engineering skills,
whose purpose is to help the user with their software development or general computer use needs."""


async def run(model: Model, prompt: str):
    """Initialise services and run agent conversation loop."""
    current_working_directory = Path.cwd()

    async with AgentBuilder(
        role=ROLE,
        capabilities=[
            FileSystemCapability(root_directory=current_working_directory),
            StdioMCPCapability(
                "github_integration",
                "Manage GitHub repositories, enabling file operations, search functionality, and integration with the GitHub API for seamless collaborative software development.",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN", "")},
            ),
        ],
    ) as builder:
        agent = Agent(model=model, tools=await builder.get_pydantic_ai_tools(), instrument=True)

        # Configure dynamic system prompt
        @agent.system_prompt(dynamic=True)
        async def system_prompt() -> str:
            return await builder.get_system_prompt()

        message_history: list[ModelMessage] = []

        while True:
            if prompt.startswith("/"):
                if prompt.lower() in EXIT_COMMANDS:
                    break
                console.print(f"Unknown command: {prompt}")
                continue

            response = await agent.run(prompt, message_history=message_history)

            assert response is not None
            console.print(f"Dev AI: {response.data}")
            message_history = response.all_messages()
            prompt = None
            while not prompt:
                prompt = Prompt.ask("You").strip()
