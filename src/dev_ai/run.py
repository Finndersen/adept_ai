import os

from pydantic_ai.models import Model
from rich.console import Console
from rich.prompt import Prompt

from dev_ai.agent import get_agent_runner
from dev_ai.commands import EXIT_COMMANDS, handle_special_command
from dev_ai.deps import AgentDeps


def run(model: Model, prompt: str):
    """Initialise services and run agent conversation loop."""
    deps = AgentDeps(console=Console(), current_working_directory=os.getcwd())

    agent_runner = get_agent_runner(model=model, deps=deps)

    while True:
        if not prompt:
            continue

        if prompt.lower() in EXIT_COMMANDS:
            break

        if prompt.startswith("/"):
            handle_special_command(prompt, agent_runner)
            continue

        response = agent_runner.run_sync(prompt)
        deps.console.print(f"Dev: {response.message}")

        # Exit if LLM indicates conversation is over
        if response.end_conversation:
            break

        prompt = Prompt.ask("You").strip()
