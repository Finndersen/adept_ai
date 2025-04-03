from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, SystemPromptPart
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
        role=ROLE, capabilities=[FileSystemCapability(root_directory=current_working_directory, enabled=True)]
    )

    agent = Agent(
        model=model,
        tools=[tool.to_pydantic_ai_tool() for tool in builder.get_tools()],
    )

    @agent.system_prompt(dynamic=True)
    async def system_prompt() -> str:
        return await builder.get_system_prompt()

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

        # with agent.iter(user_prompt=prompt, message_history=message_history) as agent_run:
        #     run_message_history = agent_run.ctx.state.message_history
        #     async for node in agent_run:
        #         if agent.is_model_request_node(node) and run_message_history:
        #             print("UPDATING SYSTEM PROMPT")
        #             system_prompt_part = run_message_history[0].parts[0]
        #             assert isinstance(system_prompt_part, SystemPromptPart)
        #             system_prompt_part.content = await builder.get_system_prompt()
        # response = agent_run.result
        response = await agent.run(prompt,  message_history=message_history)

        assert response is not None
        console.print(f"Dev AI: {response.data}")
        message_history = response.all_messages()

        prompt = Prompt.ask("You").strip()
