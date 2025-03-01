from dataclasses import dataclass
from typing import Annotated, AsyncIterator

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

from dev_ai.deps import AgentDeps
from dev_ai.tools import create_file, read_file, run_bash_command, search_web


class LLMResponse(BaseModel):
    """
    Structured response format for the LLM to use so it can indicate when the conversation should end
    """

    message: str
    end_conversation: Annotated[
        bool,
        "Always set to true unless you are asking a question.",
    ]


@dataclass
class AgentRunner:
    """
    Class which wraps an Agent to facilitate agent execution by:
    - Maintaining and managing message history
    - Providing dependenices
    """

    agent: Agent[AgentDeps, LLMResponse]
    deps: AgentDeps
    message_history: list[ModelMessage] | None = None

    def clear_message_history(self) -> None:
        """Clear the message history."""
        self.message_history = None

    def run_sync(self, query: str) -> LLMResponse:
        """Run a query and automatically provide dependencies and message history."""
        response = self.agent.run_sync(query, deps=self.deps, message_history=self.message_history)
        self.message_history = response.all_messages()
        return response.data

    async def run(self, query: str) -> LLMResponse:
        """Run a query and automatically provide dependencies and message history."""
        response = await self.agent.run(query, deps=self.deps, message_history=self.message_history)
        self.message_history = response.all_messages()
        return response.data

    async def run_stream(self, query: str) -> AsyncIterator[str]:
        """Run a query and automatically provide dependencies and message history."""
        async with self.agent.run_stream(query, deps=self.deps, message_history=self.message_history) as result:
            async for message in result.stream_text():
                yield message

            self.message_history = result.all_messages()


def get_agent_runner(model: Model, deps: AgentDeps) -> AgentRunner:
    agent = Agent(
        model=model,
        deps_type=type(deps),
        system_prompt=get_system_prompt(deps.current_working_directory),
        result_type=LLMResponse,
        tools=[search_web, run_bash_command, create_file, read_file],
    )
    return AgentRunner(agent, deps=deps)


PROMPT_TEMPLATE = """
# IDENTITY AND PURPOSE

You are a helpful assistant with strong software development and engineering skills,
whos purpose is to help the user with their software development or general computer use needs.


# IMPORTANT RULES AND EXPECTED BEHAVIOUR

* If the user request is unclear, ambigious or invalid, ask clarifying questions.
* Use the tools provided to obtain any information or perform any actions necessary to complete the user's request.
* If you have completed the users request and have no further questions to ask, set the `end_conversation` field to `True`.
* Don't assume what type of project the user is working on if it is not evident from the request. Use the available tools or ask to find out if required.
* When using the `run_bash_command` tool, you do not need to provide the output back to the user because it will be displayed to them already. 


# EXAMPLES

User: "list full paths of all python files"
Assistant: run_bash_command(command="find . -name '*.py' -type f -print", destructive=False)
User: <result from run_bash_command>

GOOD RESPONSE: I've run a command to list all python files in the current directory and its subdirectories.
BAD RESPONSE: <repeats the output back to the user>


# CONTEXTUAL INFORMATION

Current working directory: {current_working_directory}

"""


def get_system_prompt(current_working_directory: str) -> str:
    return PROMPT_TEMPLATE.format(current_working_directory=current_working_directory)
