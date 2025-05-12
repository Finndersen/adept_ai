import os

from dev_ai.agent_builder import AgentBuilder
from dev_ai.capabilities import FileSystemCapability, StdioMCPCapability

ROLE = """You are a helpful assistant with strong software development and engineering skills,
whose purpose is to help the user with their software development or general computer use needs."""


def get_agent_builder() -> AgentBuilder:
    return AgentBuilder(
        role=ROLE,
        capabilities=[
            FileSystemCapability(),
            StdioMCPCapability(
                "github_integration",
                "Manage GitHub repositories, enabling file operations, search functionality, and integration with the GitHub API for seamless collaborative software development.",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN", "")},
                tools=["search_repositories", "read_file", "search_code"],
            ),
        ],
    )
