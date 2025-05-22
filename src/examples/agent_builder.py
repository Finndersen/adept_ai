import os

from adept_ai.agent_builder import AgentBuilder
from adept_ai.capabilities import FileSystemCapability, StdioMCPCapability

ROLE = """You are a helpful assistant with strong software development and engineering skills,
whose purpose is to help the user with their software development or general computer use needs."""


def get_agent_builder() -> AgentBuilder:
    return AgentBuilder(
        role=ROLE,
        capabilities=[
            FileSystemCapability(),
            StdioMCPCapability(
                name="Github",
                description="Manage GitHub repositories, enabling file operations, search functionality, and integration with the GitHub API for seamless collaborative software development.",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN", "")},
                tools=["search_repositories", "read_file", "search_code"],
            ),
            # Register here to get API key: https://developer.accuweather.com/
            StdioMCPCapability(
                name="AccuWeather",
                description="Get weather data from AccuWeather, a weather forecasting and weather information service.",
                command="npx",
                args=["-y", "@timlukahorstmann/mcp-weather"],
                env={"ACCUWEATHER_API_KEY": os.getenv("ACCUWEATHER_API_KEY", "")},
                instructions=["Use a sessionId of '123' when calling tools that require one."],
            ),
            # Check here for setup instructions: https://github.com/GongRzhe/Gmail-MCP-Server
            StdioMCPCapability(
                name="Gmail",
                description="Access my Gmail account to read and send emails.",
                command="npx",
                args=["@gongrzhe/server-gmail-autoauth-mcp"],
                tools=["search_emails", "read_email", "send_email", "delete_email", "batch_delete_emails"],
                instructions=[
                    "Use `older_than:` or `newer_than:` instead of `after:` and `before:` for relative time queries"
                ],
                usage_examples=[
                    (
                        "Examples of email search syntax:  \n"
                        "* from: and to: - Find emails from or to a specific sender (e.g., from:john.doe@example.com).  \n"
                        "* subject: - Find emails by keywords or phrases in the subject line (e.g., subject:important meeting).  \n"
                        "* after: and before: - Find emails received after or before a specific date (e.g., after:2023/01/01, before:2023/12/31). Must only be used with a date. \n"
                        "* older_than: and newer_than: - Find emails older or newer than a specified time period (e.g., older_than:30d).  \n"
                    )
                ],
            ),
            # Check here for authentication & setup instructions: https://github.com/MCP-Mirror/GongRzhe_Calendar-MCP-Server
            StdioMCPCapability(
                name="GoogleCalendar",
                description="Access my Google Calendar to read and write events.",
                command="npx",
                args=["@gongrzhe/server-calendar-mcp"],
                env={
                    "GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID", ""),
                    "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                    "GOOGLE_REFRESH_TOKEN": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
                },
                instructions=[
                    "For the `list_events` tool, `timeMin` and `timeMax` must be provided in format: YYYY-MM-DDTHH:mm:ssZ e.g. 2025-05-01T00:00:00Z"
                ],
            ),
        ],
    )
