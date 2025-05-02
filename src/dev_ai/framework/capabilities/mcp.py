from contextlib import _AsyncGeneratorContextManager
from pathlib import Path
from typing import Container, Sequence

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import Resource as MCPResource
from mcp import Tool as MCPTool
from mcp.types import TextResourceContents
from pydantic import AnyUrl

from dev_ai.framework.capabilities import Capability
from dev_ai.framework.tool import Tool


class UninitialisedMCPSessionError(Exception):
    pass


class BaseMCPCapability(Capability):
    _mcp_client: _AsyncGeneratorContextManager | None
    _mcp_session: ClientSession | None
    _allowed_tools: Sequence[str] | None

    def __init__(self, allowed_tools: Container[str] | None = None, enabled: bool = False):
        """

        :param allowed_tools: Collection of allowed tool names, or None to allow all available tools.
        :param enabled:
        """
        self._allowed_tools = allowed_tools
        self._mcp_client = None
        self._mcp_session = None
        super().__init__(enabled=enabled)

    def _init_mcp_client(self) -> _AsyncGeneratorContextManager:
        raise NotImplementedError()

    async def setup(self) -> None:
        # Start MCP client and session
        self._mcp_client = self._init_mcp_client()
        read, write = await self._mcp_client.__aenter__()
        self._mcp_session = ClientSession(read, write)
        await self._mcp_session.__aenter__()
        await self._mcp_session.initialize()

    async def teardown(self) -> None:
        # Clean up MCP session and client
        await self._mcp_session.__aexit__(None, None, None)
        await self._mcp_client.__aexit__(None, None, None)

    async def get_tools(self) -> list[Tool]:
        def mcptool_to_tool(mcp_tool: MCPTool) -> Tool:
            async def call_mcp_tool(**kwargs):
                tool_result = await self.mcp_session.call_tool(mcp_tool.name, arguments=kwargs)
                # Only support TextContent for now
                content = "\n".join(content.text for content in tool_result.contents)
                if tool_result.error:
                    content = "Error calling tool: " + content
                return content

            return Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                parameters=mcp_tool.inputSchema,
                function=call_mcp_tool,
                updates_system_prompt=False,  # TODO: Allow specifying which tools can update resources?
            )

        return [mcptool_to_tool(tool) for tool in await self._get_allowed_mcp_tools()]

    @property
    def mcp_session(self) -> ClientSession:
        if self._mcp_session is None:
            raise UninitialisedMCPSessionError(
                "Must initialise MCP session before retrieving tools or resources. Use the AgentBuilder as a context manager"
            )
        return self._mcp_session

    async def _get_allowed_mcp_tools(self) -> list[MCPTool]:
        tools_result = await self.mcp_session.list_tools()
        return [
            tool for tool in tools_result.tools if (self._allowed_tools is None or tool.name in self._allowed_tools)
        ]

    async def _get_resources(self) -> list[MCPResource]:
        resources_result = await self.mcp_session.list_resources()
        return resources_result.resources

    async def _read_resource(self, resource_uri: AnyUrl) -> list[str]:
        """
        Read the content of a resource. Returns a list of either text, or base64-encoded string for binary data
        :param resource_uri:
        :return:
        """
        resource_result = await self.mcp_session.read_resource(resource_uri)
        return [
            content.text if isinstance(content, TextResourceContents) else content.blob
            for content in resource_result.contents
        ]


class StdioMCPCapability(BaseMCPCapability):
    """
    Capability which provides access to an STDIO MCP server's resources and tools.
    """
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        allowed_tools: Container[str] | None = None,
        enabled: bool = False,
    ):
        """

        :param command: The executable to run to start the server.
        :param args: Command line arguments to pass to the executable.
        :param env: The environment vars to use when spawning the server process.
        :param cwd: The working directory to use when spawning the server process.
        :param allowed_tools:
        :param enabled:
        """
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=Path(cwd).resolve() if cwd else Path.cwd(),
        )
        super().__init__(allowed_tools=allowed_tools, enabled=enabled)

    def _init_mcp_client(self) -> _AsyncGeneratorContextManager:
        return stdio_client(self._server_params)
