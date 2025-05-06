from contextlib import _AsyncGeneratorContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Container, Protocol, Sequence

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import Resource as MCPResourceMetadata
from mcp import Tool as MCPTool
from mcp.types import TextResourceContents
from pydantic import AnyUrl

from dev_ai.capabilities import Capability
from dev_ai.tool import Tool


class UninitialisedMCPSessionError(Exception):
    pass


@dataclass
class MCPResource:
    metadata: MCPResourceMetadata
    content: list[str]


class IncludeResource(Protocol):
    # Signature for function that determines whether a resource should be included in context
    def __call__(self, resource_uri: AnyUrl) -> bool: ...


class BaseMCPCapability(Capability):
    _mcp_client: _AsyncGeneratorContextManager | None
    _mcp_session: ClientSession | None
    _allowed_tools: Sequence[str] | None

    def __init__(
        self,
        name: str,
        description: str,
        tools: Container[str] | None = None,
        resources: IncludeResource | bool = False,
        enabled: bool = False,
    ):
        """

        :param name: Name of MCP capability.
        :param description: Description of MCP capability.
        :param tools: Collection of allowed tool names, or None to allow all available tools.
        :param resources: Whether to include resources in the initial context.
        Either a global boolean for all resources, or a callable that returns whether the resource URI should be included.
        :param enabled:
        """
        self.name = name
        self.description = description
        self._allowed_tools = tools
        self._include_resources = resources
        self._mcp_client = None
        self._mcp_session = None
        super().__init__(enabled=enabled)

    def _init_mcp_client(self) -> _AsyncGeneratorContextManager:
        raise NotImplementedError()

    async def setup(self) -> None:
        # Start MCP client and session
        print(f"Starting MCP server: {self.name}")
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
        return [self.mcptool_to_tool(tool) for tool in await self._get_allowed_mcp_tools()]

    def mcptool_to_tool(self, mcp_tool: MCPTool) -> Tool:
        async def call_mcp_tool(**kwargs: Any):
            tool_result = await self.mcp_session.call_tool(mcp_tool.name, arguments=kwargs)
            # Only support TextContent for now
            content = "\n".join(content.text for content in tool_result.content)
            if tool_result.isError:
                content = "Error calling tool: " + content
            return content

        return Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            input_schema=mcp_tool.inputSchema,
            function=call_mcp_tool,
            updates_system_prompt=False,  # TODO: Allow specifying which tools can update resources?
        )

    async def get_context_data(self) -> str:
        resource_data = [
            f"URI: {res.metadata.uri}\nName: {res.metadata.name}Content: {'\n'.join(res.content)}\n"
            for res in await self._get_included_resources()
        ]
        if resource_data:
            return f"Resources:\n{'---\n'.join(resource_data)}\n\n"
        else:
            return ""

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

    async def _get_included_resources(self) -> list[MCPResource]:
        # TODO: Cache result so its not regenerated every time system prompt is
        if self._include_resources is False:
            return []

        resources = []
        # Get data for all included resources (could parallelize this)
        for res_meta in await self._list_all_resources():
            if self._should_include_resource(res_meta.uri):
                resources.append(MCPResource(metadata=res_meta, content=await self._read_resource(res_meta.uri)))

        return resources

    def _should_include_resource(self, resource_uri: AnyUrl) -> bool:
        return self._include_resources(resource_uri) if callable(self._include_resources) else self._include_resources

    async def _list_all_resources(self) -> list[MCPResourceMetadata]:
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
        name: str,
        description: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        tools: Container[str] | None = None,
        enabled: bool = False,
    ):
        """

        :param command: The executable to run to start the server.
        :param args: Command line arguments to pass to the executable.
        :param env: The environment vars to use when spawning the server process.
        :param cwd: The working directory to use when spawning the server process.
        :param tools:
        :param enabled:
        """
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=Path(cwd).resolve() if cwd else Path.cwd(),
        )
        super().__init__(name=name, description=description, tools=tools, enabled=enabled)

    def _init_mcp_client(self) -> _AsyncGeneratorContextManager:
        return stdio_client(self._server_params)



