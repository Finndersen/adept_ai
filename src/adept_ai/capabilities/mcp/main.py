import logging
from contextlib import _AsyncGeneratorContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Container, cast

from mcp import ClientSession, McpError, StdioServerParameters, stdio_client
from mcp import Resource as MCPResourceMetadata
from mcp import Tool as MCPTool
from mcp.client.session import LoggingFnT, SamplingFnT
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextResourceContents
from pydantic import AnyUrl

from adept_ai.capabilities import Capability
from adept_ai.tool import Tool, ToolInputSchema
from adept_ai.utils import cached_method

from .lifecycle_manager import MCPLifecycleManager, UninitialisedMCPSessionError

logger = logging.getLogger(__name__)


@dataclass
class MCPResource:
    metadata: MCPResourceMetadata
    content: list[str]


# Signature for function that determines whether a resource should be included in context
ResourceURIChecker = Callable[[AnyUrl], bool]


class MCPCapability(Capability):
    """
    Base class for a capability which provides access to an MCP server's resources and tools.
    Features:
    - Handles MCP server lifecycle and client session
    - Specify which tools to use, and handles tool execution
    - Specify which resources to include in the initial context data
    - Provide callbacks to handle sampling and logging events from the MCP server
    - Smart tool and resource list caching to avoid unnecessary requests to the server,
        with server notification handling to reset the cache

    Can also be used standalone as a powerful MCP client:
    ```
    async with StdioMCPCapability(...) as mcp_client:
        tools = mcp_client.get_tools()

        resources = await mcp_client.list_all_resources()
        for resource in resources:
            resource_content = await mcp_client.read_resource(resource.uri)
            print(resource_content)

    ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        mcp_client: _AsyncGeneratorContextManager,
        tools: Container[str] | None = None,
        resources: ResourceURIChecker | bool = False,
        instructions: list[str] | None = None,
        usage_examples: list[str] | None = None,
        sampling_callback: SamplingFnT | None = None,
        logging_callback: LoggingFnT | None = None,
        **kwargs,
    ):
        """

        :param name: Name of MCP capability.
        :param description: Description of MCP capability.
        :param tools: Collection of allowed tool names, or None to allow all available tools.
        :param resources: Whether to include resources in the initial context.
            Either a global boolean for all resources, or a callable that returns whether the resource URI should be included.
        :param instructions: Instructions to be added to the system prompt, to guide usage of the MCP server
        :param sampling_callback: Callback function to be called to handle a sampling request from the MCP server.
        :param logging_callback: Callback function to be called to handle a logging event from the MCP server.
        :param enabled: Whether the capability is initially enabled.
        """
        self.name = name
        self.description = description
        self._allowed_tools = tools
        self._include_resources = resources
        self._instructions = instructions or []
        self._usage_examples = usage_examples or []
        self._context_active = False  # Whether parent AgentBuilder or this capability's context manager is active
        self._mcp_lifecycle_manager = MCPLifecycleManager(
            mcp_client,
            tool_list_changed_callback=self._handle_tool_list_changed,
            resource_list_changed_callback=self._handle_resource_list_changed,
            sampling_callback=sampling_callback,
            logging_callback=logging_callback,
        )
        super().__init__(**kwargs)

    @property
    def instructions(self) -> list[str] | None:
        return self._instructions

    @property
    def usage_examples(self) -> list[str] | None:
        return self._usage_examples

    ## MCP Client & Session management ##
    @property
    def mcp_session(self) -> ClientSession:
        if not self._mcp_lifecycle_manager.active:
            raise UninitialisedMCPSessionError(
                "Must initialise MCP session before retrieving tools or resources. Use the AgentBuilder as a context manager"
            )
        return self._mcp_lifecycle_manager.mcp_session

    ## Lifecycle Management ##
    async def setup(self) -> None:
        # Start MCP client and session
        if not self._mcp_lifecycle_manager.active:
            logger.debug(f"Starting MCP server: {self.name}")
            await self._mcp_lifecycle_manager.setup()

    async def teardown(self) -> None:
        await self._mcp_lifecycle_manager.teardown()

    ## TOOLS ##
    @cached_method
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
            name=f"{self.name}-{mcp_tool.name}",
            description=mcp_tool.description or "",
            # "required" param may not be provided if tool has no args/properties
            input_schema=cast(ToolInputSchema, {"required": [], **mcp_tool.inputSchema}),
            function=call_mcp_tool,
            updates_context_data=False,  # TODO: Allow specifying which tools can update resources?
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

    async def _get_allowed_mcp_tools(self) -> list[MCPTool]:
        tools_result = await self.mcp_session.list_tools()
        return [
            tool for tool in tools_result.tools if (self._allowed_tools is None or tool.name in self._allowed_tools)
        ]

    async def _handle_tool_list_changed(self) -> None:
        # Reset the tool list cache when the server notifies that it has changed
        self.get_tools.clear_cache()

    ## RESOURCES ##
    async def _get_included_resources(self) -> list[MCPResource]:
        """
        Get the metadata and content of all resources that should be included in the context.
        :return:
        """
        if self._include_resources is False:
            return []

        resources = []
        # Get data for all included resources (could parallelize this)
        for res_meta in await self.list_all_resources():
            if self._should_include_resource(res_meta.uri):
                resources.append(MCPResource(metadata=res_meta, content=await self.read_resource(res_meta.uri)))

        return resources

    def _should_include_resource(self, resource_uri: AnyUrl) -> bool:
        return self._include_resources(resource_uri) if callable(self._include_resources) else self._include_resources

    @cached_method
    async def list_all_resources(self) -> list[MCPResourceMetadata]:
        try:
            resources_result = await self.mcp_session.list_resources()
            return resources_result.resources
        except McpError as e:
            if str(e) == "Method not found":
                return []
            else:
                raise

    async def read_resource(self, resource_uri: AnyUrl) -> list[str]:
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

    async def _handle_resource_list_changed(self) -> None:
        # Reset the resource list cache when the server notifies that it has changed
        self.list_all_resources.clear_cache()


class StdioMCPCapability(MCPCapability):
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
        **kwargs,
    ):
        """

        :param command: The executable to run to start the server.
        :param args: Command line arguments to pass to the executable.
        :param env: The environment vars to use when spawning the server process.
        :param cwd: The working directory to use when spawning the server process.
        :param tools:
        :param enabled:
        """
        super().__init__(
            name=name,
            description=description,
            mcp_client=stdio_client(
                StdioServerParameters(
                    command=command,
                    args=args,
                    env=env,
                    cwd=Path(cwd).resolve() if cwd else Path.cwd(),
                )
            ),
            **kwargs,
        )


class HTTPMCPCapability(MCPCapability):
    """
    Capability which provides access to an HTTP MCP server's resources and tools.
    """

    def __init__(self, name: str, description: str, url: str, headers: dict[str, str] | None = None, **kwargs):
        super().__init__(
            name=name, description=description, mcp_client=streamablehttp_client(url, headers=headers), **kwargs
        )
