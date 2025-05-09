from contextlib import _AsyncGeneratorContextManager
from pathlib import Path

from mcp import StdioServerParameters, stdio_client

from dev_ai.capabilities.mcp.base import BaseMCPCapability


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
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=Path(cwd).resolve() if cwd else Path.cwd(),
        )
        super().__init__(name=name, description=description, **kwargs)

    def _init_mcp_client(self) -> _AsyncGeneratorContextManager:
        return stdio_client(self._server_params)
