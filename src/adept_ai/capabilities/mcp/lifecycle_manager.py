import asyncio
from contextlib import _AsyncGeneratorContextManager
from typing import Any, Awaitable, Callable

from adept_ai.capabilities.mcp.client_session import CustomClientSession


class UninitialisedMCPSessionError(Exception):
    pass


class MCPLifecycleManager:
    """
    Manages the asynchronous lifecycle of an MCP client and session.
    Uses a single lifecycle task with events to avoid the issue of the MCP client/session context being
    entered and exited by different tasks.
    This allows MCP server setup to be done within one task (e.g. the enable_capability() tool call task),
    and teardown to be done later in a different task (e.g. by the AgentBuilder when it exits context).

    Can also be used as a context manager to simplify MCP session handling:

    ```
    async with MCPLifecycleManager(stdio_client(...)) as mcp_session:
        # mcp_session is already initialized
        tools = await mcp_session.list_tools()
        ...

    ```
    """

    _mcp_session: CustomClientSession | None
    _mcp_client: _AsyncGeneratorContextManager
    _lifecycle_task: asyncio.Task | None

    def __init__(
        self,
        mcp_client: _AsyncGeneratorContextManager,
        **session_callbacks: Callable[..., Awaitable[Any]] | None,
    ):
        """
        Initializes the lifecycle manager with an async context manager instance.

        Args:
            mcp_client: The return value of either stdio_client() or streamablehttp_client()
            session_kwargs: Additional keyword arguments to be passed to the session constructor.
        """

        self._mcp_client = mcp_client
        self._mcp_session = None
        self._teardown_event = asyncio.Event()
        self._session_initialised_event = asyncio.Event()
        self._lifecycle_task = None
        self._session_kwargs = session_callbacks

    @property
    def active(self) -> bool:
        return self._mcp_session is not None

    @property
    def mcp_session(self) -> CustomClientSession:
        if not self.active:
            raise UninitialisedMCPSessionError(
                "MCP session is not active. Call setup() or enter context manager first."
            )
        assert self._mcp_session is not None
        return self._mcp_session

    async def _run_lifecycle(self):
        """
        Runs the lifecycle of the async context manager instance.
        """
        self._session_initialised_event.clear()
        read, write = await self._mcp_client.__aenter__()
        try:
            self._mcp_session = CustomClientSession(
                read,
                write,
                **self._session_kwargs,
            )
            await self._mcp_session.__aenter__()
            await self._mcp_session.initialize()
            self._session_initialised_event.set()
            # Wait for teardown event to be set before continuing and exiting context of session and client
            await self._teardown_event.wait()
        finally:
            if self._mcp_session:
                await self._mcp_session.__aexit__(None, None, None)
            await self._mcp_client.__aexit__(None, None, None)
            self._mcp_session = None

    async def setup(self) -> CustomClientSession:
        """Begins MCP client and session context"""
        if self.active:
            return self._mcp_session

        if self._lifecycle_task and not self._lifecycle_task.done():
            # Wait for the ongoing setup to complete or return current state
            await self._session_initialised_event.wait()
            if not self.active:
                # Await the task to propagate its exception
                try:
                    await self._lifecycle_task
                except Exception as e:
                    raise RuntimeError("Previous setup attempt failed.") from e
            return self._mcp_session

        self._teardown_event.clear()
        self._session_initialised_event.clear()
        self._mcp_session = None

        # Start lifecycle task and wait until session is initialised
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())
        await self._session_initialised_event.wait()

        if not self.active:
            try:
                await self._lifecycle_task
            except Exception as e:  # Catch the specific exception from the worker
                raise RuntimeError("Setup failed in worker task.") from e

        return self._mcp_session

    async def teardown(self):
        """Exit MCP client and session context and wait for it to finish."""
        if (
            not self._lifecycle_task or self._lifecycle_task.done() and not self.active
        ):  # If task is done but wasn't active, it might have failed early
            if self._lifecycle_task and self._lifecycle_task.done():  # ensure task is awaited if it failed early
                try:
                    await self._lifecycle_task
                except:
                    pass  # ignore exceptions here, we are just ensuring it's awaited
            return

        self._teardown_event.set()
        await self._lifecycle_task

    async def __aenter__(self):
        return await self.setup()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()
