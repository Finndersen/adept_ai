from typing import Awaitable, Callable

from mcp import ClientSession, types

ListChangedCallback = Callable[[], Awaitable[None]]


class CustomClientSession(ClientSession):
    """
    Custom MCP ClientSession class which allows handling of more notification types:
    - ResourceListChangedNotification

    """

    def __init__(self, *args, tool_list_changed_callback: ListChangedCallback | None = None, resource_list_changed_callback: ListChangedCallback | None = None, **kwargs):
        self._tool_list_changed_callback = tool_list_changed_callback
        self._resource_list_changed_callback = resource_list_changed_callback
        super().__init__(*args, **kwargs)

    async def _received_notification(self, notification: types.ServerNotification) -> None:
        """Handle notifications from the server."""
        # Process specific notification types
        match notification.root:
            case types.LoggingMessageNotification(params=params):
                await self._logging_callback(params)
            case types.ResourceListChangedNotification(params=params):
                await self._resource_list_changed_callback()
            case types.ToolListChangedNotification(params=params):
                await self._tool_list_changed_callback()
            case _:
                pass
