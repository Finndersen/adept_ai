from typing import Any, Sequence

from composio import ActionType, AppType
from composio.client.collections import ActionModel
from composio_openai import ComposioToolSet

from adept_ai import ProvidedConfigCapability
from adept_ai.tool import Tool, ToolInputSchema


class UninitialisedToolsetError(Exception):
    pass


class ComposioActionsCapability(ProvidedConfigCapability):
    """
    Capability which provides tools from Composio actions for integrating with external services.
    Specify either a list of Composio apps or actions to integrate with.
    Description can be generated based on apps or actions if not specified.

    Examples:
    ComposioToolsCapability(
        name="WebSearch",
        description="Search the web for information about a topic.",
        actions=["COMPOSIO_SEARCH_SEARCH"],
    )

    ComposioToolsCapability(name="GoogleSheets", apps=["GOOGLESHEETS"]),
    """

    _toolset: ComposioToolSet | None

    def __init__(
        self,
        name: str,
        apps: Sequence[AppType] | None = None,
        actions: Sequence[ActionType] | None = None,
        description: str | None = None,
        instructions: list[str] = None,
        usage_examples: list[str] = None,
        enabled: bool = False,
        toolset_kwargs: dict | None = None,
    ):
        """
        Args:
            name: Name of the capability
            apps: List of Composio apps to integrate with
            actions: List of Composio actions to integrate with.
            description: Description of the capability. If not provided, will be generated based on apps or actions.
            instructions: List of instructions for using the capability
            usage_examples: List of usage examples for the capability
            enabled: Whether the capability is initially enabled
            toolset_kwargs: Optional kwargs to initialize the ComposioToolSet object.
        """
        if not (apps or actions):
            raise ValueError("Must provide either a list of Composio apps or actions")
        if not description:
            if apps:
                description = "Integrate with the following apps: " + ", ".join(apps)
            else:
                description = "Integrate with the following actions: " + ", ".join(actions)
        super().__init__(
            name=name,
            description=description,
            instructions=instructions,
            usage_examples=usage_examples,
            enabled=enabled,
        )
        self._toolset = None
        self._toolset_kwargs = toolset_kwargs or {}
        self._apps = apps
        self._actions = actions

    @property
    def toolset(self) -> ComposioToolSet:
        if not self._toolset:
            raise UninitialisedToolsetError(
                "Must initialise MCP session before retrieving tools or resources. Use the AgentBuilder or capability as a context manager."
            )
        return self._toolset

    async def setup(self) -> None:
        if self._toolset is None:
            self._toolset = ComposioToolSet(**self._toolset_kwargs)

    async def get_tools(self) -> list[Tool]:
        composio_action_schemas = self.toolset.get_action_schemas(apps=self._apps, actions=self._actions)
        return [
            self._build_tool_for_composio_action(composio_action_schema)
            for composio_action_schema in composio_action_schemas
        ]

    def _build_tool_for_composio_action(self, composio_action_schema: ActionModel) -> Tool:
        def tool_function(**kwargs: Any) -> str:
            return self.toolset.execute_action(
                action=composio_action_schema.name,
                params=kwargs,
            )

        input_schema = ToolInputSchema(
            type="object",
            properties=composio_action_schema.parameters.properties,
            required=composio_action_schema.parameters.required,
        )

        return Tool(
            name=f"{self.name}-{composio_action_schema.name}",
            description=composio_action_schema.description,
            function=tool_function,
            input_schema=input_schema,
        )
