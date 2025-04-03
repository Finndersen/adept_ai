from functools import partial
from pathlib import Path

from jinja2 import Template
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition

from dev_ai.framework.capabilities import Capability
from dev_ai.framework.tool import ParameterSpec, Tool, wrap_tool_for_pydantic


class AgentBuilder:
    """
    Class which uses an agent's identity and capabilities to build a dynamic system prompt and list of tools
    """

    _role: str
    _capabilities: list[Capability]
    _system_prompt_template: Path

    def __init__(self, role: str, capabilities: list[Capability], system_prompt_template: Path | None = None) -> None:
        self._role = role
        self._system_prompt_template = system_prompt_template or Path(__file__).resolve().parent / "prompt_template.md"
        self._capabilities = capabilities

    def _get_enable_capabilities_tool(self) -> Tool:
        """
        Returns a tool which enables or disables a capability
        """
        return Tool(
            name="enable_capability",
            description="Enable a capability",
            parameters={
                "name": ParameterSpec(type="string", description="The name of the capability to enable"),
            },
            function=self.enable_capability,
        )

    def enable_capability(self, name: str) -> str:
        """
        Enables a capability by name
        """
        for capability in self._capabilities:
            if capability.name.lower() == name.lower():
                capability.enabled = True
                return f"Capability {name} enabled"

        raise ValueError(f"Capability {name} not found")

    async def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the agent, generated from the capabilities
        """
        with self._system_prompt_template.open("r") as f:
            template = f.read()

        jinja_template = Template(template, enable_async=True)

        # Render the template with the context
        prompt = await jinja_template.render_async(
            role=self._role,
            enabled_capabilities=self.enabled_capabilities,
            disabled_capabilities=self.disabled_capabilities,
        )

        return prompt

    def get_tools(self) -> list[Tool]:
        """
        Returns the tools from the enabled capabilities
        """
        if any((not c.enabled) for c in self._capabilities):
            tools = [self._get_enable_capabilities_tool()]
        else:
            tools = []

        for capability in self.enabled_capabilities:
            tools.extend(capability.get_tools())

        return tools

    @property
    def enabled_capabilities(self) -> list[Capability]:
        """
        Returns the enabled capabilities
        """
        return [c for c in self._capabilities if c.enabled]

    @property
    def disabled_capabilities(self) -> list[Capability]:
        """
        Returns the disabled capabilities
        """
        return [c for c in self._capabilities if not c.enabled]

    def get_pydantic_ai_tools(self) -> list[PydanticTool]:
        """
        Get the tools which can be used by a PydanticAI agent
        Returns tools for all capabilities, but only enables the tool if the capability is enabled
        """

        async def enable_tool(
            ctx: RunContext, tool_def: ToolDefinition, capability: Capability
        ) -> ToolDefinition | None:
            # Disable the tool if the capability is disabled
            if capability.enabled:
                return tool_def
            else:
                return None

        tools = []
        for capability in self._capabilities:
            enable_capability_tool = partial(enable_tool, capability=capability)

            for tool in capability.get_tools():
                tools.append(
                    PydanticTool(
                        function=wrap_tool_for_pydantic(
                            tool.function, self.get_system_prompt, refresh_system_prompt=tool.updates_system_prompt
                        ),
                        name=tool.name,
                        description=tool.description,
                        takes_ctx=True,
                        prepare=enable_capability_tool,
                    )
                )

        return tools
