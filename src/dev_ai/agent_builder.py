import asyncio
from pathlib import Path
from typing import Self

from jinja2 import Template
from pydantic_ai.tools import Tool as PydanticTool

from dev_ai.capabilities import Capability
from dev_ai.pydantic_ai import to_pydanticai_tool
from dev_ai.tool import ParameterSpec, Tool, ToolError

DEFAULT_PROMPT_TEMPLATE = Path(__file__).resolve().parent / "prompt_template.md"


class AgentBuilder:
    """
    Class which uses an agent's identity and capabilities to build a dynamic system prompt and list of tools
    """

    _role: str
    _capabilities: list[Capability]
    _system_prompt_template: Path

    def __init__(self, role: str, capabilities: list[Capability], system_prompt_template: Path | None = None) -> None:
        self._role = role
        self._system_prompt_template = system_prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._capabilities = capabilities

    def _get_enable_capabilities_tool(self) -> Tool:
        """
        Returns a tool which enables or disables a capability
        """
        return Tool(
            name="enable_capability",
            description="Enable a capability",
            input_schema={
                "type": "object",
                "properties": {
                    "name": ParameterSpec(
                        type="string",
                        description="The name of the capability to enable",
                        enum=[capability.name for capability in self.disabled_capabilities],
                    )
                },
                "required": ["name"],
            },
            function=self.enable_capability,
            updates_system_prompt=True,
        )

    def enable_capability(self, name: str) -> str:
        """
        Enables a capability by name
        """
        for capability in self._capabilities:
            if capability.name.lower() == name.lower():
                capability.enabled = True
                return f"Capability {name} enabled"

        raise ToolError(f"Capability {name} not found")

    async def get_system_prompt(self) -> str:
        """
        Returns the system prompt for the agent, generated based on role and capabilities
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

    async def get_tools(self) -> list[Tool]:
        """
        Returns the tools from the enabled capabilities
        """
        if self.disabled_capabilities:
            tools = [self._get_enable_capabilities_tool()]
        else:
            tools = []

        for capability_tools in await asyncio.gather(
            *(capability.get_tools() for capability in self.enabled_capabilities)
        ):
            tools.extend(capability_tools)

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

    async def get_pydantic_ai_tools(self) -> list[PydanticTool]:
        """
        Get the tools which can be used by a PydanticAI agent.
        Returns tools for all capabilities, but only enables the tool if the capability is enabled.
        This is an unfortunate limitation that means that tools must be processed for all MCP capabilities even if they are not enabled.
        """

        tools = [
            to_pydanticai_tool(
                tool=self._get_enable_capabilities_tool(), system_prompt_builder=self.get_system_prompt, enabled=True
            )
        ]
        for capability in self._capabilities:
            for tool in await capability.get_tools():
                tools.append(
                    to_pydanticai_tool(
                        tool=tool,
                        system_prompt_builder=self.get_system_prompt,
                        # Provide `capability` as a default arg so the current loop value is 'captured'
                        enabled=lambda cap=capability: cap.enabled,
                    )
                )

        return tools

    async def __aenter__(self) -> Self:
        # TODO: Could potentially setup capabilities in parallel, but this causes error:
        # RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
        for c in self._capabilities:
            await c.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for c in self._capabilities:
            await c.teardown()
