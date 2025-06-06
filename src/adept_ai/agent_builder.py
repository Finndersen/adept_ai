import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Self

from jinja2 import Template

from adept_ai.capabilities import Capability
from adept_ai.tool import ParameterSpec, Tool, ToolError

DEFAULT_PROMPT_TEMPLATE = Path(__file__).resolve().parent / "prompt_template.md"

logger = logging.getLogger(__name__)


class InvalidCapabilityName(Exception):
    pass


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
        self._capabilities = self._validate_capabilities(capabilities)

    @staticmethod
    def _validate_capabilities(capabilities) -> list[Capability]:
        seen_names = set()
        for capability in capabilities:
            if capability.name.lower() in seen_names:
                raise ValueError(f"Duplicate capability name: {capability.name}")
            seen_names.add(capability.name.lower())
        return capabilities

    def get_enable_capabilities_tool(self) -> Tool:
        """
        Returns a tool which enables capabilities
        """
        return Tool(
            name="enable_capabilities",
            description="Enable a set of capabilities",
            input_schema={
                "type": "object",
                "properties": {
                    "capabilities": ParameterSpec(
                        type="array",
                        description="The names of the capabilities to enable.",
                        items=ParameterSpec(
                            type="string", enum=[capability.name for capability in self.disabled_capabilities]
                        ),
                    )
                },
                "required": ["capabilities"],
            },
            function=self._enable_capabilities_tool_function,
            updates_context_data=True,
        )

    async def _enable_capabilities_tool_function(self, capabilities: list[str]) -> str:
        """
        Tool function for enabling the specified capabilities
        """
        try:
            # Enable capabilities in parallel
            await asyncio.gather(*(self.enable_capability(name) for name in capabilities))
        except InvalidCapabilityName as e:
            raise ToolError(str(e)) from e

        return f"Enabled capabilities: [{', '.join(capabilities)}]"

    async def enable_capability(self, name: str) -> None:
        """
        Enables a capability by name
        """
        for capability in self._capabilities:
            if capability.name.lower() == name.lower():
                await capability.enable()
                return

        raise InvalidCapabilityName(f"Capability '{name}' not found")

    async def enable_all_capabilities(self) -> None:
        """
        Enables all capabilities
        """
        await asyncio.gather(*(capability.enable() for capability in self._capabilities))

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
            local_time=datetime.now(),
        )
        return prompt

    async def get_tools(self) -> list[Tool]:
        """
        Returns the tools from the enabled capabilities
        """
        if self.disabled_capabilities:
            tools = [self.get_enable_capabilities_tool()]
        else:
            tools = []

        for capability_tools in await asyncio.gather(
            *(capability.get_tools() for capability in self.enabled_capabilities)
        ):
            tools.extend(capability_tools)
        logger.debug(f"Agent tools: \n{'\n'.join(str(tool) for tool in tools)}")
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

    async def __aenter__(self) -> Self:
        # Setup enabled capabilities when entering context manager
        await asyncio.gather(*(c.setup() for c in self.enabled_capabilities))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.gather(*(c.teardown() for c in self._capabilities))
