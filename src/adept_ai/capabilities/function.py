from typing import Sequence

from adept_ai.capabilities.base import ProvidedConfigCapability
from adept_ai.tool import Tool, ToolFunction


class FunctionToolsCapability(ProvidedConfigCapability):
    """
    Capability that provides a collection of tools from functions or callables.

    This capability allows you to:
    - Provide a sequence of Tool objects directly
    - Provide functions/callables which will be converted to tools upon initialization
    - Specify name, description, instructions, and usage examples for the capability
    """

    def __init__(
        self,
        tools_or_functions: Sequence[Tool | ToolFunction],
        name: str,
        description: str,
        instructions: list[str] = None,
        usage_examples: list[str] = None,
        enabled: bool = False,
    ):
        """
        Initialize the FunctionToolsCapability.

        Args:
            tools_or_functions: A sequence of Tool objects or functions/callables to be converted to tools
            name: Name of the capability
            description: Description of the capability
            instructions: List of instructions for using the capability
            usage_examples: List of usage examples for the capability
            enabled: Whether the capability is initially enabled
        """
        super().__init__(
            name=name,
            description=description,
            instructions=instructions,
            usage_examples=usage_examples,
            enabled=enabled,
        )

        # Convert functions to tools immediately
        self._tools = []
        for item in tools_or_functions:
            if isinstance(item, Tool):
                tool = item
            else:
                tool = Tool.from_function(function=item, name_prefix=self.name)
            self._tools.append(tool)

    async def get_tools(self) -> list[Tool]:
        """
        Returns a list of tools that the capability provides.

        All functions are already converted to Tool objects during initialization.
        """
        return self._tools
