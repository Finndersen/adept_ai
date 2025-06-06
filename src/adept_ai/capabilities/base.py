from abc import ABC
from typing import Self

from adept_ai.tool import Tool


class Capability(ABC):
    """
    Base class for a capability, which represents a collection of tools and behaviours that an agent can use to perform tasks,
    along with associated instructions and usage examples.
    """

    name: str
    description: str
    enabled: bool

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def get_tools(self) -> list[Tool]:
        """
        Returns a list of tools that the capability provides.
        """
        raise NotImplementedError

    @property
    def instructions(self) -> list[str] | None:
        """
        Returns the list instructions for the capability, to be added to the system prompt
        """
        return None

    @property
    def usage_examples(self) -> list[str]:
        """
        Returns a list of usage examples for the capability, to be added to the system prompt
        """
        return []

    async def get_context_data(self) -> str:
        """
        Returns any relevant contextual data for the capability, to be added to the system prompt
        """
        return ""

    async def enable(self) -> None:
        if self.enabled:
            return

        self.enabled = True
        await self.setup()

    async def disable(self) -> None:
        self.enabled = False

    async def setup(self) -> None:  # noqa: B027
        """
        Perform any necessary setup or pre-processing required before tools or context data can be provided.
        :return:
        """
        pass

    async def teardown(self) -> None:  # noqa: B027
        """
        Perform any necessary teardown or cleanup.
        May be called when the capability was never enabled or setup() not called
        :return:
        """
        pass

    async def __aenter__(self) -> Self:
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()


class ProvidedConfigCapability(Capability):
    """
    Base class for a capability with provided configuration.

    This capability allows you to specify:
    - Name and description for the capability
    - Instructions for using the capability
    - Usage examples for the capability
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: list[str] = None,
        usage_examples: list[str] = None,
        enabled: bool = False,
    ):
        """
        Initialize the ProvidedConfigCapability.

        Args:
            name: Name of the capability
            description: Description of the capability
            instructions: List of instructions for using the capability
            usage_examples: List of usage examples for the capability
            enabled: Whether the capability is initially enabled
        """
        super().__init__(enabled=enabled)
        self.name = name
        self.description = description
        self._instructions = instructions or []
        self._usage_examples = usage_examples or []

    @property
    def instructions(self) -> list[str] | None:
        """
        Returns the list of instructions for the capability.
        """
        return self._instructions

    @property
    def usage_examples(self) -> list[str]:
        """
        Returns the list of usage examples for the capability.
        """
        return self._usage_examples
