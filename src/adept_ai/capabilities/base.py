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
