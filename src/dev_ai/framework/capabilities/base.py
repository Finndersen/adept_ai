from abc import ABC

from dev_ai.framework.tool import Tool


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
    def prompt_instructions(self) -> list[str]:
        """
        Returns the list instructions for the capability, to be added to the system prompt
        """
        return []

    @property
    def prompt_examples(self) -> list[str]:
        """
        Returns a list of usage examples for the capability, to be added to the system prompt
        """
        return []

    async def get_context_data(self) -> str:
        """
        Returns any relevant contextual data for the capability, to be added to the system prompt
        """
        return ""

    async def setup(self) -> None:
        """
        Perform any necessary setup or pre-processing required before tools or context data can be provided.
        :return:
        """
        pass

    async def teardown(self) -> None:
        """
        Perform any necessary teardown or cleanup
        :return:
        """
        pass
