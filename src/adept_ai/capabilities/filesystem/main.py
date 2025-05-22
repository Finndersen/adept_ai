from pathlib import Path
from typing import Annotated

import aiofiles

from adept_ai.capabilities.base import Capability
from adept_ai.capabilities.filesystem.directory_tree import DirectoryTree
from adept_ai.tool import Tool, ToolError
from examples.console import console


class FileSystemCapability(Capability):
    """
    A capability for viewing directory structures and reading & writing files from the file system.
    Root directory defaults to the current working directory.
    """

    name = "Filesystem"
    description = "View the local working directory structure and read & write content of files from the file system. Does not support editing existing files."

    def __init__(
        self,
        root_directory: str | Path | None = None,
        enabled: bool = False,
        initial_directory_depth: int = 3,
        respect_gitignore: bool = True,
    ):
        self.root_directory = (Path(root_directory) if root_directory else Path.cwd()).resolve()
        self.directory_tree = DirectoryTree(self.root_directory, initial_directory_depth, respect_gitignore)
        super().__init__(enabled)

    def get_context_data(self) -> str:
        return f"Current working directory: `{self.root_directory}`  \nDirectory structure:  \n```{self.directory_tree.format_as_paths()}```"

    async def get_tools(self) -> list[Tool]:
        return [
            Tool.from_function(self.create_file, name_prefix=self.name),
            Tool.from_function(self.read_file, name_prefix=self.name),
            Tool.from_function(self.expand_directory, name_prefix=self.name, updates_context_data=True),
        ]

    async def create_file(
        self,
        path: Annotated[str, "The path to create the file at (relative to the working directory)"],
        content: Annotated[str, "The content to write to the file"],
    ) -> str:
        """
        Create a file at the given path with the given content. DO NOT use to overwrite or edit existing files.
        """
        abs_path = self._get_abs_path(path)
        if abs_path.exists():
            raise ToolError(f"File already exists at {path}")

        console.print(f"[bold blue]Creating file: {abs_path}[/bold blue]")

        try:
            async with aiofiles.open(abs_path, "w") as f:
                await f.write(content)
            return f"File created at {path}"
        except OSError as e:
            raise ToolError(f"Error creating file: {repr(e)}") from e

    async def read_file(
        self, rel_path: Annotated[str, "The path to read the file from (relative to the working directory)"]
    ) -> str:
        """
        Read the content of the file at the given path.
        """
        abs_path = self._get_abs_path(rel_path)
        try:
            async with aiofiles.open(abs_path, "r") as f:
                return await f.read()
        except OSError as e:
            raise ToolError(f"Error reading file: {repr(e)}") from e

    async def expand_directory(
        self,
        rel_path: Annotated[str, "The directory path to expand (relative to the working directory)"],
    ) -> str:
        """
        Expand the directory tree to view the requested path that is not currently expanded (marked with [not expanded]).
        Updates the existing directory structure in-place. This is not visible to the user, only you.
        """
        abs_path = self._get_abs_path(rel_path)
        if not abs_path.exists():
            raise ToolError(f"Directory does not exist: {rel_path}")
        if not abs_path.is_dir():
            raise ToolError(f"Path is not a directory: {rel_path}")

        console.print(f"[bold blue]Expanding directory: {abs_path}[/bold blue]")

        # Update the directory tree to show the requested path
        if self.directory_tree.expand_directory(abs_path):
            return f"Expanded directory: {rel_path}"
        else:
            return f"Directory not found or could not be expanded: {rel_path}"

    def _get_abs_path(self, path: str) -> Path:
        # Verify that the path is relative and not absolute
        if Path(path).is_absolute():
            raise ToolError(f"Path must be relative, not absolute: {path}")
        return self.root_directory / path


async def edit_file(
    file_path: Annotated[str, "The path of the file to edit"],
    instructions: Annotated[str, "The instructions for the changes to make to the file"],
) -> str:
    """
    Edit the file at the given path by providing instructions for the changes to make.
    The instructions should be detailed enough for another agent to complete the task.
    When resolving errors in a file, just provide the details of the errors and let the agent decide how to fix them.
    Summarise or simplify the error details if possible (such as excluding the file path),
    but ensure there is enough detail  (such as line numbers and other context) to resolve the errors.
    """
    console.print(f"[bold blue]Editing file: {file_path}[/bold blue]")
    console.print(f"[bold yellow]Instructions:[/bold yellow] {instructions}")

    return f"File edited at {file_path}"
