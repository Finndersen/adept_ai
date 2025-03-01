import os
import subprocess
from typing import Annotated

from duckduckgo_search import DDGS
from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult, DuckDuckGoSearchTool
from rich.prompt import Confirm

from dev_ai.deps import AgentDeps


async def search_web(ctx: RunContext[AgentDeps], query: Annotated[str, "The search query"]) -> list[DuckDuckGoResult]:
    """
    Searches the web for the given query and returns the results.
    """
    ctx.deps.console.print(f"[bold blue]Searching web for: {query}[/bold blue]")
    ddg_tool = DuckDuckGoSearchTool(client=DDGS(), max_results=10)

    results = await ddg_tool(query)
    print(results)
    return results


async def run_bash_command(
    ctx: RunContext[AgentDeps],
    command: Annotated[str, "The bash command to run"],
    destructive: Annotated[bool, "Whether the command is destructive (modifies the system)"] = False,
) -> str:
    """
    Run a bash command and return the output (up to 100 lines). The output will also be displayed to the user.
    Only use this tool if the action cannot be completed by other tools.
    Set destructive to True if it is possible that the command will modify the system.
    """
    if destructive:
        if not Confirm.ask(f"Run potentially destructive command: {command}?", default=False, console=ctx.deps.console):
            return "Command cancelled by user"

    try:
        # Create a panel to display command output
        ctx.deps.console.print(f"[bold blue]$ {command}[/bold blue]")
        with ctx.deps.console.status("", spinner="bouncingBall"):
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=ctx.deps.current_working_directory,
                bufsize=1,
                universal_newlines=True,
            )

            output = []
            # Stream output in real-time
            if process.stdout:  # Check if stdout is not None
                while True:
                    line = process.stdout.readline()
                    if line:
                        output.append(line)
                        ctx.deps.console.print(f"[dim]{line.rstrip()}[/dim]")

                    # Read any remaining output after process ends
                    if process.poll() is not None:
                        remaining = process.stdout.read()
                        if remaining:
                            for line in remaining.splitlines():
                                output.append(line + '\n')
                                ctx.deps.console.print(f"[dim]{line}[/dim]")
                        break
                # Limit output to 100 lines
        if len(output) > 100:
            output = output[:100]
            output.append("... (output truncated)")

        if process.returncode != 0:
            output_str = "\n".join(output)
            error_msg = f"Command failed with exit code {process.returncode}:\nOutput:\n{output_str}"
            if process.stderr:  # Check if stderr is not None
                stderr = process.stderr.read()
                if stderr:  # Only add error section if there was stderr output
                    error_msg += f"\nError:\n{stderr}"
            return error_msg

        return "\n".join(output)
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        ctx.deps.console.print(f"[red]{error_msg}[/red]")
        return error_msg


async def create_file(
    ctx: RunContext[AgentDeps],
    path: Annotated[str, "The path to create the file at"],
    content: Annotated[str, "The content to write to the file"],
) -> str:
    """
    Create a file at the given path with the given content. DO NOT use this tool to overwrite or edit existing files.
    """
    if os.path.exists(path):
        error_msg = f"File already exists at {path}"
        ctx.deps.console.print(f"[red]{error_msg}[/red]")
        return error_msg

    ctx.deps.console.print(f"[bold blue]Creating file: {path}[/bold blue]")

    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File created at {path}"
    except OSError as e:
        error_msg = f"Error creating file: {repr(e)}"
        ctx.deps.console.print(f"[red]{error_msg}[/red]")
        return error_msg


async def read_file(ctx: RunContext[AgentDeps], path: Annotated[str, "The path to read the file from"]) -> str:
    """
    Read the content of the file at the given path.
    """
    ctx.deps.console.print(f"[bold blue]Reading file: {path}[/bold blue]")
    try:
        with open(path, "r") as f:
            content = f.read()

            return content
    except OSError as e:
        error_msg = f"Error reading file: {repr(e)}"
        ctx.deps.console.print(f"[red]{error_msg}[/red]")
        return error_msg


async def edit_file(ctx: RunContext[AgentDeps],
                    file_path: Annotated[str, "The path of the file to edit"],
                    instructions: Annotated[str, "The instructions for the changes to make to the file"]) -> str:
    """
    Edit the file at the given path by providing instructions for the changes to make.
    The instructions should be detailed enough for another agent to complete the task.
    When fixing errors reported in a file, just provide the details of the errors and let the agent work out how to fix them.
    Summarise or simplify the error details if possible (such as excluding the file path), 
    but ensure there is enough detail  (such as line numbers and other context) to resolve the errors. 
    """
    ctx.deps.console.print(f"[bold blue]Editing file: {file_path}[/bold blue]")
    ctx.deps.console.print(f"[bold yellow]Instructions:[/bold yellow] {instructions}")

    return f"File edited at {file_path}"


async def fix_file_errors(ctx: RunContext[AgentDeps],
                    file_path: Annotated[str, "The path of the file with errors"],
                    error_details: Annotated[str, "Details of errors in the file to be resolved"]) -> str:
    """
    Report errors or issues in a file to be resolved by another agent. 
    Should be used instead of edit_file() for linting, typing, exceptions and other types of errors.
    Summarise or simplify the error details if possible (such as excluding the file path), 
    but ensure there is enough detail  (such as line numbers and other context) to resolve the errors.
    """
    ctx.deps.console.print(f"[bold blue]Reporting errors in file: {file_path}[/bold blue]")
    ctx.deps.console.print(f"[bold yellow]Error details:[/bold yellow] {error_details}")

    return f"Errors reported in file {file_path}"

