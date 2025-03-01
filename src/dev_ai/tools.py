import subprocess
from typing import Annotated, cast

from duckduckgo_search import DDGS
from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoSearchTool
from pydantic_ai.tools import Tool
from rich.prompt import Confirm

from dev_ai.deps import AgentDeps

search_tool = Tool(
    DuckDuckGoSearchTool(client=DDGS(), max_results=10).__call__,
    name="search_web",
    description="Searches the web for the given query and returns the results.",
)


def run_bash_command(
    ctx: RunContext[AgentDeps],
    command: Annotated[str, "The bash command to run"],
    destructive: Annotated[bool, "Whether the command is destructive (modifies the system)"] = False,
) -> str:
    """
    Run a bash command and return the output. Only use this tool if the action cannot be completed by other tools.
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
                        ctx.deps.console.print(f"[dim]```\n{line.rstrip()}\n```[/dim]")

                    if process.poll() is not None:
                        break

            if process.returncode != 0 and process.stderr:  # Check if stderr is not None
                stderr = process.stderr.read()
                return f"Command failed with exit code {process.returncode}:\n{stderr}"

        return "\n".join(output)
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        ctx.deps.console.print(f"[red]{error_msg}[/red]")
        return error_msg


def create_file(
    ctx: RunContext[AgentDeps],
    path: Annotated[str, "The path to create the file at"],
    content: Annotated[str, "The content to write to the file"],
) -> str:
    """
    Create a file at the given path with the given content.
    """
    ctx.deps.console.print(f"[bold blue]Creating file: {path}[/bold blue]")
    with open(path, "w") as f:
        f.write(content)
    return f"File created at {path}"


def read_file(ctx: RunContext[AgentDeps], path: Annotated[str, "The path to read the file from"]) -> str:
    """
    Read the content of the file at the given path.
    """
    ctx.deps.console.print(f"[bold blue]Reading file: {path}[/bold blue]")
    try:
        with open(path, "r") as f:
            content = f.read()

            return content
    except FileNotFoundError:
        ctx.deps.console.print(f"[red]File not found at {path}[/red]")
        return f"File not found at {path}"
