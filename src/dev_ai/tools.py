import asyncio
import subprocess
from typing import Annotated

from duckduckgo_search import DDGS
from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult, DuckDuckGoSearchTool
from pydantic_ai.tools import Tool
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
    Create a file at the given path with the given content.
    """
    ctx.deps.console.print(f"[bold blue]Creating file: {path}[/bold blue]")
    with open(path, "w") as f:
        f.write(content)
    return f"File created at {path}"


async def read_file(ctx: RunContext[AgentDeps], path: Annotated[str, "The path to read the file from"]) -> str:
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
