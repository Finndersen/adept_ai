import subprocess
from typing import Annotated

from duckduckgo_search import DDGS
from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult, DuckDuckGoSearchTool
from rich.prompt import Confirm

from adept_ai.tool import ToolError
from examples.console import console
from examples.pydantic_ai.deps import AgentDeps


async def search_web(query: Annotated[str, "The search query"]) -> list[DuckDuckGoResult]:
    """
    Searches the web for the given query and returns the results.
    """
    ddg_tool = DuckDuckGoSearchTool(client=DDGS(), max_results=10)

    results = await ddg_tool(query)
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
        if not Confirm.ask(f"Run potentially destructive command: {command}?", default=False, console=console):
            return "Command cancelled by user"

    try:
        # Create a panel to display command output
        with console.status("", spinner="bouncingBall"):
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
                        console.print(f"[dim]{line.rstrip()}[/dim]")

                    # Read any remaining output after process ends
                    if process.poll() is not None:
                        remaining = process.stdout.read()
                        if remaining:
                            for line in remaining.splitlines():
                                output.append(line + "\n")
                                console.print(f"[dim]{line}[/dim]")
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
            raise ToolError(error_msg)

        return "\n".join(output)
    except OSError as e:
        raise ToolError(f"Error executing command: {repr(e)}") from e
