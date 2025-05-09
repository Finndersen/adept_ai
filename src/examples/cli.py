#!/usr/bin/env python3

import argparse
import asyncio
from pathlib import Path

import logfire
from dotenv import load_dotenv


def main():
    """Command-line interface for dev_ai."""
    parser = argparse.ArgumentParser(description="Run an AI development assistant with a specified LLM model.")

    parser.add_argument(
        "prompt",
        type=str,
        help="Request prompt to provide to agent",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name to use, e.g. gpt-4o, claude-3-7-sonnet-latest, gemini-2.0-flash. "
        "Can infer from environment variable API keys if not provided.",
    )

    parser.add_argument(
        "--framework",
        type=str,
        choices=["pydantic_ai", "langchain", "openai"],
        default="pydantic_ai",
        help="Which framework example to test",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM service. If not provided, will try to use environment variable.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, which will print debug messages to the console.",
    )

    args = parser.parse_args()

    logfire.configure(send_to_logfire="if-token-present", console=None if args.debug else False)

    # Run the assistant
    from examples.langchain.run import run_langchain
    from examples.openai.run import run_openai
    from examples.pydantic_ai.run import run_pydantic_ai

    RUN_FUNCS = {
        "pydantic_ai": run_pydantic_ai,
        "langchain": run_langchain,
        "openai": run_openai,
    }
    asyncio.run(RUN_FUNCS[args.framework](prompt=args.prompt, model_name=args.model, api_key=args.api_key))

    return 0


if __name__ == "__main__":
    # Load environment variables from .env files
    # Try loading from home directory first, then current directory
    env_paths = [Path.home() / ".env", Path.cwd() / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)

    exit(main())
