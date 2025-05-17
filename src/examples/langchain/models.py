import os

from langchain_core.language_models import BaseChatModel


def get_model_from_name_and_api_key(model_name: str | None, api_key: str | None = None) -> BaseChatModel:
    if not model_name:
        if os.environ.get("OPENAI_API_KEY"):
            model_name = "gpt-4o"
            api_key = os.environ.get("OPENAI_API_KEY")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            model_name = "claude-3-7-sonnet-latest"
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif os.environ.get("GEMINI_API_KEY"):
            model_name = "gemini-2.0-flash"
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            raise ValueError("No model name provided and no API keys found in environment")

    if model_name.startswith("gpt-"):
        from langchain_openai import ChatOpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, openai_api_key=api_key)

    elif model_name.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, anthropic_api_key=api_key)

    elif model_name.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
