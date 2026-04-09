from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_google_genai import ChatGoogleGenerativeAI


class Context(TypedDict):
    user_role: str


basic_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

advanced_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")

    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} You are an expert in your field. Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} You are a beginner. Explain concepts simply and avoid jargon."

    return base_prompt


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = basic_model
    else:
        model = advanced_model

    return handler(request.override(model=model))


agent = create_agent(
    context_schema=Context,
    middleware=[dynamic_model_selection, user_role_prompt],
    model=basic_model,
    name="ai_agent",
)
