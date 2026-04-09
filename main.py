from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

advanced_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = basic_model
    else:
        model = advanced_model

    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    middleware=[dynamic_model_selection],
)
