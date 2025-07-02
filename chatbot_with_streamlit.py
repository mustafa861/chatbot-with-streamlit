import os
import streamlit as st
import asyncio
from dotenv import load_dotenv, find_dotenv

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig


load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")


provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent = Agent(
    instructions="You are a helpful AI assistant that provides clear and accurate responses to user queries.",
    name="Assistant",
)


st.set_page_config(page_title="Conversational UI", page_icon="💬")

st.markdown("""
    <style>
        .chat-title {
            text-align: center;
            font-weight: bold;
            font-size: 88px;
            margin-bottom: 40px;
        }

        .block-container {
            max-width: 1000px;
            margin: auto;
        }

        section.main > div:has(div[data-testid="stChatInput"]) {
            max-width: 1000px;
            margin: 30px auto;
        }

        textarea[data-testid="stChatInput"] {
            min-height: 60px !important;
            font-size: 1.2em !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-title">NeuroBot</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    async def get_agent_response(user_input):
        result = await Runner.run(agent, user_input, run_config=run_config)
        return result.final_output

    response = asyncio.run(get_agent_response(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
