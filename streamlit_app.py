import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("🌸 일본 문화 및 관광 전문가 챗봇")
st.write(
    "이 챗봇은 일본 문화, 관광, 전통, 현대 생활에 대한 전문적인 정보를 제공합니다. "
    "OpenAI의 GPT-4 모델을 활용하며, 일본에 대해 궁금한 모든 것을 물어보세요! "
    "OpenAI API 키가 필요하며, [여기](https://platform.openai.com/account/api-keys)에서 발급받을 수 있습니다."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in Japanese culture and tourism. Provide detailed "
                    "and professional insights on topics related to Japan, including its "
                    "traditional culture, modern society, travel destinations, cuisine, festivals, "
                    "and history. Respond in a friendly and engaging manner."
                ),
            }
        ]

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("일본에 대해 무엇이든 물어보세요!"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
