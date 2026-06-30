import streamlit as st

from models.rag_pipeline import chat


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="GUVI Multilingual GPT Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ----------------------------
# Title
# ----------------------------

st.title("🤖 GUVI Multilingual GPT Chatbot")

st.write(
    "Ask questions about GUVI in any language 🌐"
)


# ----------------------------
# Chat History
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )


# ----------------------------
# User Input
# ----------------------------

user_question = st.chat_input(
    "Ask your question..."
)


if user_question:


    # Store user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    with st.chat_message("user"):
        st.write(user_question)



    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            response = chat(
                user_question
            )


            st.write(
                response
            )


    # Store bot response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
