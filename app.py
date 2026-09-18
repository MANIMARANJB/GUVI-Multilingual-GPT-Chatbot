import streamlit as st

from src.chatbot import chatbot_response


# PAGE CONFIG

st.set_page_config(
    page_title="GUVI Multilingual AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# TITLE

st.title("GUVI Multilingual AI Assistant")

st.caption(
    "Ask questions about GUVI courses and learning programs "
    "in English, Tamil, Hindi, Telugu, Kannada, or Malayalam."
)


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []


# SIDEBAR

with st.sidebar:
    st.header("About")

    st.write(
        "This chatbot uses language detection, translation, "
        "semantic search, FAISS retrieval, and a local language model "
        "to answer questions using GUVI website content."
    )

    st.subheader("Supported Languages")

    st.write(
        "English\n\n"
        "Tamil\n\n"
        "Hindi\n\n"
        "Telugu\n\n"
        "Kannada\n\n"
        "Malayalam"
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# DISPLAY CHAT HISTORY

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# CHAT INPUT

user_input = st.chat_input(
    "Ask something about GUVI..."
)


# PROCESS USER MESSAGE

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Generating response..."):

            try:
                result = chatbot_response(user_input)

                final_response = result["final_response"]

                st.markdown(final_response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response
                    }
                )

            except Exception as error:

                error_message = (
                    "Sorry, I couldn't process your request. "
                    "Please try again."
                )

                st.error(error_message)

                print("Chatbot Error:", error)