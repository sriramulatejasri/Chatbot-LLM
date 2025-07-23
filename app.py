import streamlit as st
from chatbot import load_data,model
from vector_store import find_best_match

st.set_page_config(page_title="Custome LLM Chatbot",page_icon="🤖")
st.title("🤖 Custom LLM Chatbot")
st.write("Ask a question based on the knowledge loaded from the application.")
user_input=st.text_input("💬 Enter your question:")
if "prompts" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.prompts,st.session_state.responses,st.session_state.embeddings=load_data()

if user_input:
    response=find_best_match(
        user_input,
        st.session_state.prompts,
        st.session_state.responses,
        st.session_state.embeddings,
        model
    )
    st.markdown("### 🤖 Chatbot says:")
    st.success(response)
