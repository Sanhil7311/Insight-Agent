import streamlit as st
import pandas as pd

try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_community.llms import Ollama
    from langchain.agents import AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def _load_dataframe(file_path: str) -> pd.DataFrame | None:
    try:
        if file_path.endswith(".csv"):
            try:
                return pd.read_csv(file_path)
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding="latin1")
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Failed to load dataset for chat: {e}")
        return None

def _get_agent(df: pd.DataFrame):
    llm = Ollama(
        model="qwen2.5-coder:3b",
        base_url="http://127.0.0.1:11434",
        temperature=0.1,
    )
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

def render_data_chat(file_path: str):
    st.markdown("---")
    st.markdown("## 💬 Ask Your Data")
    st.caption(f"Querying: `{file_path}`")

    if not LANGCHAIN_AVAILABLE:
        st.error("LangChain packages are missing. Please install them to use this feature:\n\n`pip install langchain langchain-community langchain-experimental`")
        return

    history_key = f"chat_history_{file_path}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for message in st.session_state[history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about your data...")

    if user_input:
        st.session_state[history_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                df = _load_dataframe(file_path)
                if df is None:
                    answer = "I could not load the dataset. Please re-upload the file and run the pipeline again."
                else:
                    try:
                        agent = _get_agent(df)
                        answer = agent.run(user_input)
                    except Exception as e:
                        answer = f"I encountered an error while analyzing your data: {str(e)}"
            st.markdown(answer)

        st.session_state[history_key].append({"role": "assistant", "content": answer})