import streamlit as st

def render_auth_sidebar():
    with st.sidebar:
        st.title("🔐 Authentication")
        
        if st.session_state.get("user"):
            st.success(f"Welcome, {st.session_state['user']}!")
            if st.button("Logout", use_container_width=True):
                st.session_state.pop("user", None)
                st.rerun()
        else:
            st.info("Please log in to access the dashboard.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True):
                if username and password:
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Please enter a username and password.")