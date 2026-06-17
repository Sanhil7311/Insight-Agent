import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine, text

def _save_dataframe(df: pd.DataFrame, filename: str) -> str:
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", filename)
    df.to_csv(save_path, index=False)
    return save_path

def _render_file_upload_tab() -> str | None:
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is None:
        return None

    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        st.dataframe(df.head())

        save_path = _save_dataframe(df, uploaded_file.name if uploaded_file.name.endswith(".csv") else uploaded_file.name.rsplit(".", 1)[0] + ".csv")
        st.success(f"File saved to `{save_path}`")

        return save_path

    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return None

def _build_connection_url(host: str, port: str, dbname: str, user: str, password: str) -> str:
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

def _render_sql_connection_tab() -> str | None:
    st.subheader("PostgreSQL Connection")

    col1, col2 = st.columns([3, 1])
    with col1:
        host = st.text_input("Host", value="localhost", key="pg_host")
    with col2:
        port = st.text_input("Port", value="5432", key="pg_port")

    dbname = st.text_input("Database Name", key="pg_dbname")

    col3, col4 = st.columns(2)
    with col3:
        user = st.text_input("Username", key="pg_user")
    with col4:
        password = st.text_input("Password", type="password", key="pg_password")

    sql_query = st.text_area(
        "SQL Query",
        value="SELECT * FROM your_table LIMIT 1000;",
        height=120,
        key="pg_query",
    )

    credentials_provided = all([host, port, dbname, user, password, sql_query.strip()])

    col_test, col_ingest = st.columns(2)

    with col_test:
        if st.button("🔌 Test Connection", use_container_width=True, disabled=not credentials_provided):
            try:
                engine = create_engine(_build_connection_url(host, port, dbname, user, password), connect_args={"connect_timeout": 5})
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                st.success("Connection successful!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    with col_ingest:
        if st.button("📥 Ingest Data", use_container_width=True, type="primary", disabled=not credentials_provided):
            try:
                engine = create_engine(_build_connection_url(host, port, dbname, user, password), connect_args={"connect_timeout": 10})
                with st.spinner("Fetching data from database..."):
                    with engine.connect() as conn:
                        df = pd.read_sql(text(sql_query), conn)

                st.dataframe(df.head())
                st.info(f"Fetched **{len(df):,} rows** and **{len(df.columns)} columns**.")

                safe_dbname = "".join(c if c.isalnum() else "_" for c in dbname)
                filename = f"sql_{safe_dbname}.csv"
                save_path = _save_dataframe(df, filename)
                st.success(f"Data saved to `{save_path}`")
                st.session_state.sql_save_path = save_path

            except Exception as e:
                st.error(f"Data ingestion failed: {e}")
                st.session_state.pop("sql_save_path", None)

    if "sql_save_path" in st.session_state:
        return st.session_state.sql_save_path

    return None

def render_dashboard() -> str | None:
    st.header("Data Ingestion Dashboard")

    tab_file, tab_sql = st.tabs(["📁 File Upload", "🗄️ SQL Connection"])

    with tab_file:
        save_path = _render_file_upload_tab()
        if save_path:
            st.session_state.pop("sql_save_path", None)
            return save_path

    with tab_sql:
        save_path = _render_sql_connection_tab()
        if save_path:
            return save_path

    return None