from __future__ import annotations

import concurrent.futures
import io
import logging
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

from ui.auth import render_auth_sidebar
from ui.dashboard import render_dashboard
from ui.data_chat import render_data_chat
from logic.crew_pipeline import run_local_crew
from logic.export_engine import build_pptx_bytes

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Insight-Agent",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def get_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)

_CREW_EXECUTOR = get_executor()

_PLOTLY_DISPATCH: dict[str, Any] = {
    "bar": px.bar,
    "scatter": px.scatter,
    "histogram": px.histogram,
    "box": px.box,
    "line": px.line,
    "violin": px.violin,
    "pie": px.pie,
    "area": px.area,
    "heatmap": None,
}

_AGENT_COLORS: dict[str, str] = {
    "Pipeline": "#6c757d",
    "Data Cleaner": "#0d6efd",
    "ML Analyst": "#198754",
    "Model Engineer": "#fd7e14",
    "Business Storyteller": "#dc3545",
    "Slides Formatter": "#6f42c1",
}

_AGENT_ICONS: dict[str, str] = {
    "Pipeline": "⚙️",
    "Data Cleaner": "🧹",
    "ML Analyst": "📊",
    "Model Engineer": "🤖",
    "Business Storyteller": "✍️",
    "Slides Formatter": "🖼️",
}

@dataclass
class CrewPipelineState:
    status: str = "idle"
    result: dict[str, Any] | None = None
    error: str = ""
    future: concurrent.futures.Future | None = field(default=None, repr=False)
    submitted_at: float = field(default_factory=time.monotonic, repr=False)
    step_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)

def _submit_crew_pipeline(
    file_path: str,
    target_column: str | None,
    task_type: str | None,
) -> CrewPipelineState:
    state = CrewPipelineState(status="running", submitted_at=time.monotonic())
    state.future = _CREW_EXECUTOR.submit(
        run_local_crew, file_path, state.step_queue, target_column, task_type
    )
    logger.info(
        "CrewAI pipeline submitted | file=%r | target=%r | task_type=%r",
        file_path, target_column, task_type,
    )
    return state

def _poll_crew_state(state: CrewPipelineState) -> CrewPipelineState:
    if state.future is None:
        return state
    if not state.future.done():
        return state
    elapsed = time.monotonic() - state.submitted_at
    try:
        payload: dict[str, Any] = state.future.result()
        payload["elapsed_seconds"] = round(elapsed, 2)
        state.result = payload
        state.status = "complete"
        logger.info("CrewAI pipeline complete | elapsed=%.1fs", elapsed)
    except FileNotFoundError as exc:
        state.status = "error"
        state.error = (
            f"Dataset file not found: {exc}. "
            "Please re-upload the file and try again."
        )
        logger.error("CrewAI pipeline error (FileNotFoundError): %s", exc)
    except ValueError as exc:
        state.status = "error"
        state.error = f"Invalid input data: {exc}"
        logger.error("CrewAI pipeline error (ValueError): %s", exc)
    except RuntimeError as exc:
        state.status = "error"
        state.error = (
            f"CrewAI runtime error after {elapsed:.1f}s: {exc}. "
            "Verify that Ollama is running and the model is loaded."
        )
        logger.error("CrewAI pipeline error (RuntimeError): %s", exc)
    except Exception as exc:
        state.status = "error"
        state.error = (
            f"Pipeline failed after {elapsed:.1f}s — {type(exc).__name__}: {exc}. "
            "Check application logs for the full traceback."
        )
        logger.exception("CrewAI pipeline error (unhandled): %s", exc)
    finally:
        state.future = None
    return state

def _drain_step_queue(state: CrewPipelineState) -> None:
    if "workspace_log" not in st.session_state:
        st.session_state.workspace_log = []
    drained = 0
    while drained < 50:
        try:
            event = state.step_queue.get_nowait()
            st.session_state.workspace_log.append(event)
            drained += 1
        except queue.Empty:
            break

def _render_workspace_log(log: list[dict]) -> None:
    if not log:
        st.caption("Waiting for first agent output…")
        return
    for event in log:
        agent = event.get("agent", "Pipeline")
        kind = event.get("type", "info")
        content = event.get("content", "")
        if not content:
            continue
        icon = _AGENT_ICONS.get(agent, "🤖")
        color = _AGENT_COLORS.get(agent, "#6c757d")
        if kind == "task_complete":
            label = f"{icon} **{agent}** — task output"
            with st.expander(label, expanded=False):
                st.markdown(content)
        elif kind == "step":
            label = f"{icon} **{agent}** — reasoning step"
            with st.expander(label, expanded=False):
                st.markdown(content)
        else:
            st.markdown(
                f'<span style="color:{color};font-size:0.85rem">'
                f'{icon} <b>{agent}</b>: {content}'
                f'</span>',
                unsafe_allow_html=True,
            )

def _load_dataframe_for_charts(file_path: str) -> pd.DataFrame | None:
    try:
        if file_path.endswith(".csv"):
            try:
                return pd.read_csv(file_path)
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding="latin1")
        return pd.read_excel(file_path)
    except Exception as exc:
        st.warning(f"Could not load dataset for chart rendering: {exc}")
        return None

def _render_chart_from_spec(spec: dict, df: pd.DataFrame) -> None:
    chart_type = str(spec.get("chart_type", "")).lower().strip()
    x_col: str | None = spec.get("x_axis") or None
    y_col: str | None = spec.get("y_axis") or None
    title: str = spec.get("title", chart_type.capitalize())
    if x_col and x_col not in df.columns:
        x_col = None
    if y_col and y_col not in df.columns:
        y_col = None
    try:
        if chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                st.info(f"Not enough numeric columns to render heatmap: {title}")
                return
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, title=title, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            return
        plot_fn = _PLOTLY_DISPATCH.get(chart_type)
        if plot_fn is None:
            st.info(f"Unsupported chart type '{chart_type}' — skipping: {title}")
            return
        kwargs: dict[str, Any] = {"data_frame": df, "title": title}
        if x_col:
            kwargs["x"] = x_col
        if y_col:
            kwargs["y"] = y_col
        fig = plot_fn(**kwargs)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render chart '{title}': {exc}")

def _render_automl_config(file_path: str) -> tuple[str | None, str | None]:
    st.markdown("---")
    st.markdown("### 🤖 AutoML — Predictive Modelling (Optional)")
    st.caption(
        "Select a target column to train a baseline model and include predictive "
        "metrics and forecasts in the report. Leave blank to skip."
    )
    df_preview: pd.DataFrame | None = _load_dataframe_for_charts(file_path)
    if df_preview is None:
        st.warning("Could not load dataset to populate column selector.")
        return None, None
    column_options = ["— Skip AutoML —"] + df_preview.columns.tolist()
    selected_col = st.selectbox(
        "Target Column for Prediction",
        options=column_options,
        index=0,
        key="automl_target_col",
    )
    if selected_col == "— Skip AutoML —":
        return None, None
    task_type = st.radio(
        "Task Type",
        options=["Classification", "Regression"],
        index=0,
        horizontal=True,
        key="automl_task_type",
    )
    st.info(
        f"A RandomForest model will be trained to predict **{selected_col}** "
        f"as a **{task_type}** task."
    )
    return selected_col, task_type.lower()

def _render_automl_results(predictive_metrics: dict) -> None:
    if not predictive_metrics or predictive_metrics.get("skipped"):
        return
    error = predictive_metrics.get("error")
    if error:
        st.error(f"AutoML encountered an error: {error}")
        return
    task = predictive_metrics.get("task_type", "unknown")
    target = predictive_metrics.get("target_column", "unknown")
    model_name = predictive_metrics.get("model", "RandomForest")
    n_train = predictive_metrics.get("train_samples", 0)
    n_test = predictive_metrics.get("test_samples", 0)
    st.markdown(
        f"**Model:** {model_name} &nbsp;|&nbsp; "
        f"**Target:** `{target}` &nbsp;|&nbsp; "
        f"**Task:** {task.capitalize()} &nbsp;|&nbsp; "
        f"**Train/Test Split:** {n_train:,} / {n_test:,}"
    )
    metrics = predictive_metrics.get("metrics", {})
    if metrics:
        metric_cols = st.columns(len(metrics))
        for col_ctx, (metric_name, metric_val) in zip(metric_cols, metrics.items()):
            with col_ctx:
                display_val = (
                    f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
                )
                st.metric(label=metric_name.upper().replace("_", " "), value=display_val)
    feature_importances: dict = predictive_metrics.get("feature_importances", {})
    if feature_importances:
        st.markdown("**Top Feature Importances**")
        fi_df = pd.DataFrame(
            list(feature_importances.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature Importances for predicting '{target}'",
        )
        st.plotly_chart(fig, use_container_width=True)

def _build_pdf_bytes(report_markdown: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()
    heading1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
    )
    heading2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontSize=8,
        leading=12,
        backColor=colors.HexColor("#f4f4f4"),
        spaceAfter=8,
    )
    chart_note_style = ParagraphStyle(
        "ChartNote",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
        backColor=colors.HexColor("#fffbe6"),
        spaceAfter=10,
        borderPadding=6,
    )
    story = []
    story.append(Paragraph("Insight-Agent: Analytical Report", heading1_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Note: Interactive Plotly charts are available in the web dashboard. "
            "This PDF export contains the written analytical report only.",
            chart_note_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    for line in report_markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.05 * inch))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], heading2_style))
        elif stripped.startswith("## ") or stripped.startswith("# "):
            text = stripped.lstrip("#").strip()
            story.append(Paragraph(text, heading1_style))
        elif stripped.startswith("```") or stripped.startswith("    "):
            story.append(Preformatted(stripped.lstrip("`"), code_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph(f"&bull; {stripped[2:]}", body_style))
        elif stripped.startswith("**") and stripped.endswith("**"):
            story.append(Paragraph(f"<b>{stripped[2:-2]}</b>", body_style))
        else:
            safe = (
                stripped
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            story.append(Paragraph(safe, body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _build_pptx_from_pipeline_result(
    pipeline_result: dict,
    analyzed_path: str | None,
) -> bytes:
    slide_json: list[dict] = pipeline_result.get("slide_json", [])
    chart_specs: list[dict] = pipeline_result.get("chart_recommendations", [])
    file_name: str = pipeline_result.get("file_path", "dataset")
    file_name = os.path.basename(file_name)
    analytics: dict = pipeline_result.get("analytics", {})
    predictive_metrics: dict = pipeline_result.get("predictive_metrics", {})
    report_markdown: str = pipeline_result.get("report", "")

    df: pd.DataFrame | None = None
    if analyzed_path and os.path.exists(analyzed_path):
        try:
            if analyzed_path.endswith(".csv"):
                try:
                    df = pd.read_csv(analyzed_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(analyzed_path, encoding="latin1")
            else:
                df = pd.read_excel(analyzed_path)
        except Exception as exc:
            logger.warning("Could not load DataFrame for PPTX chart rendering: %s", exc)

    if not slide_json:
        slide_json = _fallback_slide_json_from_markdown(report_markdown)

    return build_pptx_bytes(
        slide_json=slide_json,
        chart_specs=chart_specs,
        df=df,
        file_name=file_name,
        analytics=analytics,
        predictive_metrics=predictive_metrics,
        report_markdown=report_markdown,
    )


def _fallback_slide_json_from_markdown(report_markdown: str) -> list[dict]:
    slides: list[dict] = []
    current_title: str | None = None
    current_bullets: list[str] = []
    for line in report_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            if current_title is not None:
                slides.append({"title": current_title, "bullets": current_bullets[:6]})
            current_title = stripped.lstrip("#").strip()
            current_bullets = []
        elif stripped.startswith("- ") or stripped.startswith("* "):
            current_bullets.append(stripped[2:].strip())
        elif stripped and current_title and not stripped.startswith("#"):
            if len(current_bullets) < 6:
                current_bullets.append(stripped[:120])
    if current_title is not None:
        slides.append({"title": current_title, "bullets": current_bullets[:6]})
    return slides


def _render_final_report(pipeline_result: dict) -> None:
    st.success("✅ CrewAI Pipeline Complete!")
    elapsed: float = pipeline_result.get("elapsed_seconds", 0.0)
    if elapsed:
        st.caption(f"⏱ Total crew execution time: {elapsed:.1f}s")
    st.markdown("---")
    st.markdown("## 📋 Results")
    analyzed_path: str | None = (
        st.session_state.get("analyzed_file_path")
        or pipeline_result.get("file_path")
    )
    tab_automl, tab_charts, tab_report, tab_agent, tab_chat = st.tabs([
        "🤖 AutoML",
        "📊 Charts",
        "📄 Report",
        "🧠 Agent Workflow",
        "💬 Data Chat",
    ])
    with tab_automl:
        predictive_metrics: dict = pipeline_result.get("predictive_metrics", {})
        if predictive_metrics and not predictive_metrics.get("skipped"):
            _render_automl_results(predictive_metrics)
        else:
            st.info("AutoML was skipped or no predictive metrics are available for this run.")
    with tab_charts:
        chart_recommendations: list[dict] = pipeline_result.get("chart_recommendations", [])
        if chart_recommendations:
            df: pd.DataFrame | None = None
            if analyzed_path and os.path.exists(analyzed_path):
                df = _load_dataframe_for_charts(analyzed_path)
            if df is not None and not df.empty:
                valid_specs = [s for s in chart_recommendations if isinstance(s, dict)]
                if valid_specs:
                    cols_per_row = 2
                    for i in range(0, len(valid_specs), cols_per_row):
                        row_specs = valid_specs[i: i + cols_per_row]
                        cols = st.columns(len(row_specs))
                        for col_ctx, spec in zip(cols, row_specs):
                            with col_ctx:
                                _render_chart_from_spec(spec, df)
                else:
                    st.info("No valid chart specifications were returned by the pipeline.")
            else:
                st.warning(
                    "Dataset not available for chart rendering. "
                    "Charts require the source file to be present on disk."
                )
                for spec in chart_recommendations:
                    if isinstance(spec, dict):
                        st.markdown(
                            f"- **{spec.get('title', 'Untitled')}** — "
                            f"`{spec.get('chart_type', 'N/A')}` | "
                            f"x: `{spec.get('x_axis', '-')}` | "
                            f"y: `{spec.get('y_axis', '-')}`"
                        )
        else:
            st.info("No charts were recommended for this dataset.")
    with tab_report:
        report_text: str = pipeline_result.get("report", "")
        st.markdown(report_text)
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        export_col_pdf, export_col_pptx = st.columns(2)
        with export_col_pdf:
            try:
                pdf_bytes = _build_pdf_bytes(report_text)
                st.download_button(
                    label="⬇️ Download Report as PDF",
                    data=pdf_bytes,
                    file_name="insight_agent_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
                logger.exception("PDF export error: %s", exc)
        with export_col_pptx:
            try:
                with st.spinner("Building PowerPoint deck…"):
                    pptx_bytes = _build_pptx_from_pipeline_result(pipeline_result, analyzed_path)
                st.download_button(
                    label="⬇️ Download Report as PowerPoint",
                    data=pptx_bytes,
                    file_name="insight_agent_report.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"PowerPoint generation failed: {exc}")
                logger.exception("PPTX export error: %s", exc)
    with tab_agent:
        workspace_log: list[dict] = st.session_state.get("workspace_log", [])
        if workspace_log:
            _render_workspace_log(workspace_log)
        else:
            st.info("No agent execution logs found for this session.")
    with tab_chat:
        if analyzed_path and os.path.exists(analyzed_path):
            render_data_chat(analyzed_path)
        else:
            st.warning("Source data file not found. Re-upload to enable the chat feature.")

def _render_running_indicator(state: CrewPipelineState) -> None:
    elapsed = round(time.monotonic() - state.submitted_at, 1)
    st.info(
        f"⏳ **CrewAI Agents Running** — Data Cleaner → ML Analyst → Model Engineer → Business Storyteller → Slides Formatter… "
        f"({elapsed}s elapsed)"
    )
    workspace_log: list[dict] = st.session_state.get("workspace_log", [])
    with st.expander("🧠 Agent Workspace — Live Thoughts", expanded=True):
        if workspace_log:
            _render_workspace_log(workspace_log)
        else:
            st.caption("Waiting for first agent output…")
    col_check, _ = st.columns(2)
    with col_check:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

def _render_pipeline_error(error_message: str) -> None:
    st.error("❌ CrewAI pipeline failed.")
    with st.expander("Error details", expanded=True):
        st.code(error_message, language="text")
    workspace_log: list[dict] = st.session_state.get("workspace_log", [])
    if workspace_log:
        with st.expander("🧠 Agent Workspace — Execution Log Before Failure", expanded=False):
            _render_workspace_log(workspace_log)
    st.caption("Fix the issue above, then click **Generate Business Report** to retry.")

def main() -> None:
    st.title("Insight-Agent: Autonomous Data Science Dashboard")
    render_auth_sidebar()
    if not st.session_state.get("user"):
        st.warning("Please log in to access the Data Ingestion Dashboard.")
        return
    if "pipeline_stage" not in st.session_state:
        st.session_state.pipeline_stage = "idle"
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = None
    save_path: str | None = render_dashboard()
    if save_path and save_path != st.session_state.get("explorer_file_path"):
        st.session_state.explorer_file_path = save_path
        st.session_state.pipeline_stage = "idle"
        st.session_state.pipeline_state = None
        for key in ("pipeline_result", "analyzed_file_path", "workspace_log",
                    "automl_target_col", "automl_task_type"):
            st.session_state.pop(key, None)
    stage: str = st.session_state.pipeline_stage
    if stage == "idle":
        if st.session_state.get("explorer_file_path"):
            file_path_for_config: str = st.session_state.explorer_file_path
            target_col, task_type = _render_automl_config(file_path_for_config)
            if st.button("🚀 Generate Business Report", use_container_width=True):
                crew_state: CrewPipelineState = _submit_crew_pipeline(
                    file_path_for_config, target_col, task_type
                )
                st.session_state.analyzed_file_path = file_path_for_config
                st.session_state.pipeline_state = crew_state
                st.session_state.pipeline_stage = "running"
                st.session_state.workspace_log = []
                st.session_state.automl_target_column = target_col
                st.session_state.pop("pipeline_result", None)
                st.rerun()
    elif stage == "running":
        current_state: CrewPipelineState | None = st.session_state.get("pipeline_state")
        if current_state is None:
            st.session_state.pipeline_stage = "idle"
            st.rerun()
            return
        _drain_step_queue(current_state)
        updated_state: CrewPipelineState = _poll_crew_state(current_state)
        st.session_state.pipeline_state = updated_state
        if updated_state.status == "complete":
            _drain_step_queue(updated_state)
            st.session_state.pipeline_result = updated_state.result
            st.session_state.pipeline_stage = "complete"
            st.rerun()
        elif updated_state.status == "error":
            _drain_step_queue(updated_state)
            st.session_state.pipeline_stage = "error"
            st.rerun()
        else:
            _render_running_indicator(updated_state)
            time.sleep(2)
            st.rerun()
    elif stage == "complete":
        pipeline_result: dict | None = st.session_state.get("pipeline_result")
        if pipeline_result:
            _render_final_report(pipeline_result)
        else:
            st.warning("Report result not found in session. Please re-run the pipeline.")
            st.session_state.pipeline_stage = "idle"
    elif stage == "error":
        error_state: CrewPipelineState | None = st.session_state.get("pipeline_state")
        error_msg = (
            error_state.error
            if error_state is not None
            else "Unknown error — check application logs."
        )
        _render_pipeline_error(error_msg)
        if st.button("🔄 Reset and Try Again", use_container_width=True):
            st.session_state.pipeline_stage = "idle"
            st.session_state.pipeline_state = None
            for key in ("pipeline_result", "workspace_log"):
                st.session_state.pop(key, None)
            st.rerun()

if __name__ == "__main__":
    main()