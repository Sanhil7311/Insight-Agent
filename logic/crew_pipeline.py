from __future__ import annotations

import json
import logging
import queue
import re
import time
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task

from logic.agent_factory import (
    build_analytics_bundle,
    get_local_llm,
    _get_data_context,
    _load_raw_dataframe,
)
from logic.analytics_engine import (
    load_dataset,
    run_cleaning,
    compute_analytics,
    recommend_charts,
    run_automl,
)

logger = logging.getLogger(__name__)

_CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "scatter", "histogram", "box", "heatmap", "line", "pie", "area"}
)


def _build_data_cleaner_agent(step_q: queue.Queue) -> Agent:
    def _step_cb(step_output: Any) -> None:
        try:
            text = (
                step_output.text
                if hasattr(step_output, "text")
                else str(step_output)
            )
            step_q.put_nowait({
                "agent": "Data Cleaner",
                "type": "step",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass

    return Agent(
        role="Data Cleaner",
        goal=(
            "Inspect the raw dataset profile and produce a precise, structured Markdown "
            "cleaning strategy. Identify every data quality issue — missing values, "
            "duplicates, outliers, incorrect dtypes — and prescribe a deterministic "
            "remediation step for each issue."
        ),
        backstory=(
            "You are a meticulous data engineer with ten years of experience preparing "
            "messy real-world datasets for downstream analytics. You never guess; you "
            "reason from evidence. Your output is always a numbered Markdown list of "
            "concrete actions another engineer could execute without ambiguity."
        ),
        llm=get_local_llm(),
        verbose=False,
        allow_delegation=False,
        memory=False,
        max_iter=3,
        max_retry_limit=2,
        step_callback=_step_cb,
    )


def _build_ml_analyst_agent(step_q: queue.Queue) -> Agent:
    def _step_cb(step_output: Any) -> None:
        try:
            text = (
                step_output.text
                if hasattr(step_output, "text")
                else str(step_output)
            )
            step_q.put_nowait({
                "agent": "ML Analyst",
                "type": "step",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass

    return Agent(
        role="Machine Learning Analyst",
        goal=(
            "Consume the cleaning strategy and the full pre-computed analytics bundle "
            "to produce a structured analytical summary that includes: descriptive "
            "statistics interpretation, correlation findings, categorical distributions, "
            "and a JSON chart-recommendation block."
        ),
        backstory=(
            "You are a quantitative analyst who bridges the gap between raw statistics "
            "and actionable insight. You communicate findings in plain language backed "
            "by exact numeric evidence. You always close your response with a single "
            "CHARTS JSON block so visualisation engineers can render charts without "
            "further interpretation."
        ),
        llm=get_local_llm(),
        verbose=False,
        allow_delegation=False,
        memory=False,
        max_iter=3,
        max_retry_limit=2,
        step_callback=_step_cb,
    )


def _build_model_engineer_agent(step_q: queue.Queue) -> Agent:
    def _step_cb(step_output: Any) -> None:
        try:
            text = (
                step_output.text
                if hasattr(step_output, "text")
                else str(step_output)
            )
            step_q.put_nowait({
                "agent": "Model Engineer",
                "type": "step",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass

    return Agent(
        role="Predictive Model Engineer",
        goal=(
            "Interpret pre-computed AutoML metrics from a trained RandomForest model. "
            "Explain the model's performance, what the top features reveal about the "
            "dataset, and what predictions mean in a business context. "
            "Produce a concise Markdown section titled '## Predictive Model Insights' "
            "that can be inserted directly into an executive report."
        ),
        backstory=(
            "You are an applied ML engineer with expertise translating model outputs "
            "into plain-language business narratives. You understand that accuracy "
            "numbers alone mean nothing without context — you always relate metrics "
            "back to business impact, flag limitations of the baseline model, and "
            "recommend next steps for productionisation."
        ),
        llm=get_local_llm(),
        verbose=False,
        allow_delegation=False,
        memory=False,
        max_iter=3,
        max_retry_limit=2,
        step_callback=_step_cb,
    )


def _build_business_storyteller_agent(step_q: queue.Queue) -> Agent:
    def _step_cb(step_output: Any) -> None:
        try:
            text = (
                step_output.text
                if hasattr(step_output, "text")
                else str(step_output)
            )
            step_q.put_nowait({
                "agent": "Business Storyteller",
                "type": "step",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass

    return Agent(
        role="Business Storyteller",
        goal=(
            "Synthesise the ML analyst's structured summary and the Model Engineer's "
            "predictive insights into a polished, executive-ready Markdown business "
            "report with six clearly delineated sections: Executive Summary, "
            "Key Statistical Findings, Notable Correlations and Patterns, Data Quality "
            "Observations, Predictive Forecast, and Actionable Business Recommendations."
        ),
        backstory=(
            "You are a senior management consultant who translates analytical output into "
            "board-level narratives. Every sentence you write is precise, evidence-backed, "
            "and oriented toward business decisions. You never fabricate statistics; you "
            "restate only what the analyst and model engineer provided."
        ),
        llm=get_local_llm(),
        verbose=False,
        allow_delegation=False,
        memory=False,
        max_iter=3,
        max_retry_limit=2,
        step_callback=_step_cb,
    )


def _build_slides_formatter_agent(step_q: queue.Queue) -> Agent:
    def _step_cb(step_output: Any) -> None:
        try:
            text = (
                step_output.text
                if hasattr(step_output, "text")
                else str(step_output)
            )
            step_q.put_nowait({
                "agent": "Slides Formatter",
                "type": "step",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass

    return Agent(
        role="Presentation Formatter",
        goal=(
            "Convert a Markdown business report into a structured JSON array of "
            "presentation slides, one slide per major report section. Each slide must "
            "have a 'title' and a 'bullets' list of concise, self-contained bullet strings."
        ),
        backstory=(
            "You are a presentation design specialist who distils long-form reports into "
            "executive slide decks. You extract only the most impactful points, write each "
            "bullet as a single complete clause, and always output valid JSON. You never "
            "add slides beyond what the report contains."
        ),
        llm=get_local_llm(),
        verbose=False,
        allow_delegation=False,
        memory=False,
        max_iter=2,
        max_retry_limit=2,
        step_callback=_step_cb,
    )


def _make_task_callback(agent_name: str, step_q: queue.Queue):
    def _cb(task_output: Any) -> None:
        try:
            text = (
                task_output.raw
                if hasattr(task_output, "raw")
                else str(task_output)
            )
            step_q.put_nowait({
                "agent": agent_name,
                "type": "task_complete",
                "content": text.strip(),
                "ts": time.monotonic(),
            })
        except Exception:
            pass
    return _cb


def _build_task_clean(
    agent: Agent,
    data_context: str,
    cleaning_report: dict[str, Any],
    step_q: queue.Queue,
) -> Task:
    cleaning_json = json.dumps(cleaning_report, separators=(",", ":"), ensure_ascii=False)

    description = (
        "You have been provided with a structured profile of a raw dataset and the "
        "deterministic cleaning actions that have already been applied by the pipeline.\n\n"
        "## Dataset Profile\n"
        f"{data_context}\n\n"
        "## Applied Cleaning Actions (JSON)\n"
        f"{cleaning_json}\n\n"
        "## Your Task\n"
        "Produce a numbered Markdown cleaning strategy that:\n"
        "1. Acknowledges every action already taken (with the original values where given).\n"
        "2. Flags any residual concerns not covered by the automated steps.\n"
        "3. Recommends additional manual steps, if any, in priority order.\n"
        "4. Closes with a one-sentence overall data-readiness verdict.\n\n"
        "Do not invent column names or statistics not present in the profile above."
    )

    expected_output = (
        "A numbered Markdown list of cleaning observations and recommendations, "
        "ending with a one-sentence data-readiness verdict."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        callback=_make_task_callback("Data Cleaner", step_q),
    )


def _build_task_analyze(
    agent: Agent,
    analytics: dict[str, Any],
    file_name: str,
    step_q: queue.Queue,
) -> Task:
    analytics_json = json.dumps(analytics, separators=(",", ":"), ensure_ascii=False)
    shape = analytics.get("shape", ["?", "?"])
    rows, cols = (shape[0], shape[1]) if len(shape) == 2 else ("?", "?")

    column_names: list[str] = list(analytics.get("dtype_map", {}).keys())
    columns_hint = ", ".join(column_names) if column_names else "unknown"

    description = (
        f"Dataset: {file_name}  |  Shape: {rows} rows × {cols} columns\n"
        f"Available columns: {columns_hint}\n\n"
        "## Pre-Computed Analytics Bundle (JSON)\n"
        f"{analytics_json}\n\n"
        "## Cleaning Strategy\n"
        "{task_clean_output}\n\n"
        "## Your Task\n"
        "Produce a structured analytical summary with the following sections:\n\n"
        "### 1. Descriptive Statistics\n"
        "Interpret mean, std, skew, and range for every numeric column present in the analytics.\n\n"
        "### 2. Correlation Analysis\n"
        "Describe every correlated pair (|r| > 0.30) listed in the analytics. "
        "State the direction and practical implication of each.\n\n"
        "### 3. Categorical Distributions\n"
        "Summarise the top-value distribution for each categorical column.\n\n"
        "### 4. Anomaly Notes\n"
        "Highlight any high-skew columns, constant columns, or columns with residual nulls.\n\n"
        "After the four sections, append on its own line:\n"
        "CHARTS: [{\"chart_type\":\"<type>\",\"x_axis\":\"<col_or_null>\","
        "\"y_axis\":\"<col_or_null>\",\"title\":\"<title>\"}]\n\n"
        "STRICT RULES FOR THE CHARTS BLOCK:\n"
        "- Output exactly one CHARTS: line containing a valid JSON array.\n"
        "- Include 3 to 4 chart objects only.\n"
        f"- Use ONLY column names from this list: {columns_hint}.\n"
        "- Valid chart_type values: bar, scatter, histogram, box, heatmap, line, pie.\n"
        "- Set x_axis or y_axis to null (not the string 'null') when not applicable.\n"
        "- Do not add any text after the CHARTS line.\n"
        "- Do not fabricate or rename columns.\n"
        "- Do not use statistics not present in the analytics bundle above."
    )

    expected_output = (
        "A structured Markdown analytical summary with four numbered sections "
        "(Descriptive Statistics, Correlation Analysis, Categorical Distributions, "
        "Anomaly Notes), followed by a single CHARTS: JSON line."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=[],
        callback=_make_task_callback("ML Analyst", step_q),
    )


def _build_task_model_engineer(
    agent: Agent,
    predictive_metrics: dict[str, Any],
    step_q: queue.Queue,
) -> Task:
    metrics_json = json.dumps(predictive_metrics, separators=(",", ":"), ensure_ascii=False)

    target = predictive_metrics.get("target_column", "unknown")
    task_type = predictive_metrics.get("task_type", "unknown")
    model_name = predictive_metrics.get("model", "RandomForest")
    metrics = predictive_metrics.get("metrics", {})
    fi = predictive_metrics.get("feature_importances", {})
    error = predictive_metrics.get("error")

    metrics_readable = ", ".join(
        f"{k.upper().replace('_', ' ')} = {v}" for k, v in metrics.items()
    ) if metrics else "No metrics available."

    fi_readable = "\n".join(
        f"  - {feat}: {imp}" for feat, imp in fi.items()
    ) if fi else "  No feature importances available."

    if error:
        error_section = f"\n## AutoML Error\nThe model training encountered an error: {error}\n"
    else:
        error_section = ""

    description = (
        f"A baseline {model_name} model has been trained to predict **{target}** "
        f"as a **{task_type}** task.\n\n"
        f"## Raw AutoML Output (JSON)\n{metrics_json}\n\n"
        f"## Model Performance Metrics\n{metrics_readable}\n"
        f"{error_section}\n"
        f"## Top Feature Importances\n{fi_readable}\n\n"
        "## Your Task\n"
        "Write a concise Markdown section titled exactly:\n\n"
        "## Predictive Model Insights\n\n"
        "This section must cover:\n"
        "1. **Model Summary** — state the algorithm, target variable, task type, and train/test split sizes.\n"
        "2. **Performance Assessment** — interpret each metric in plain language. "
        "For classification: is the accuracy/F1 strong, moderate, or weak? "
        "For regression: is the R² meaningful and the RMSE acceptable relative to the target's range?\n"
        "3. **Key Predictors** — explain what the top 3–5 features reveal about the drivers of the target variable.\n"
        "4. **Limitations and Next Steps** — note that this is a baseline model and recommend "
        "at least two concrete improvements (e.g. hyperparameter tuning, feature engineering, "
        "cross-validation, or trying gradient boosting).\n\n"
        "RULES:\n"
        "- Use only the metrics and feature names provided above.\n"
        "- Do not invent numbers or column names.\n"
        "- If an error occurred, explain it plainly and still provide a useful 'Next Steps' subsection.\n"
        "- Do not include a CHARTS block."
    )

    expected_output = (
        "A Markdown section titled '## Predictive Model Insights' covering model summary, "
        "performance assessment, key predictors, and limitations with next steps."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=[],
        callback=_make_task_callback("Model Engineer", step_q),
    )


def _build_task_story(
    agent: Agent,
    file_name: str,
    has_predictive_metrics: bool,
    step_q: queue.Queue,
) -> Task:
    if has_predictive_metrics:
        predictive_section_instruction = (
            "## Predictive Forecast\n"
            "Incorporate the Model Engineer's '## Predictive Model Insights' section here. "
            "Translate the model findings into a forward-looking business narrative: "
            "what can the organisation predict, with what confidence, and what should they do next?\n\n"
        )
        context_note = (
            "## Analytical Summary (from ML Analyst)\n"
            "{task_analyze_output}\n\n"
            "## Predictive Model Insights (from Model Engineer)\n"
            "{task_model_engineer_output}\n\n"
        )
        sections = (
            "Executive Summary, Key Statistical Findings, Notable Correlations and Patterns, "
            "Data Quality Observations, Predictive Forecast, and Actionable Business Recommendations"
        )
        n_sections = "six"
    else:
        predictive_section_instruction = ""
        context_note = (
            "## Analytical Summary (from ML Analyst)\n"
            "{task_analyze_output}\n\n"
        )
        sections = (
            "Executive Summary, Key Statistical Findings, Notable Correlations and Patterns, "
            "Data Quality Observations, and Actionable Business Recommendations"
        )
        n_sections = "five"

    description = (
        f"You are writing the final business report for the dataset: {file_name}.\n\n"
        f"{context_note}"
        f"## Your Task\n"
        f"Synthesise the inputs above into a polished executive Markdown report "
        f"with the following {n_sections} sections (use ## headings):\n\n"
        "## Executive Summary\n"
        "2–3 sentences describing the dataset and its business context at a glance.\n\n"
        "## Key Statistical Findings\n"
        "Bullet list citing specific numeric values from the analyst's summary.\n\n"
        "## Notable Correlations and Patterns\n"
        "Describe each correlated pair and its business implication in plain language.\n\n"
        "## Data Quality Observations\n"
        "Summarise the cleaning actions taken and any residual concerns.\n\n"
        f"{predictive_section_instruction}"
        "## Actionable Business Recommendations\n"
        "3–5 concrete, prioritised recommendations a decision-maker can act on.\n\n"
        "RULES:\n"
        "- Use only facts from the analytical summary and model engineer output. Do not invent statistics.\n"
        "- Do not include a CHARTS block in your output.\n"
        f"- Do not truncate or omit any of the {n_sections} sections.\n"
        "- Write in clear, professional British English.\n"
        "- Keep bullet points concise (one clause per bullet)."
    )

    expected_output = (
        f"A complete {n_sections}-section Markdown business report covering {sections}."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=[],
        callback=_make_task_callback("Business Storyteller", step_q),
    )


def _build_task_slides(
    agent: Agent,
    has_predictive_metrics: bool,
    step_q: queue.Queue,
) -> Task:
    if has_predictive_metrics:
        section_list = (
            "Executive Summary, Key Statistical Findings, Notable Correlations and Patterns, "
            "Data Quality Observations, Predictive Forecast, Actionable Business Recommendations"
        )
    else:
        section_list = (
            "Executive Summary, Key Statistical Findings, Notable Correlations and Patterns, "
            "Data Quality Observations, Actionable Business Recommendations"
        )

    description = (
        "You will be given the final business report in Markdown (from the Business Storyteller).\n\n"
        "## Business Report\n"
        "{task_story_output}\n\n"
        "## Your Task\n"
        "Convert this report into a JSON array of slide objects. "
        "Create exactly one slide per major section of the report.\n\n"
        f"The expected sections are: {section_list}.\n\n"
        "Each slide object must have exactly two keys:\n"
        "  - \"title\": a short string (the section heading, stripped of ## markers)\n"
        "  - \"bullets\": a JSON array of strings, each being one concise bullet point "
        "from that section (maximum 6 bullets per slide, each under 20 words)\n\n"
        "OUTPUT RULES:\n"
        "- Output ONLY a valid JSON array. No preamble, no explanation, no markdown fences.\n"
        "- Do not include chart specifications.\n"
        "- Do not fabricate content not present in the report.\n"
        "- Bullets must be plain text strings, no markdown bold/italic markers.\n"
        "- The array must start with [ and end with ].\n\n"
        "Example of correct output format:\n"
        '[{"title":"Executive Summary","bullets":["Dataset covers 5,000 sales records across 12 months.","Primary objective is to identify revenue drivers."]}]'
    )

    expected_output = (
        "A valid JSON array of slide objects, each with 'title' (string) and "
        "'bullets' (array of strings). No other text."
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=[],
        callback=_make_task_callback("Slides Formatter", step_q),
    )


def _extract_charts_from_analyst_output(analyst_output: str) -> list[dict[str, Any]]:
    for line in reversed(analyst_output.splitlines()):
        stripped = line.strip()
        if not stripped.upper().startswith("CHARTS:"):
            continue
        fragment = stripped[len("CHARTS:"):].strip()
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse CHARTS line from analyst output: %s", exc)
            return []
        if not isinstance(parsed, list):
            return []
        validated: list[dict[str, Any]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            chart_type = str(entry.get("chart_type", "")).lower().strip()
            if chart_type not in _CHART_TYPES:
                logger.debug("Skipping unknown chart_type %r", chart_type)
                continue
            validated.append(
                {
                    "chart_type": chart_type,
                    "x_axis": entry.get("x_axis") or None,
                    "y_axis": entry.get("y_axis") or None,
                    "title": str(entry.get("title", "Chart")),
                }
            )
        return validated
    logger.debug("No CHARTS line found in analyst output.")
    return []


def _strip_charts_line(text: str) -> str:
    lines = text.splitlines()
    idx: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().upper().startswith("CHARTS:"):
            idx = i
            break
    if idx is None:
        return text
    trimmed = lines[:idx]
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed)


def _extract_slide_json(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                validated: list[dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "Slide"))
                    bullets_raw = item.get("bullets", [])
                    if isinstance(bullets_raw, list):
                        bullets = [str(b) for b in bullets_raw if b]
                    elif isinstance(bullets_raw, str):
                        bullets = [b.strip() for b in bullets_raw.splitlines() if b.strip()]
                    else:
                        bullets = []
                    validated.append({"title": title, "bullets": bullets})
                return validated
        except json.JSONDecodeError as exc:
            logger.warning("_extract_slide_json JSON parse failed: %s", exc)

    logger.warning("Could not extract slide JSON from slides formatter output; returning empty list.")
    return []


def run_local_crew(
    file_path: str,
    step_q: queue.Queue | None = None,
    target_column: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    if not file_path or not isinstance(file_path, str):
        raise ValueError(f"file_path must be a non-empty string; got {file_path!r}")

    if step_q is None:
        step_q = queue.Queue()

    t_start = time.monotonic()
    file_name = Path(file_path).name

    logger.info(
        "run_local_crew started | file=%r | target=%r | task_type=%r",
        file_path, target_column, task_type,
    )

    def _push(agent: str, msg: str, kind: str = "info") -> None:
        step_q.put_nowait({"agent": agent, "type": kind, "content": msg, "ts": time.monotonic()})

    _push("Pipeline", f"Loading dataset: {file_name}", "info")
    t0 = time.monotonic()
    raw_df = load_dataset(file_path)
    _push(
        "Pipeline",
        f"Dataset loaded — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns ({time.monotonic()-t0:.2f}s)",
        "info",
    )

    t1 = time.monotonic()
    _push("Pipeline", "Running deterministic cleaning pipeline…", "info")
    cleaned_df, cleaning_report = run_cleaning(raw_df)
    actions = cleaning_report.get("actions", [])
    _push(
        "Pipeline",
        f"Cleaning complete — {len(actions)} action(s) applied ({time.monotonic()-t1:.2f}s)",
        "info",
    )
    for action in actions:
        _push("Data Cleaner", action, "info")

    t2 = time.monotonic()
    _push("Pipeline", "Computing analytics bundle…", "info")
    analytics = compute_analytics(cleaned_df)
    _push("Pipeline", f"Analytics bundle ready ({time.monotonic()-t2:.2f}s)", "info")

    t3 = time.monotonic()
    rule_chart_recs = recommend_charts(cleaned_df)
    _push(
        "Pipeline",
        f"Rule-based chart recommendations: {len(rule_chart_recs)} chart(s) ({time.monotonic()-t3:.2f}s)",
        "info",
    )

    predictive_metrics: dict[str, Any] = {"skipped": True}
    has_predictive = False

    if target_column and task_type:
        _push(
            "Pipeline",
            f"Running AutoML — training RandomForest to predict '{target_column}' ({task_type})…",
            "info",
        )
        t_automl = time.monotonic()
        predictive_metrics = run_automl(cleaned_df, target_column, task_type)
        elapsed_automl = time.monotonic() - t_automl

        if predictive_metrics.get("error"):
            _push(
                "Model Engineer",
                f"⚠️ AutoML encountered an error: {predictive_metrics['error']}",
                "info",
            )
        else:
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in predictive_metrics.get("metrics", {}).items()
            )
            _push(
                "Model Engineer",
                f"AutoML complete in {elapsed_automl:.2f}s — {metrics_str}",
                "info",
            )
        has_predictive = True
    else:
        _push("Pipeline", "AutoML skipped — no target column selected.", "info")

    data_context = _get_data_context(file_path)

    _push("Pipeline", "Initialising CrewAI agents…", "info")

    data_cleaner_agent = _build_data_cleaner_agent(step_q)
    ml_analyst_agent = _build_ml_analyst_agent(step_q)
    business_storyteller_agent = _build_business_storyteller_agent(step_q)
    slides_formatter_agent = _build_slides_formatter_agent(step_q)

    task_clean = _build_task_clean(data_cleaner_agent, data_context, cleaning_report, step_q)
    task_analyze = _build_task_analyze(ml_analyst_agent, analytics, file_name, step_q)
    task_analyze.context = [task_clean]

    agents = [data_cleaner_agent, ml_analyst_agent]
    tasks = [task_clean, task_analyze]

    if has_predictive:
        model_engineer_agent = _build_model_engineer_agent(step_q)
        task_model = _build_task_model_engineer(model_engineer_agent, predictive_metrics, step_q)
        task_model.context = [task_analyze]
        agents.append(model_engineer_agent)
        tasks.append(task_model)
        task_story = _build_task_story(
            business_storyteller_agent, file_name, has_predictive_metrics=True, step_q=step_q
        )
        task_story.context = [task_analyze, task_model]
    else:
        task_story = _build_task_story(
            business_storyteller_agent, file_name, has_predictive_metrics=False, step_q=step_q
        )
        task_story.context = [task_analyze]

    task_slides = _build_task_slides(slides_formatter_agent, has_predictive, step_q)
    task_slides.context = [task_story]

    agents.append(business_storyteller_agent)
    tasks.append(task_story)
    agents.append(slides_formatter_agent)
    tasks.append(task_slides)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    _push("Pipeline", "🚀 CrewAI kickoff started — agents are now running sequentially.", "info")
    logger.info("CrewAI kickoff started | file=%r", file_path)
    t_crew = time.monotonic()
    crew_result = crew.kickoff()

    _push(
        "Pipeline",
        f"✅ CrewAI kickoff finished ({time.monotonic()-t_crew:.1f}s) — parsing results…",
        "info",
    )
    logger.info("CrewAI kickoff completed | t=%.1fs", time.monotonic() - t_crew)

    if hasattr(crew_result, "tasks_output") and crew_result.tasks_output:
        tasks_output_list = crew_result.tasks_output
        story_task_index = 3 if has_predictive else 2
        slides_task_index = story_task_index + 1

        story_raw: str = ""
        if len(tasks_output_list) > story_task_index:
            to = tasks_output_list[story_task_index]
            story_raw = to.raw if hasattr(to, "raw") else str(to)

        slides_raw: str = ""
        if len(tasks_output_list) > slides_task_index:
            to = tasks_output_list[slides_task_index]
            slides_raw = to.raw if hasattr(to, "raw") else str(to)

        final_report_raw: str = story_raw
    else:
        if hasattr(crew_result, "raw"):
            final_report_raw = crew_result.raw or ""
        elif isinstance(crew_result, str):
            final_report_raw = crew_result
        else:
            final_report_raw = str(crew_result) if crew_result is not None else ""
        slides_raw = ""

    if not final_report_raw.strip():
        raise RuntimeError(
            "CrewAI returned an empty final output. "
            "Check that the Ollama model is running and accessible."
        )

    analyst_output: str = ""
    if hasattr(crew_result, "tasks_output") and crew_result.tasks_output:
        tasks_output_list = crew_result.tasks_output
        if len(tasks_output_list) >= 2:
            task_out = tasks_output_list[1]
            analyst_output = (
                task_out.raw if hasattr(task_out, "raw") else str(task_out)
            )

    llm_chart_recs = _extract_charts_from_analyst_output(analyst_output)
    chart_recommendations: list[dict[str, Any]] = (
        llm_chart_recs if llm_chart_recs else rule_chart_recs
    )

    report_markdown = _strip_charts_line(final_report_raw)
    slide_json = _extract_slide_json(slides_raw)

    if not slide_json:
        _push("Pipeline", "⚠️ Slide JSON extraction failed — PPTX export will use fallback.", "info")
        logger.warning("slide_json is empty after extraction; PPTX will use report-section fallback.")

    elapsed = round(time.monotonic() - t_start, 2)
    _push("Pipeline", f"🎉 Pipeline complete in {elapsed:.1f}s — returning report to UI.", "info")
    logger.info("run_local_crew finished | elapsed=%.1fs", elapsed)

    return {
        "report": report_markdown,
        "chart_recommendations": chart_recommendations,
        "cleaning_report": cleaning_report,
        "analytics": analytics,
        "predictive_metrics": predictive_metrics,
        "file_path": file_path,
        "elapsed_seconds": elapsed,
        "slide_json": slide_json,
    }