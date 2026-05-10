"""Evaluation comparison UI for Bali flood-risk models.

Run from the project root:
    uv run streamlit run app/eval_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from bali_flood_prevention.schema import LABEL_FIELDS, RISK_LEVELS

PROJECT_ROOT = Path(__file__).parent.parent
EVALS_DIR = PROJECT_ROOT / "evals"
DATA_DIR = PROJECT_ROOT / "data"
EVAL_FIELDS = list(LABEL_FIELDS)
RISK_FIELD = "flood_risk_level"


@st.cache_data
def load_eval_run(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    run_dir = EVALS_DIR / run_id
    results_path = run_dir / "results.json"
    meta_path = run_dir / "meta.json"
    if not results_path.exists():
        return None
    results = json.loads(results_path.read_text(encoding="utf-8"))
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {"eval_run_id": run_id, "model": "unknown", "dataset": "unknown"}
    )
    if not isinstance(results, list) or not isinstance(meta, dict):
        return None
    return meta, results


def list_runs() -> list[str]:
    if not EVALS_DIR.exists():
        return []
    return sorted((path.name for path in EVALS_DIR.iterdir() if path.is_dir()), reverse=True)


def run_label(run_id: str, meta: dict[str, Any]) -> str:
    model = str(meta.get("model", "?"))
    dataset = str(meta.get("dataset", "?"))
    short_model = model.split("/")[-1]
    return f"{run_id} | {short_model} | {dataset}"


def compute_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    field_accs: dict[str, float] = {}
    for field in EVAL_FIELDS:
        matches = [bool(row.get("field_matches", {}).get(field, False)) for row in results]
        field_accs[field] = sum(matches) / len(matches) if matches else 0.0
    all_matches = [
        bool(row.get("field_matches", {}).get(field, False))
        for row in results
        for field in EVAL_FIELDS
    ]
    valid_json = sum(bool(row.get("valid_json")) for row in results) / len(results)
    fields_present = sum(bool(row.get("fields_present")) for row in results) / len(results)
    avg_latency = sum(float(row.get("latency_s", 0.0)) for row in results) / len(results)
    overall = sum(all_matches) / len(all_matches) if all_matches else 0.0
    return {
        "valid_json": valid_json,
        "fields_present": fields_present,
        **field_accs,
        "overall": overall,
        "avg_latency_s": avg_latency,
    }


def confusion_matrix(results: list[dict[str, Any]]) -> pd.DataFrame:
    labels = list(RISK_LEVELS)
    counts = {(true, pred): 0 for true in labels for pred in labels}
    for row in results:
        truth = row.get("ground_truth") or {}
        pred = row.get("prediction") or {}
        true_value = str(truth.get(RISK_FIELD, ""))
        pred_value = str(pred.get(RISK_FIELD, ""))
        if true_value in labels and pred_value in labels:
            counts[(true_value, pred_value)] += 1
    df = pd.DataFrame(
        {pred: [counts[(true, pred)] for true in labels] for pred in labels},
        index=labels,
    )
    df.index.name = "true \\ predicted"
    return df


def risk_distribution(results: list[dict[str, Any]]) -> pd.DataFrame:
    labels = list(RISK_LEVELS)
    true_counts: dict[str, int] = {label: 0 for label in labels}
    pred_counts: dict[str, int] = {label: 0 for label in labels}
    for row in results:
        truth = row.get("ground_truth") or {}
        pred = row.get("prediction") or {}
        true_value = str(truth.get(RISK_FIELD, ""))
        pred_value = str(pred.get(RISK_FIELD, ""))
        if true_value in true_counts:
            true_counts[true_value] += 1
        if pred_value in pred_counts:
            pred_counts[pred_value] += 1
    return pd.DataFrame({"true": true_counts, "predicted": pred_counts}, index=labels)


def find_images(result: dict[str, Any]) -> tuple[Path | None, Path | None]:
    rgb_path = result.get("rgb_path")
    swir_path = result.get("swir_path")
    if isinstance(rgb_path, str) and isinstance(swir_path, str):
        rgb = PROJECT_ROOT / rgb_path
        swir = PROJECT_ROOT / swir_path
        if rgb.exists() and swir.exists():
            return rgb, swir

    sample_id = str(result.get("id", ""))
    if "/" not in sample_id:
        return None, None
    region, sample_key = sample_id.split("/", 1)
    for run_dir in sorted((path for path in DATA_DIR.iterdir() if path.is_dir()), reverse=True):
        sample_dir = run_dir / "test" / region / sample_key
        rgb = sample_dir / "rgb.png"
        swir = sample_dir / "swir.png"
        if rgb.exists() and swir.exists():
            return rgb, swir
    return None, None


def selected_loaded_runs(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    return [(run_id, *loaded[run_id]) for run_id in selected_run_ids]


def render_summary_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    columns: dict[str, dict[str, float | int]] = {}
    seen_models: dict[str, int] = {}
    for run_id, meta, results in selected_loaded_runs(selected_run_ids, loaded):
        summary = compute_summary(results)
        model = str(meta.get("model", "?"))
        model_label = model.split("/")[-1]
        seen_models[model_label] = seen_models.get(model_label, 0) + 1
        if seen_models[model_label] > 1:
            model_label = f"{model_label} ({run_id})"

        column: dict[str, float | int] = {"n": len(results)}
        column.update(summary)
        columns[model_label] = column

    df = pd.DataFrame(columns)
    percent_cols = ["valid_json", "fields_present", *EVAL_FIELDS, "overall"]
    df.loc[percent_cols] = (df.loc[percent_cols] * 100).round(1)
    if "avg_latency_s" in df.index:
        df.loc["avg_latency_s"] = df.loc["avg_latency_s"].round(2)
    df = df.rename(
        index={
            **{column: f"{column} %" for column in percent_cols},
            "avg_latency_s": "avg_latency_s",
        }
    )
    row_order = [
        "n",
        *[f"{column} %" for column in ["valid_json", "fields_present", *EVAL_FIELDS, "overall"]],
        "avg_latency_s",
    ]
    df = df.loc[[row for row in row_order if row in df.index]]
    df.index.name = "metric"
    st.subheader("Model comparison matrix")
    st.dataframe(df, use_container_width=True)


def render_risk_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    for run_id, meta, results in selected_loaded_runs(selected_run_ids, loaded):
        model = str(meta.get("model", "?")).split("/")[-1]
        st.markdown(f"### {run_id} | {model}")
        col_dist, col_cm = st.columns(2)
        with col_dist:
            st.markdown("**Flood risk distribution**")
            st.bar_chart(risk_distribution(results))
        with col_cm:
            st.markdown("**Confusion matrix**")
            st.dataframe(confusion_matrix(results), use_container_width=True)

        wrong_rows = []
        for row in results:
            truth = row.get("ground_truth") or {}
            pred = row.get("prediction") or {}
            if truth.get(RISK_FIELD) != pred.get(RISK_FIELD):
                wrong_rows.append(
                    {
                        "id": row.get("id"),
                        "region": row.get("region"),
                        "true": truth.get(RISK_FIELD),
                        "predicted": pred.get(RISK_FIELD, "missing"),
                        "valid_json": row.get("valid_json"),
                        "fields_present": row.get("fields_present"),
                    }
                )
        if wrong_rows:
            st.markdown(f"**Wrong or missing flood risk ({len(wrong_rows)} / {len(results)})**")
            st.dataframe(pd.DataFrame(wrong_rows), use_container_width=True, height=240)
        else:
            st.success("All flood_risk_level predictions match.")
        st.divider()


def build_sample_rows(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, meta, results in selected_loaded_runs(selected_run_ids, loaded):
        model = str(meta.get("model", "?")).split("/")[-1]
        for result in results:
            truth = result.get("ground_truth") or {}
            pred = result.get("prediction") or {}
            field_matches = result.get("field_matches") or {}
            mismatch_count = sum(
                1 for field in EVAL_FIELDS if not bool(field_matches.get(field, False))
            )
            rows.append(
                {
                    "run": run_id,
                    "model": model,
                    "id": result.get("id"),
                    "region": result.get("region"),
                    "timestamp": result.get("timestamp"),
                    "valid_json": bool(result.get("valid_json")),
                    "fields_present": bool(result.get("fields_present")),
                    "latency_s": round(float(result.get("latency_s", 0.0)), 2),
                    "gt_risk": truth.get(RISK_FIELD),
                    "pred_risk": pred.get(RISK_FIELD, "missing"),
                    "gt_quality_limited": truth.get("cloud_shadow_or_image_quality_limited"),
                    "gt_confidence": truth.get("confidence"),
                    "mismatches": mismatch_count,
                }
            )
    return pd.DataFrame(rows)


def render_sample_tab(
    selected_run_ids: list[str],
    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    if not selected_run_ids:
        st.info("Select one or more eval runs in the sidebar.")
        return

    df = build_sample_rows(selected_run_ids, loaded)
    if df.empty:
        st.info("No sample rows found.")
        return

    col_region, col_risk, col_flags, col_field = st.columns(4)
    with col_region:
        regions = ["all", *sorted(str(region) for region in df["region"].dropna().unique())]
        region_filter = st.selectbox("Region", regions)
    with col_risk:
        risk_filter = st.selectbox("True flood risk", ["all", *list(RISK_LEVELS)])
    with col_flags:
        wrong_only = st.checkbox("Wrong risk only")
        quality_only = st.checkbox("Image-quality limited")
        low_conf_only = st.checkbox("Low confidence")
    with col_field:
        field_filter = st.selectbox("Field mismatch", ["any", *EVAL_FIELDS])

    filtered = df.copy()
    if region_filter != "all":
        filtered = filtered[filtered["region"] == region_filter]
    if risk_filter != "all":
        filtered = filtered[filtered["gt_risk"] == risk_filter]
    if wrong_only:
        filtered = filtered[filtered["gt_risk"] != filtered["pred_risk"]]
    if quality_only:
        filtered = filtered[filtered["gt_quality_limited"] == True]  # noqa: E712
    if low_conf_only:
        filtered = filtered[filtered["gt_confidence"] == "low"]
    if field_filter != "any":
        mismatched_ids = set()
        for run_id in selected_run_ids:
            _, results = loaded[run_id]
            for result in results:
                matches = result.get("field_matches") or {}
                if not bool(matches.get(field_filter, False)):
                    mismatched_ids.add((run_id, result.get("id")))
        filtered = filtered[
            filtered.apply(lambda row: (row["run"], row["id"]) in mismatched_ids, axis=1)
        ]

    st.dataframe(filtered, use_container_width=True, height=360)
    st.subheader("Sample detail")

    sample_ids = sorted(str(value) for value in filtered["id"].dropna().unique())
    if not sample_ids:
        st.info("No samples match the current filters.")
        return
    selected_sample = st.selectbox("Sample", sample_ids)

    first_result = None
    for run_id in selected_run_ids:
        _, results = loaded[run_id]
        for result in results:
            if result.get("id") == selected_sample:
                first_result = result
                break
        if first_result:
            break

    image_col, compare_col = st.columns([2, 1])
    with image_col:
        if first_result:
            rgb_path, swir_path = find_images(first_result)
            if rgb_path and swir_path:
                rgb_col, swir_col = st.columns(2)
                with rgb_col:
                    st.image(str(rgb_path), caption="RGB", use_container_width=True)
                with swir_col:
                    st.image(str(swir_path), caption="SWIR", use_container_width=True)
            else:
                st.warning("Images not found for this sample.")

    with compare_col:
        for run_id in selected_run_ids:
            meta, results = loaded[run_id]
            sample_results = [row for row in results if row.get("id") == selected_sample]
            if not sample_results:
                continue
            result = sample_results[0]
            truth = result.get("ground_truth") or {}
            pred = result.get("prediction") or {}
            model = str(meta.get("model", "?")).split("/")[-1]
            st.markdown(f"**{run_id} | {model}**")
            comparison = []
            for field in EVAL_FIELDS:
                comparison.append(
                    {
                        "field": field,
                        "ground truth": str(truth.get(field, "?")),
                        "predicted": str(pred.get(field, "missing")),
                        "match": bool(result.get("field_matches", {}).get(field, False)),
                    }
                )
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, height=360)
            error = result.get("error")
            if error:
                st.caption(f"error: {error}")
            st.caption(f"latency: {float(result.get('latency_s', 0.0)):.2f}s")
            st.divider()


def main() -> None:
    st.set_page_config(page_title="Bali Flood Eval Comparison", layout="wide")
    st.title("Bali flood model evaluation comparison")

    all_runs = list_runs()
    if not all_runs:
        st.error(f"No eval runs found in {EVALS_DIR}. Run scripts/evaluate.py first.")
        return

    loaded: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    report_only: list[str] = []
    for run_id in all_runs:
        data = load_eval_run(run_id)
        if data is None:
            report_only.append(run_id)
        else:
            loaded[run_id] = data

    with st.sidebar:
        st.header("Eval runs")
        options = {run_label(run_id, loaded[run_id][0]): run_id for run_id in loaded}
        selected_labels = st.multiselect(
            "Runs",
            list(options.keys()),
            default=list(options.keys())[:3],
        )
        selected_run_ids = [options[label] for label in selected_labels]
        if report_only:
            with st.expander(f"Report-only runs ({len(report_only)})"):
                for run_id in report_only:
                    st.write(run_id)

    tab_summary, tab_risk, tab_samples = st.tabs(
        ["Summary", "Flood risk", "Sample explorer"]
    )
    with tab_summary:
        render_summary_tab(selected_run_ids, loaded)
    with tab_risk:
        render_risk_tab(selected_run_ids, loaded)
    with tab_samples:
        render_sample_tab(selected_run_ids, loaded)


if __name__ == "__main__":
    main()
