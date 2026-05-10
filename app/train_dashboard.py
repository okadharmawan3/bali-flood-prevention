"""Training progress dashboard for Bali flood fine-tuning runs.

Run from the project root:
    uv run streamlit run app/train_dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATTERNS = (
    re.compile(r"checkpoint-(\d+)$", re.IGNORECASE),
    re.compile(r"global_step(\d+)", re.IGNORECASE),
    re.compile(r"-e\d+s(\d+)-", re.IGNORECASE),
)
HEAVY_SUFFIXES = {
    ".bin",
    ".gguf",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
WEIGHT_NAMES = {
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
}
CORE_HISTORY_COLUMNS = [
    "step",
    "epoch",
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def extract_step(name: str) -> int | None:
    for pattern in CHECKPOINT_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def is_checkpoint_dir(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("checkpoint-")
        or name.startswith("final-global_step")
        or extract_step(name) is not None
    )


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def human_size(size: int | float | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    try:
        iterator = path.rglob("*") if path.is_dir() else iter([path])
        for item in iterator:
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def has_model_weights(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        iterator = path.rglob("*") if path.is_dir() else iter([path])
        for item in iterator:
            if not item.is_file():
                continue
            if item.name in WEIGHT_NAMES or item.suffix.lower() in {".safetensors", ".gguf"}:
                return True
    except OSError:
        return False
    return False


def parse_history(trainer_state: dict[str, Any] | None) -> pd.DataFrame:
    if not trainer_state:
        return pd.DataFrame()
    rows = trainer_state.get("log_history", [])
    if not isinstance(rows, list):
        return pd.DataFrame()

    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean_rows.append(
            {
                key: value
                for key, value in row.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
        )

    if not clean_rows:
        return pd.DataFrame()

    df = pd.DataFrame(clean_rows)
    if "step" not in df.columns:
        df["step"] = range(1, len(df) + 1)

    numeric_columns = set(CORE_HISTORY_COLUMNS)
    numeric_columns.update(column for column in df.columns if column.startswith("lr/"))
    for column in numeric_columns.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("step", kind="stable").reset_index(drop=True)


def trainer_state_score(path: Path, state: dict[str, Any]) -> tuple[int, int, float]:
    step = int(state.get("global_step") or 0)
    history = state.get("log_history", [])
    history_len = len(history) if isinstance(history, list) else 0
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return step, history_len, modified


def find_primary_trainer_state(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    states: list[tuple[Path, dict[str, Any]]] = []
    try:
        paths = list(run_dir.rglob("trainer_state.json"))
    except OSError:
        paths = []

    for path in paths:
        state = read_json(path)
        if state is not None:
            states.append((path, state))

    if not states:
        return None, None
    return max(states, key=lambda item: trainer_state_score(item[0], item[1]))


def checkpoint_record(path: Path, run_dir: Path) -> dict[str, Any]:
    step = extract_step(path.name)
    modified = None
    size_bytes = dir_size_bytes(path)
    try:
        modified = pd.to_datetime(path.stat().st_mtime, unit="s")
    except OSError:
        pass

    return {
        "name": path.name,
        "step": step,
        "path": str(path),
        "relative_path": str(path.relative_to(run_dir)) if path != run_dir else ".",
        "modified": modified,
        "size_bytes": size_bytes,
        "size": human_size(size_bytes),
        "has_model_weights": has_model_weights(path),
        "is_final": path.name.startswith("final-global_step"),
    }


def find_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dirs: dict[Path, None] = {}

    if is_checkpoint_dir(run_dir) or (run_dir / "trainer_state.json").exists():
        checkpoint_dirs[run_dir] = None

    try:
        for path in run_dir.rglob("*"):
            if path.is_dir() and is_checkpoint_dir(path):
                checkpoint_dirs[path] = None
    except OSError:
        pass

    records = [checkpoint_record(path, run_dir) for path in checkpoint_dirs]
    return sorted(
        records,
        key=lambda row: (
            row.get("step") is None,
            row.get("step") or -1,
            str(row.get("relative_path", "")),
        ),
    )


def run_dirs_from_outputs(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []

    candidates: set[Path] = set()

    for pattern in ("modal-checkpoints/**/trainer_state.json", "**/train_meta.json"):
        for path in outputs_dir.glob(pattern):
            candidates.add(path.parent)

    try:
        for path in outputs_dir.rglob("*"):
            if path.is_dir() and is_checkpoint_dir(path):
                parent = path.parent
                if (parent / "train_meta.json").exists():
                    candidates.add(parent)
                else:
                    sibling_checkpoints = [
                        child
                        for child in parent.iterdir()
                        if child.is_dir() and is_checkpoint_dir(child)
                    ]
                    candidates.add(parent if len(sibling_checkpoints) > 1 else path)
    except OSError:
        pass

    try:
        for path in outputs_dir.iterdir():
            if path.name == "modal-checkpoints":
                continue
            if path.is_dir() and any(
                child.is_dir() and is_checkpoint_dir(child) for child in path.iterdir()
            ):
                candidates.add(path)
    except OSError:
        pass

    ordered = sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))
    pruned: list[Path] = []
    for candidate in ordered:
        if any(candidate != root and path_is_relative_to(candidate, root) for root in pruned):
            continue
        pruned.append(candidate)
    return sorted(pruned, key=lambda path: str(path).lower())


def infer_model_name(run_dir: Path, train_meta: dict[str, Any] | None, trainer_state_path: Path | None) -> str:
    if train_meta:
        for key in ("model_id", "model_name", "base_model", "base_model_id"):
            value = train_meta.get(key)
            if isinstance(value, str) and value:
                return value

    lower_name = run_dir.name.lower()
    if "smolvlm2" in lower_name:
        return "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    if "lfm2.5" in lower_name or "lfm2" in lower_name:
        return "LiquidAI/lfm2.5-VL-450M"

    config_paths = [run_dir / "config.json"]
    if trainer_state_path:
        config_paths.append(trainer_state_path.parent / "config.json")
    for config_path in config_paths:
        config = read_json(config_path)
        if not config:
            continue
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            return str(architectures[0])
        model_type = config.get("model_type")
        if isinstance(model_type, str):
            return model_type
    return run_dir.name


def infer_source_type(run_dir: Path, model_name: str, train_meta: dict[str, Any] | None) -> str:
    name = f"{run_dir.name} {model_name}".lower()
    if "smolvlm" in name and ("transformers" in name or train_meta):
        return "hf-transformers"
    if "vlm_sft" in name or "lfm2" in name:
        return "leap-finetune"
    if train_meta:
        return "native-trainer"
    return "checkpoint"


def status_for_run(final_step: int | None, max_steps: int | None, checkpoints: list[dict[str, Any]]) -> str:
    if any(bool(row.get("is_final")) for row in checkpoints):
        return "completed"
    if final_step is not None and max_steps is not None and max_steps > 0 and final_step >= max_steps:
        return "completed"
    if final_step:
        return "partial"
    return "artifacts only"


def run_label(model_name: str, run_dir: Path) -> str:
    short_model = model_name.split("/")[-1]
    return f"{short_model} | {run_dir.name}"


def build_run_record(run_dir: Path, outputs_dir: Path) -> dict[str, Any]:
    train_meta = read_json(run_dir / "train_meta.json")
    trainer_state_path, trainer_state = find_primary_trainer_state(run_dir)
    history = parse_history(trainer_state)
    checkpoints = find_checkpoints(run_dir)
    model_name = infer_model_name(run_dir, train_meta, trainer_state_path)
    source_type = infer_source_type(run_dir, model_name, train_meta)

    final_step = None
    max_steps = None
    final_epoch = None
    if trainer_state:
        final_step = trainer_state.get("global_step")
        max_steps = trainer_state.get("max_steps")
        final_epoch = trainer_state.get("epoch")

    if final_step is None and not history.empty and "step" in history:
        final_step = int(history["step"].max())
    checkpoint_steps = [row["step"] for row in checkpoints if row.get("step") is not None]
    if checkpoint_steps:
        final_step = max(int(final_step or 0), max(checkpoint_steps))
    if final_epoch is None and not history.empty and "epoch" in history:
        final_epoch = float(history["epoch"].max())

    status = status_for_run(
        int(final_step) if final_step is not None else None,
        int(max_steps) if max_steps is not None else None,
        checkpoints,
    )
    size_bytes = dir_size_bytes(run_dir)
    try:
        relative_run_path = str(run_dir.relative_to(outputs_dir))
    except ValueError:
        relative_run_path = str(run_dir)

    overview = {
        "model": model_name,
        "run": run_dir.name,
        "run_path": str(run_dir),
        "relative_path": relative_run_path,
        "source_type": source_type,
        "final_step": int(final_step) if final_step is not None else None,
        "max_steps": int(max_steps) if max_steps is not None else None,
        "final_epoch": float(final_epoch) if final_epoch is not None else None,
        "status": status,
        "checkpoint_count": len(checkpoints),
        "artifact_size": human_size(size_bytes),
        "artifact_size_bytes": size_bytes,
        "history_rows": len(history),
    }

    summary = {}
    if trainer_state:
        summary = {key: value for key, value in trainer_state.items() if key != "log_history"}

    return {
        "label": run_label(model_name, run_dir),
        "run_dir": run_dir,
        "overview": overview,
        "history": history,
        "checkpoints": checkpoints,
        "train_meta": train_meta or {},
        "trainer_state_summary": summary,
        "trainer_state_path": str(trainer_state_path) if trainer_state_path else None,
    }


def discover_runs(outputs_dir: Path = OUTPUTS_DIR) -> list[dict[str, Any]]:
    return [build_run_record(run_dir, outputs_dir) for run_dir in run_dirs_from_outputs(outputs_dir)]


def gguf_artifacts(outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not outputs_dir.exists():
        return pd.DataFrame()
    for path in sorted(outputs_dir.glob("*.gguf")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "name": path.name,
                "kind": "mmproj" if path.name.startswith("mmproj") else "backbone",
                "size": human_size(stat.st_size),
                "size_bytes": stat.st_size,
                "modified": pd.to_datetime(stat.st_mtime, unit="s"),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_runs_cached() -> list[dict[str, Any]]:
    return discover_runs(OUTPUTS_DIR)


def overview_dataframe(runs: list[dict[str, Any]]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    columns = [
        "model",
        "source_type",
        "status",
        "final_step",
        "max_steps",
        "final_epoch",
        "checkpoint_count",
        "history_rows",
        "artifact_size",
        "relative_path",
    ]
    return pd.DataFrame([run["overview"] for run in runs])[columns]


def checkpoint_dataframe(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for checkpoint in run["checkpoints"]:
            rows.append(
                {
                    "run": run["label"],
                    "name": checkpoint["name"],
                    "step": checkpoint["step"],
                    "modified": checkpoint["modified"],
                    "size": checkpoint["size"],
                    "has_model_weights": checkpoint["has_model_weights"],
                    "is_final": checkpoint["is_final"],
                    "path": checkpoint["path"],
                }
            )
    return pd.DataFrame(rows)


def combined_history(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        history = run["history"]
        if history.empty:
            continue
        df = history.copy()
        df["run"] = run["label"]
        df["model"] = run["overview"]["model"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def line_chart_for_metrics(history: pd.DataFrame, metrics: list[str]) -> None:
    frames: list[pd.DataFrame] = []
    for metric in metrics:
        if metric not in history.columns:
            continue
        temp = history[["step", "run", metric]].dropna().rename(columns={metric: "value"})
        if temp.empty:
            continue
        temp["series"] = temp["run"] + " | " + metric
        frames.append(temp[["step", "series", "value"]])

    if not frames:
        st.info("No data found for this curve yet.")
        return

    chart_data = pd.concat(frames, ignore_index=True)
    pivot = chart_data.pivot_table(index="step", columns="series", values="value", aggfunc="last")
    st.line_chart(pivot.sort_index())


def render_overview(runs: list[dict[str, Any]]) -> None:
    st.subheader("Overview")
    if not runs:
        st.info("No local training outputs found yet.")
        return

    overview = overview_dataframe(runs)
    completed = int((overview["status"] == "completed").sum())
    total_checkpoints = int(overview["checkpoint_count"].sum())
    history_rows = int(overview["history_rows"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", len(runs))
    col2.metric("Completed", completed)
    col3.metric("Checkpoints", total_checkpoints)
    col4.metric("Log rows", history_rows)
    st.dataframe(overview, use_container_width=True, hide_index=True)


def render_curves(runs: list[dict[str, Any]]) -> None:
    st.subheader("Curves")
    history = combined_history(runs)
    if history.empty:
        st.info("No trainer_state.json log history is available for the selected runs.")
        return

    lr_columns = sorted(column for column in history.columns if column.startswith("lr/"))

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    with top_left:
        st.markdown("**Training loss**")
        line_chart_for_metrics(history, ["loss"])
    with top_right:
        st.markdown("**Eval loss**")
        line_chart_for_metrics(history, ["eval_loss"])
    with bottom_left:
        st.markdown("**Grad norm**")
        line_chart_for_metrics(history, ["grad_norm"])
    with bottom_right:
        st.markdown("**Learning rates**")
        line_chart_for_metrics(history, ["learning_rate", *lr_columns])

    with st.expander("Parsed log history"):
        display_columns = [
            column
            for column in ["run", *CORE_HISTORY_COLUMNS, *lr_columns]
            if column in history.columns
        ]
        st.dataframe(history[display_columns], use_container_width=True, hide_index=True)


def render_checkpoints(runs: list[dict[str, Any]]) -> None:
    st.subheader("Checkpoints")
    checkpoints = checkpoint_dataframe(runs)
    if checkpoints.empty:
        st.info("No checkpoint folders found for the selected runs.")
        return

    styled = checkpoints.sort_values(["run", "step"], na_position="last")
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_run_details(runs: list[dict[str, Any]]) -> None:
    st.subheader("Run details")
    if not runs:
        st.info("No runs selected.")
        return

    labels = [run["label"] for run in runs]
    selected_label = st.selectbox("Run", labels)
    run = next(item for item in runs if item["label"] == selected_label)

    st.write("Run path")
    st.code(str(run["run_dir"]), language="text")
    st.write("Trainer state path")
    st.code(str(run["trainer_state_path"] or "not found"), language="text")

    st.write("Summary")
    st.json(run["trainer_state_summary"])

    if run["train_meta"]:
        st.write("train_meta.json")
        st.json(run["train_meta"])

    history = run["history"]
    if not history.empty:
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download parsed history CSV",
            data=csv,
            file_name=f"{run['run_dir'].name}_history.csv",
            mime="text/csv",
        )
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("This run does not have log_history entries yet.")


def render_evaluation_link() -> None:
    st.subheader("Evaluation link")
    artifacts = gguf_artifacts(OUTPUTS_DIR)
    if artifacts.empty:
        st.info("No local GGUF artifacts found in outputs yet.")
    else:
        st.dataframe(artifacts, use_container_width=True, hide_index=True)

    st.write("Open the evaluation comparison app after you run model evaluations:")
    st.code("uv run streamlit run app/eval_compare.py", language="powershell")


def main() -> None:
    st.set_page_config(page_title="Bali flood training dashboard", layout="wide")
    st.title("Bali flood fine-tuning dashboard")

    with st.sidebar:
        st.header("Local outputs")
        st.code(str(OUTPUTS_DIR), language="text")
        if st.button("Refresh"):
            st.cache_data.clear()

    runs = load_runs_cached()
    if runs:
        selected_labels = st.sidebar.multiselect(
            "Runs",
            [run["label"] for run in runs],
            default=[run["label"] for run in runs],
        )
        selected_runs = [run for run in runs if run["label"] in selected_labels]
    else:
        selected_runs = []

    tab_overview, tab_curves, tab_checkpoints, tab_details, tab_eval = st.tabs(
        ["Overview", "Curves", "Checkpoints", "Run details", "Evaluation link"]
    )
    with tab_overview:
        render_overview(selected_runs)
    with tab_curves:
        render_curves(selected_runs)
    with tab_checkpoints:
        render_checkpoints(selected_runs)
    with tab_details:
        render_run_details(selected_runs)
    with tab_eval:
        render_evaluation_link()


if __name__ == "__main__":
    main()
