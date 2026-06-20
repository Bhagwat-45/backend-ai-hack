import io
import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter

# ─── Constants ───────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["case_id", "activity", "timestamp", "resource"]

COLUMN_MAPPING = {
    "case_id": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp",
    "resource": "org:resource"
}

# ─── Log Parser Functions ────────────────────────────────────────────────────

def load_raw_df(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    lines = [line.strip('"') for line in content.split('\n') if line.strip()]
    df = pd.read_csv(io.StringIO('\n'.join(lines)), low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def parse_event_log(filepath: str) -> tuple[EventLog, pd.DataFrame]:
    df = load_raw_df(filepath)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_slim = df[REQUIRED_COLUMNS].copy()
    df_renamed = df_slim.rename(columns=COLUMN_MAPPING)

    event_log = log_converter.apply(
        df_renamed,
        variant=log_converter.Variants.TO_EVENT_LOG
    )
    return event_log, df_slim

def get_log_summary(df: pd.DataFrame) -> dict:
    return {
        "total_cases": df["case_id"].nunique(),
        "total_events": len(df),
        "activities": df["activity"].unique().tolist(),
        "resources": df["resource"].unique().tolist(),
        "start_date": str(df["timestamp"].min()),
        "end_date": str(df["timestamp"].max())
    }