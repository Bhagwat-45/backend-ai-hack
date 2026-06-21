# services/case_analysis.py

import pandas as pd


def top_longest_cases(df: pd.DataFrame, limit: int = 10):

    working_df = df.copy()

    working_df["timestamp"] = pd.to_datetime(
        working_df["timestamp"],
        errors="coerce"
    )

    working_df = working_df.dropna(subset=["timestamp"])

    case_metrics = (
        working_df.groupby("case_id")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max")
        )
    )

    case_metrics["duration_hours"] = (
        case_metrics["end_time"] -
        case_metrics["start_time"]
    ).dt.total_seconds() / 3600

    case_metrics = (
        case_metrics
        .sort_values("duration_hours", ascending=False)
        .head(limit)
        .reset_index()
    )

    case_metrics["start_time"] = (
        case_metrics["start_time"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    case_metrics["end_time"] = (
        case_metrics["end_time"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    case_metrics["duration_hours"] = (
        case_metrics["duration_hours"]
        .round(2)
    )

    return case_metrics.to_dict(orient="records")


def cases_with_most_rework(df: pd.DataFrame, limit: int = 10):

    working_df = df.copy()

    counts = (
        working_df.groupby(
            ["case_id", "activity"]
        )
        .size()
        .reset_index(name="occurrences")
    )

    rework = counts[
        counts["occurrences"] > 1
    ]

    if rework.empty:
        return []

    summary = (
        rework.groupby("case_id")
        .agg(
            rework_count=("occurrences", "sum")
        )
        .sort_values(
            "rework_count",
            ascending=False
        )
        .head(limit)
        .reset_index()
    )

    return summary.to_dict(
        orient="records"
    )


def longest_activities(
    df: pd.DataFrame,
    limit: int = 10
):

    working_df = df.copy()

    working_df["timestamp"] = pd.to_datetime(
        working_df["timestamp"],
        errors="coerce"
    )

    working_df = working_df.dropna(
        subset=["timestamp"]
    )

    working_df = working_df.sort_values(
        ["case_id", "timestamp"]
    )

    working_df["next_timestamp"] = (
        working_df.groupby("case_id")["timestamp"]
        .shift(-1)
    )

    working_df["wait_hours"] = (
        working_df["next_timestamp"] -
        working_df["timestamp"]
    ).dt.total_seconds() / 3600

    result = (
        working_df.groupby("activity")
        .agg(
            avg_wait_hours=("wait_hours", "mean"),
            max_wait_hours=("wait_hours", "max")
        )
        .sort_values(
            "avg_wait_hours",
            ascending=False
        )
        .head(limit)
        .reset_index()
    )

    result["avg_wait_hours"] = (
        result["avg_wait_hours"]
        .round(2)
    )

    result["max_wait_hours"] = (
        result["max_wait_hours"]
        .round(2)
    )

    return result.to_dict(
        orient="records"
    )