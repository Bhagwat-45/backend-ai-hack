import pm4py
import pandas as pd
import numpy as np
from scipy import stats
import io
from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.concurrent_activities.log import get as concurrent_activities

REQUIRED_COLUMNS = ["case_id", "activity", "timestamp", "resource"]

COLUMN_MAPPING = {
    "case_id": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp",
    "resource": "org:resource"
}

WORKING_DAY_SECONDS  = 8 * 3600   # gaps above this are treated as idle time
PERCENTILE_FILTER    = 95          # also strip top 5 % of kept gaps
MIN_SAMPLES_AFTER_FILTER = 30      # fall back to capped full set if too few


def _filter_working_durations(durations: list) -> list:
    """
    Remove idle-time gaps from an inter-event duration list.

    Strategy
    --------
    1.  Keep only gaps <= WORKING_DAY_SECONDS  (drops overnight/weekend gaps)
    2.  Strip the top (100 - PERCENTILE_FILTER) % of what remains
    3.  If fewer than MIN_SAMPLES_AFTER_FILTER survive, fall back to the full
        list capped at WORKING_DAY_SECONDS so we always have something to fit
    """
    arr = np.array(durations, dtype=float)

    # Step 1 — keep only within-working-day gaps
    working = arr[arr <= WORKING_DAY_SECONDS]

    if len(working) < MIN_SAMPLES_AFTER_FILTER:
        # Not enough clean samples — cap the full set instead
        working = np.clip(arr, 0, WORKING_DAY_SECONDS)

    # Step 2 — strip top tail of remaining
    if len(working) >= 10:
        p_cutoff = np.percentile(working, PERCENTILE_FILTER)
        working = working[working <= p_cutoff]

    return working.tolist() if len(working) >= 2 else durations


def fit_distribution(durations: list) -> dict:
    """Fit best statistical distribution to (already-cleaned) duration data."""
    cleaned = _filter_working_durations(durations)

    if len(cleaned) < 2:
        val = cleaned[0] if cleaned else 0
        return {"type": "fixed", "value": val,
                "mean": val, "std": 0, "median": val, "p95": val}

    arr = np.array(cleaned)

    try:
        shape, loc, scale = stats.lognorm.fit(arr, floc=0)
        return {
            "type":   "lognormal",
            "shape":  float(shape),
            "loc":    float(loc),
            "scale":  float(scale),
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95":    float(np.percentile(arr, 95))
        }
    except Exception:
        return {
            "type":   "normal",
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95":    float(np.percentile(arr, 95))
        }


# ─── Main extractor ───────────────────────────────────────────────────────────

def extract_simulation_parameters(event_log: EventLog, df_raw: pd.DataFrame) -> dict:
    """
    Core function — extracts everything SimPy needs from the event log.
    """

    # ── 1. Compute inter-event durations ──────────────────────────────────────
    df = df_raw.copy()
    df = df.sort_values(["case_id", "timestamp"])

    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    df["duration_seconds"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds()

    # Drop last event of each case and zero-duration rows
    df = df.dropna(subset=["duration_seconds"])
    df = df[df["duration_seconds"] > 0]

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n--- Activity Duration Diagnostics ---")
    print(f"  {'Activity':<33} {'raw_med':>9} {'clean_med':>11} {'clean_p95':>11} {'kept%':>7}")
    print("  " + "-" * 78)
    for activity in sorted(df["activity"].unique()):
        raw = df[df["activity"] == activity]["duration_seconds"].values
        cleaned = np.array(_filter_working_durations(raw.tolist()))
        kept_pct = 100 * len(cleaned) / len(raw) if len(raw) > 0 else 0
        raw_med   = np.median(raw)
        clean_med = np.median(cleaned)
        clean_p95 = np.percentile(cleaned, 95)
        print(
            f"  {activity:<33}"
            f"  {raw_med/3600:>7.2f}h"
            f"  {clean_med:>8.0f}s ({clean_med/3600:.2f}h)"
            f"  {clean_p95:>8.0f}s"
            f"  {kept_pct:>5.1f}%"
        )
    print()

    # ── 2. Activity stats ─────────────────────────────────────────────────────
    activity_stats = {}
    for activity in df["activity"].unique():
        act_df = df[df["activity"] == activity]
        activity_stats[activity] = {
            "overall": fit_distribution(act_df["duration_seconds"].tolist()),
            "by_resource": {}
        }
        for resource in act_df["resource"].unique():
            res_df = act_df[act_df["resource"] == resource]
            if len(res_df) > 1:
                activity_stats[activity]["by_resource"][resource] = \
                    fit_distribution(res_df["duration_seconds"].tolist())

    # ── 3. Resource stats (using cleaned durations) ───────────────────────────
    resource_stats = {}
    for resource in df["resource"].unique():
        res_df = df[df["resource"] == resource]
        cleaned = np.array(_filter_working_durations(res_df["duration_seconds"].tolist()))
        resource_stats[resource] = {
            "total_tasks":          len(res_df),
            "total_active_seconds": float(cleaned.sum()),
            "activities_handled":   res_df["activity"].unique().tolist(),
            "avg_task_duration":    float(np.mean(cleaned))
        }

    # ── 4. Branching probabilities ────────────────────────────────────────────
    branching = {}
    df_sorted = df_raw.sort_values(["case_id", "timestamp"]).copy()
    df_sorted["next_activity"] = df_sorted.groupby("case_id")["activity"].shift(-1)

    for activity in df_sorted["activity"].unique():
        act_df = df_sorted[df_sorted["activity"] == activity].dropna(subset=["next_activity"])
        if len(act_df) > 0:
            counts = act_df["next_activity"].value_counts(normalize=True)
            branching[activity] = counts.to_dict()

    # ── 5. Arrival rate (cases/day) ───────────────────────────────────────────
    first_events = (
        df_raw.sort_values("timestamp")
              .groupby("case_id")["timestamp"]
              .first()
    )
    date_range = (df_raw["timestamp"].max() - df_raw["timestamp"].min()).days
    arrival_rate_per_day = len(first_events) / date_range if date_range > 0 else 0

    # ── 6. Handover matrix ────────────────────────────────────────────────────
    handover = {}
    df_sorted["next_resource"] = df_sorted.groupby("case_id")["resource"].shift(-1)
    for resource in df_sorted["resource"].unique():
        res_df = df_sorted[df_sorted["resource"] == resource].dropna(subset=["next_resource"])
        if len(res_df) > 0:
            counts = res_df["next_resource"].value_counts(normalize=True)
            handover[resource] = counts.to_dict()

    return {
        "activity_stats":          activity_stats,
        "resource_stats":          resource_stats,
        "branching_probabilities": branching,
        "arrival_rate_per_day":    float(arrival_rate_per_day),
        "handover_matrix":         handover,
        "total_cases":             df_raw["case_id"].nunique(),
        "total_events":            len(df_raw),
        "activities":              df_raw["activity"].unique().tolist(),
        "resources":               df_raw["resource"].unique().tolist()
    }