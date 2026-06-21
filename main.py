from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from sqlalchemy.orm import Session
import shutil
import os
import json
import uuid

from database import init_db, get_db
from models.db_models import EventLog

from router import chat as chat_router
from router import conversation as conversation_router

# Import your working services
from services.process_miner import extract_simulation_parameters
from services.simulator import run_simulation
from services.log_parser import parse_event_log  # Assuming you moved the log parser here
from services.insight_generator import generate_insights


app = FastAPI(title="Process Mining Simulation Engine")

# CORS wide open for now - lock this down before any real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(conversation_router.router)


@app.on_event("startup")
def on_startup():
    init_db()


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ScenarioPatch(BaseModel):
    activity_speedup: Optional[Dict[str, float]] = None
    resource_capacities: Optional[Dict[str, int]] = None

class WhatIfRequest(BaseModel):
    event_log_id: int  # NEW: which uploaded log this run is against
    scenario_patch: ScenarioPatch
    sim_days: int = 30
    num_runs: int = 100

class InsightsRequest(BaseModel):
    simulation_results: dict

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/simulate/upload")
async def upload_log(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Receives the CSV, parses it, and persists the baseline parameters as a
    new event_logs row - replacing the old current_params.json single file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Unique filename per upload so two uploads landing close together
    # can't read/write the same temp file (the old version used a fixed
    # "temp_{filename}" path with no collision protection).
    file_location = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run your existing extraction logic
        log, df = parse_event_log(file_location)
        params = extract_simulation_parameters(log, df)

        event_log = EventLog(
            filename=file.filename,
            parameters_json=json.dumps(params),
        )
        db.add(event_log)
        db.commit()
        db.refresh(event_log)

        return {
            "status": "success",
            "message": "Log processed",
            "event_log_id": event_log.id,
            "parameters": params,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


def _load_params(event_log_id: int, db: Session) -> dict:
    event_log = db.query(EventLog).filter(EventLog.id == event_log_id).first()
    if not event_log:
        raise HTTPException(status_code=404, detail="event_log_id not found. Upload a log first.")
    return json.loads(event_log.parameters_json)


@app.post("/simulate/run")
async def run_baseline(event_log_id: int, sim_days: int = 30, num_runs: int = 100,
                        db: Session = Depends(get_db)):
    """Run baseline simulation with no what-if patches, for a specific uploaded log."""
    params = _load_params(event_log_id, db)

    results = run_simulation(
        params=params,
        scenario_patch=None,  # no patches = reality as-is
        sim_days=sim_days,
        num_runs=num_runs
    )
    return {"status": "success", "results": results}


@app.post("/simulate/whatif")
async def run_whatif(request: WhatIfRequest, db: Session = Depends(get_db)):
    """Runs the SimPy simulation with injected variables, for a specific uploaded log."""
    params = _load_params(request.event_log_id, db)

    try:
        results = run_simulation(
            params=params,
            scenario_patch=request.scenario_patch.dict(exclude_unset=True),
            sim_days=request.sim_days,
            num_runs=request.num_runs
        )
        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/insights")
async def get_insights(request: InsightsRequest):
    """Takes simulation results and generates plain-English analysis via Azure OpenAI."""
    try:
        insights_text = generate_insights(request.simulation_results)
        return {
            "status": "success",
            "insights": insights_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/risk/analyze")
async def analyze_risk(
    event_log_id: int,
    db: Session = Depends(get_db)
):
    params = _load_params(event_log_id, db)

    activity_stats = params.get("activity_stats", {})
    resource_stats = params.get("resource_stats", {})

    register = []

    # --------------------------------------------------
    # Resource bottlenecks
    # --------------------------------------------------

    for resource, stats in resource_stats.items():

        task_count = stats.get("total_tasks", 0)
        avg_duration = stats.get("avg_task_duration", 0)

        score = 0

        if task_count > 5000:
            score += 3
        elif task_count > 2000:
            score += 2
        else:
            score += 1

        if avg_duration > 10000:
            score += 2
        elif avg_duration > 5000:
            score += 1

        if score >= 5:
            likelihood = 5
            impact = 4
        elif score >= 4:
            likelihood = 4
            impact = 4
        elif score >= 3:
            likelihood = 4
            impact = 3
        else:
            likelihood = 3
            impact = 2

        register.append({
            "name": f"Resource Bottleneck · {resource}",
            "cat": "operational",
            "l": likelihood,
            "i": impact,
            "owner": resource,
            "status": "Open",
            "evidence": {
                "tasks": task_count,
                "avg_duration": round(avg_duration, 0)
            }
        })

    # --------------------------------------------------
    # Long-running activities
    # --------------------------------------------------

    for activity, stats in activity_stats.items():

        overall = stats.get("overall", {})

        mean_seconds = overall.get("mean", 0)
        p95_seconds = overall.get("p95", mean_seconds)

        if mean_seconds < 5000:
            continue

        if mean_seconds > 12000:
            likelihood = 5
            impact = 5
        elif mean_seconds > 9000:
            likelihood = 4
            impact = 4
        else:
            likelihood = 3
            impact = 4

        register.append({
            "name": f"Activity Delay · {activity}",
            "cat": "process",
            "l": likelihood,
            "i": impact,
            "owner": "Operations",
            "status": "Open",
            "evidence": {
                "mean_seconds": round(mean_seconds, 0),
                "p95_seconds": round(p95_seconds, 0)
            }
        })

    # --------------------------------------------------
    # Manual processing dependency
    # --------------------------------------------------

    if "Invoice Manual Processing" in activity_stats:

        manual_mean = (
            activity_stats["Invoice Manual Processing"]
            .get("overall", {})
            .get("mean", 0)
        )

        register.append({
            "name": "Manual Processing Dependency",
            "cat": "operational",
            "l": 5,
            "i": 5,
            "owner": "Accounts Payable",
            "status": "Open",
            "evidence": {
                "mean_seconds": round(manual_mean, 0)
            }
        })

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    seen = set()
    deduped = []

    for risk in register:
        if risk["name"] not in seen:
            seen.add(risk["name"])
            deduped.append(risk)

    register = deduped

    # --------------------------------------------------
    # Sort highest risk first
    # --------------------------------------------------

    register = sorted(
        register,
        key=lambda r: r["l"] * r["i"],
        reverse=True
    )

    # --------------------------------------------------
    # Overview
    # --------------------------------------------------

    scores = [r["l"] * r["i"] for r in register]

    overview = {
        "risk_count": len(register),

        "critical_count": len([
            r for r in register
            if r["l"] * r["i"] >= 20
        ]),

        "high_count": len([
            r for r in register
            if 12 <= r["l"] * r["i"] < 20
        ]),

        "overall_score": (
            round(sum(scores) / len(scores))
            if scores else 0
        ),

        "top_risk": (
            register[0]["name"]
            if register else None
        )
    }

    # --------------------------------------------------
    # Matrix
    # --------------------------------------------------

    matrix = []

    for impact in range(5, 0, -1):

        row = []

        for likelihood in range(1, 6):

            count = len([
                r for r in register
                if r["l"] == likelihood
                and r["i"] == impact
            ])

            row.append({
                "likelihood": likelihood,
                "count": count
            })

        matrix.append({
            "impact": impact,
            "cells": row
        })

    # --------------------------------------------------
    # Risk Map
    # --------------------------------------------------

    nodes = []

    for activity, stats in activity_stats.items():

        mean_seconds = (
            stats.get("overall", {})
            .get("mean", 0)
        )

        if mean_seconds > 12000:
            risk = "critical"
        elif mean_seconds > 8000:
            risk = "high"
        elif mean_seconds > 4000:
            risk = "medium"
        else:
            risk = "low"

        nodes.append({
            "activity": activity,
            "risk": risk,
            "duration": round(mean_seconds, 0)
        })

    nodes = sorted(
        nodes,
        key=lambda n: n["duration"],
        reverse=True
    )

    controls = []

# Manual processing control
    if "Invoice Manual Processing" in activity_stats:
        controls.append({
            "name": "Invoice Automation Coverage",
            "eff": 35
        })

    # Approval bottleneck
    if any(
        r["name"] == "Activity Delay · Approve Invoice"
        for r in register
    ):
        controls.append({
            "name": "Approval Workflow Efficiency",
            "eff": 42
        })

    # Matching bottleneck
    if any(
        r["name"] == "Activity Delay · Match Invoice"
        for r in register
    ):
        controls.append({
            "name": "Invoice Matching Effectiveness",
            "eff": 68
        })

    # AP Clerk bottleneck
    if any(
        r["name"] == "Resource Bottleneck · Accounts Payable Clerk"
        for r in register
    ):
        controls.append({
            "name": "Accounts Payable Capacity",
            "eff": 48
        })

    # ERP performance
    if any(
        r["name"] == "Resource Bottleneck · ERP"
        for r in register
    ):
        controls.append({
            "name": "ERP Processing Efficiency",
            "eff": 62
        })

    # Verify Info delays
    if any(
        r["name"] == "Activity Delay · Verify Info"
        for r in register
    ):
        controls.append({
            "name": "Invoice Verification Control",
            "eff": 58
        })

    # Ensure at least something exists
    if not controls:
        controls = [
            {
                "name": "Process Monitoring",
                "eff": 75
            }
        ]

    return {
        "status": "success",
        "overview": overview,
        "register": register,
        "matrix": matrix,
        "map": nodes,
        "controls": controls
    }