from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional
import shutil
import os
import json

# Import your working services
from services.process_miner import extract_simulation_parameters
from services.simulator import run_simulation
from services.log_parser import parse_event_log # Assuming you moved the log parser here
from services.insight_generator import generate_insights


app = FastAPI(title="Process Mining Simulation Engine")

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ScenarioPatch(BaseModel):
    activity_speedup: Optional[Dict[str, float]] = None
    resource_capacities: Optional[Dict[str, int]] = None

class WhatIfRequest(BaseModel):
    scenario_patch: ScenarioPatch
    sim_days: int = 30
    num_runs: int = 100

class InsightsRequest(BaseModel):
    simulation_results: dict

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/simulate/upload")
async def upload_log(file: UploadFile = File(...)):
    """Receives the CSV from .NET, parses it, and returns the baseline parameters."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run your existing extraction logic
        log, df = parse_event_log(file_location)
        params = extract_simulation_parameters(log, df)
        
        # Save params to disk or a fast cache (like Redis) for the What-If endpoint to use later
        with open("current_params.json", "w") as f:
            json.dump(params, f)
            
        return {"status": "success", "message": "Log processed", "parameters": params}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.post("/simulate/whatif")
async def run_whatif(request: WhatIfRequest):
    """Runs the SimPy simulation with injected variables."""
    if not os.path.exists("current_params.json"):
        raise HTTPException(status_code=400, detail="No baseline parameters found. Upload a log first.")
        
    with open("current_params.json", "r") as f:
        params = json.load(f)
        
    try:
        # Run your existing simulation logic
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
    """Takes simulation results and generates plain-English analysis via Claude."""
    try:
        insights_text = generate_insights(request.simulation_results)
        return {
            "status": "success", 
            "insights": insights_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))