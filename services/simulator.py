import simpy
import numpy as np
import random
from scipy import stats
from typing import Dict, List

MAX_ACTIVITY_SECONDS = 7 * 24 * 3600


def sample_duration(dist: Dict) -> float:
    try:
        if dist["type"] == "lognormal":
            raw = max(0, stats.lognorm.rvs(
                s=dist["shape"],
                loc=dist["loc"],
                scale=dist["scale"]
            ))
        elif dist["type"] == "normal":
            raw = max(0, np.random.normal(dist["mean"], dist["std"]))
        else:
            raw = dist.get("value", 0)
    except Exception:
        raw = dist.get("mean", 3600)

    return min(raw, MAX_ACTIVITY_SECONDS)


def _infer_start_activity(params: Dict) -> str:
    """
    Dynamically find the start activity instead of hardcoding 'Receive Invoice'.

    Strategy: the start activity is the one that never appears as a
    *target* in any branching probability — i.e. nothing transitions INTO it.
    Falls back to the activity with the most cases if no clean root is found.
    """
    all_activities = set(params["activities"])
    branching = params.get("branching_probabilities", {})

    # Collect every activity that is a destination of some transition
    activities_that_are_targets = set()
    for targets in branching.values():
        activities_that_are_targets.update(targets.keys())

    # Start activity = has activity_stats but is never a target
    candidates = all_activities - activities_that_are_targets

    # Filter to only activities that actually have stats (i.e. they appear in the log)
    candidates = {a for a in candidates if a in params.get("activity_stats", {})}

    if len(candidates) == 1:
        return candidates.pop()

    if len(candidates) > 1:
        # Multiple roots — pick the one with the highest total task count via resource_stats
        # as a proxy for "most common starting point"
        print(f"[warn] Multiple start activity candidates found: {candidates}. Picking by frequency.")
        activity_counts = {}
        for activity in candidates:
            count = sum(
                1 for r_stats in params["resource_stats"].values()
                if activity in r_stats.get("activities_handled", [])
            )
            activity_counts[activity] = count
        return max(activity_counts, key=activity_counts.get)

    # Fallback — every activity is a target (loop-heavy process), just pick first
    print("[warn] Could not determine start activity from branching graph. Using first activity in list.")
    return params["activities"][0]


def _infer_terminal_activities(params: Dict) -> set:
    """
    Dynamically find terminal activities — ones that have no outgoing
    transitions (or only self-loops), instead of hardcoding endpoint names.
    """
    branching = params.get("branching_probabilities", {})
    all_activities = set(params["activities"])

    terminals = set()
    for activity in all_activities:
        targets = branching.get(activity, {})
        # Terminal if: no outgoing transitions, or only transitions to itself
        non_self_targets = {t for t in targets if t != activity}
        if not non_self_targets:
            terminals.add(activity)

    if not terminals:
        # Fallback — nothing looks terminal, treat activities with no stats as terminals
        print("[warn] No terminal activities detected. Simulation will rely on max_steps to exit.")

    return terminals


class ProcessSimulator:
    def __init__(self, params: Dict, scenario_patch: Dict = None):
        self.params = params
        self.patch = scenario_patch or {}

        # ── FIX: derive start + terminal activities dynamically ──
        self.start_activity = _infer_start_activity(params)
        self.terminal_activities = _infer_terminal_activities(params)

        print(f"[info] Start activity   : {self.start_activity}")
        print(f"[info] Terminal activities: {self.terminal_activities}")

        self.results = {
            "cycle_times": [],
            "activity_wait_times": {a: [] for a in params["activities"]},
            "activity_process_times": {a: [] for a in params["activities"]},
            "resource_busy_time": {r: 0 for r in params["resources"]},
            "resource_total_time": {r: 0 for r in params["resources"]},
            "rework_count": 0,
            "completed_cases": 0,
            "rejected_cases": 0,
            "timeout_cases": 0,
        }

    def get_resource_capacity(self, resource: str) -> int:
        overrides = self.patch.get("resource_capacities", {})
        return overrides.get(resource, 1)

    def get_activity_speedup(self, activity: str) -> float:
        overrides = self.patch.get("activity_speedup", {})
        return overrides.get(activity, 1.0)

    def run(self, env: simpy.Environment, resources: Dict):
        case_start = env.now
        # ── FIX: use dynamic start activity ──
        current_activity = self.start_activity
        visited = []

        max_steps = 30
        steps = 0

        # ── FIX: use dynamic terminal activities ──
        while current_activity not in self.terminal_activities and steps < max_steps:
            steps += 1
            act_stats = self.params["activity_stats"].get(current_activity)
            if not act_stats:
                break

            resource_name = self._get_resource_for_activity(current_activity)
            resource = resources.get(resource_name)

            wait_start = env.now
            if resource:
                with resource.request() as req:
                    yield req
                    wait_time = env.now - wait_start
                    self.results["activity_wait_times"][current_activity].append(wait_time)

                    duration = sample_duration(act_stats["overall"])
                    speedup = self.get_activity_speedup(current_activity)
                    duration = duration / speedup

                    self.results["activity_process_times"][current_activity].append(duration)
                    self.results["resource_busy_time"][resource_name] += duration
                    yield env.timeout(duration)
            else:
                duration = sample_duration(act_stats["overall"])
                speedup = self.get_activity_speedup(current_activity)
                duration = duration / speedup
                yield env.timeout(duration)

            if current_activity in visited:
                self.results["rework_count"] += 1
            visited.append(current_activity)

            branching = self.params["branching_probabilities"].get(current_activity, {})
            if not branching:
                break

            next_activities = list(branching.keys())
            probabilities = list(branching.values())

            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            current_activity = np.random.choice(next_activities, p=probabilities)

        cycle_time = env.now - case_start
        self.results["cycle_times"].append(cycle_time)

        if current_activity in self.terminal_activities:
            # ── FIX: instead of guessing "Payment Confirmed" by name,
            # just count terminal exits as completed. If you want to split
            # completed vs rejected, add a "terminal_type" map to your params.
            self.results["completed_cases"] += 1
        else:
            self.results["timeout_cases"] += 1

    def _get_resource_for_activity(self, activity: str) -> str:
        resource_map = {}
        for resource, r_stats in self.params["resource_stats"].items():
            for act in r_stats["activities_handled"]:
                resource_map[act] = resource
        return resource_map.get(activity, self.params["resources"][0])
        # ── FIX: fallback to first resource instead of hardcoded "Accounts Payable Clerk"


def case_generator(env: simpy.Environment, simulator: ProcessSimulator,
                   resources: Dict, arrival_rate_per_day: float):
    arrival_rate_per_second = arrival_rate_per_day / (24 * 3600)
    while True:
        inter_arrival = np.random.exponential(1 / arrival_rate_per_second)
        yield env.timeout(inter_arrival)
        env.process(simulator.run(env, resources))


def run_simulation(params: Dict, scenario_patch: Dict = None,
                   sim_days: int = 30, num_runs: int = 100) -> Dict:
    all_cycle_times = []
    all_rework_counts = []
    all_completion_rates = []
    all_resource_utilization = []
    all_timeout_counts = []

    sim_duration_seconds = sim_days * 24 * 3600

    for run in range(num_runs):
        env = simpy.Environment()
        simulator = ProcessSimulator(params, scenario_patch)

        resources = {
            resource: simpy.Resource(
                env,
                capacity=simulator.get_resource_capacity(resource)
            )
            for resource in params["resources"]
        }

        for r in simulator.results["resource_total_time"]:
            simulator.results["resource_total_time"][r] = sim_duration_seconds

        env.process(case_generator(
            env, simulator, resources,
            params["arrival_rate_per_day"]
        ))

        env.run(until=sim_duration_seconds)

        if simulator.results["cycle_times"]:
            all_cycle_times.extend(simulator.results["cycle_times"])

        all_rework_counts.append(simulator.results["rework_count"])
        all_timeout_counts.append(simulator.results["timeout_cases"])

        total_cases = (
            simulator.results["completed_cases"]
            + simulator.results["rejected_cases"]
            + simulator.results["timeout_cases"]
        )
        if total_cases > 0:
            all_completion_rates.append(
                simulator.results["completed_cases"] / total_cases
            )

        run_utilization = {}
        for resource in params["resources"]:
            busy = simulator.results["resource_busy_time"].get(resource, 0)
            total = sim_duration_seconds
            run_utilization[resource] = min(busy / total, 1.0) if total > 0 else 0
        all_resource_utilization.append(run_utilization)

    if not all_cycle_times:
        raise ValueError(
            "Simulation produced no completed cycle times. "
            "Check arrival_rate_per_day, branching_probabilities, "
            "activity_stats, and process termination paths.\n"
            f"Hint: MAX_ACTIVITY_SECONDS={MAX_ACTIVITY_SECONDS}."
        )

    cycle_arr = np.array(all_cycle_times)

    avg_utilization = {}
    for resource in params["resources"]:
        avg_utilization[resource] = float(np.mean([
            run[resource] for run in all_resource_utilization
        ]))

    avg_timeout = float(np.mean(all_timeout_counts))
    if avg_timeout > 0:
        print(
            f"[warn] avg {avg_timeout:.1f} cases/run hit max_steps and exited "
            "without reaching a terminal activity."
        )

    return {
        "cycle_time": {
            "mean_seconds": float(np.mean(cycle_arr)),
            "mean_days": float(np.mean(cycle_arr) / 86400),
            "median_days": float(np.median(cycle_arr) / 86400),
            "p95_days": float(np.percentile(cycle_arr, 95) / 86400),
            "p99_days": float(np.percentile(cycle_arr, 99) / 86400),
            "std_days": float(np.std(cycle_arr) / 86400),
        },
        "resource_utilization": avg_utilization,
        "avg_rework_per_run": float(np.mean(all_rework_counts)),
        "avg_timeout_per_run": avg_timeout,
        "completion_rate": (
            float(np.mean(all_completion_rates)) if all_completion_rates else 0
        ),
        "num_runs": num_runs,
        "sim_days": sim_days,
        "scenario_patch": scenario_patch or {},
        # ── bonus: expose what was inferred so the frontend can show it ──
        "inferred_start_activity": ProcessSimulator(params, scenario_patch).start_activity,
    }