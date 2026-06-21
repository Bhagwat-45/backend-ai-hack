"""
Chat service - builds bounded, summarized prompts for the conversational
risk-analysis agent, and talks to Azure OpenAI.

This replaces the .NET ChatController.BuildPrompt(), which looped over
EVERY raw event-log row and EVERY past message with no limit at all -
the actual cause of the context-length errors. Here we instead:

  1. Summarize the parsed process-mining parameters (top activities by
     volume, resource utilization, common transitions) instead of raw events
  2. Cap conversation history to the last N turns
  3. Cap the final prompt size as a hard backstop, regardless of (1) and (2)
  4. Wrap the Azure call in proper error handling instead of letting
     response.raise_for_status() blow up uncaught
"""
import os
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_HISTORY_MESSAGES = 10       # last N messages (user+assistant combined)
MAX_ACTIVITIES_IN_PROMPT = 15   # top N activities/resources by volume
MAX_PROMPT_CHARS = 12000        # hard backstop regardless of the above


def _get_client() -> OpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
        )

    return OpenAI(
        base_url=endpoint,
        api_key=api_key,
        default_headers={"api-key": api_key},
    )


def summarize_parameters(params: Dict) -> str:
    """
    Condense process_miner's extract_simulation_parameters() output into a
    short, bounded text block - not the full nested JSON (which includes
    per-resource activity breakdowns and a full handover matrix, and can
    get large on real-world logs with many activities/resources).
    """
    lines = []

    lines.append(f"Total cases: {params.get('total_cases')}")
    lines.append(f"Total events: {params.get('total_events')}")
    lines.append(f"Arrival rate: {params.get('arrival_rate_per_day', 0):.2f} cases/day")

    activity_stats = params.get("activity_stats", {})
    resource_stats = params.get("resource_stats", {})

    # Rank activities by average duration (time impact), since process_miner
    # doesn't track a true per-activity occurrence count in activity_stats -
    # summing resource_stats.total_tasks per activity would double-count
    # (it's "tasks handled by resources that touch this activity", not
    # "occurrences of this activity"), so we don't fabricate that number.
    ranked_activities = sorted(
        activity_stats.keys(),
        key=lambda a: activity_stats[a].get("overall", {}).get("mean", 0),
        reverse=True,
    )

    lines.append("\nActivities (by average duration, longest first):")
    shown = ranked_activities[:MAX_ACTIVITIES_IN_PROMPT]
    for act in shown:
        overall = activity_stats[act].get("overall", {})
        mean_hours = overall.get("mean", 0) / 3600
        p95_hours = overall.get("p95", 0) / 3600
        lines.append(
            f"  - {act}: avg {mean_hours:.1f}h (p95 {p95_hours:.1f}h)"
        )

    omitted = len(ranked_activities) - len(shown)
    if omitted > 0:
        lines.append(f"  ... and {omitted} more activities omitted for brevity")

    lines.append("\nResource utilization (total tasks handled):")
    for resource, r_stats in list(resource_stats.items())[:MAX_ACTIVITIES_IN_PROMPT]:
        lines.append(
            f"  - {resource}: {r_stats.get('total_tasks', 0)} tasks, "
            f"avg {r_stats.get('avg_task_duration', 0) / 3600:.1f}h/task"
        )

    lines.append("\nMost common transitions:")
    branching = params.get("branching_probabilities", {})
    for act, targets in list(branching.items())[:MAX_ACTIVITIES_IN_PROMPT]:
        if not targets:
            continue
        top_target = max(targets, key=targets.get)
        lines.append(f"  - {act} -> {top_target} ({targets[top_target] * 100:.0f}% of the time)")

    summary = "\n".join(lines)
    return summary[:MAX_PROMPT_CHARS]


def build_messages(params: Dict, history: List, user_message: str) -> List[Dict]:
    """
    Build the OpenAI-style messages list: a system message with the
    summarized process data, the last N turns of history, then the new
    user message. `history` is a list of Message ORM objects (or anything
    with .sender_type / .content), already ordered oldest -> newest.
    """
    system_content = (
            "You are a senior business process risk analyst.\n\n"

            "You are helping users analyze process-mining results and operational risks.\n\n"

            "IMPORTANT RESPONSE RULES:\n"
            "- Be concise and executive-focused.\n"
            "- Use bullet points instead of long paragraphs.\n"
            "- Do NOT use headings like '## 1)' or '### What this means'.\n"
            "- Do NOT write consultant-style reports.\n"
            "- Focus on bottlenecks, delays, operational risks, compliance risks, and recommendations.\n"
            "- Use actual metrics from the process summary whenever possible.\n"
            "- Keep responses under 300 words unless explicitly asked for detailed analysis.\n"
            "- Prioritize the top 3-5 findings only.\n\n"

            "Preferred response format:\n\n"

            "Executive Summary\n"
            "- One short paragraph.\n\n"

            "Top Findings\n"
            "🔴 Critical Finding\n"
            "- Evidence\n"
            "- Business impact\n\n"

            "🟠 High-Risk Finding\n"
            "- Evidence\n"
            "- Business impact\n\n"

            "Recommendations\n"
            "1. Recommendation\n"
            "2. Recommendation\n"
            "3. Recommendation\n\n"

            "Process Summary:\n\n"
            f"{summarize_parameters(params)}"
        )

    messages = [{"role": "system", "content": system_content}]

    recent_history = history[-MAX_HISTORY_MESSAGES:]
    for msg in recent_history:
        role = "assistant" if msg.sender_type == "assistant" else "user"
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})
    return messages


def call_azure_chat(messages: List[Dict], model_name: str = None) -> str:
    """
    Calls Azure OpenAI chat completions. Raises RuntimeError with a clean
    message on failure (bad credentials, Azure-side error, etc.) instead of
    letting an unhandled exception turn into an opaque 500.
    """
    deployment_name = model_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI request failed: {e}") from e