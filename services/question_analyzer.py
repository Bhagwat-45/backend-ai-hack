# services/question_analyzer.py

from services.case_analysis import (
    top_longest_cases,
    cases_with_most_rework,
    longest_activities
)


def analyze_question(df, question: str):

    q = question.lower()

    # Top delayed / longest cases
    if any(
        phrase in q
        for phrase in [
            "top cases",
            "longest cases",
            "delayed cases",
            "slowest cases",
            "longest running cases"
        ]
    ):
        return {
            "analysis_type": "top_longest_cases",
            "results": top_longest_cases(df)
        }

    # Rework
    if any(
        phrase in q
        for phrase in [
            "rework",
            "repeated activities",
            "redo",
            "repeated steps"
        ]
    ):
        return {
            "analysis_type": "rework",
            "results": cases_with_most_rework(df)
        }

    # Bottlenecks / waiting time
    if any(
        phrase in q
        for phrase in [
            "bottleneck",
            "waiting",
            "longest activity",
            "slowest activity",
            "activity delays"
        ]
    ):
        return {
            "analysis_type": "activity_wait_times",
            "results": longest_activities(df)
        }

    return None