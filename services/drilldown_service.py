# services/drilldown_service.py

from sqlalchemy.orm import Session

from models.db_models import EventLog
from services.blob_storage import download_csv_to_dataframe
from services.question_analyzer import analyze_question


def run_drilldown_analysis(
    event_log_id: int,
    question: str,
    db: Session
):

    event_log = (
        db.query(EventLog)
        .filter(EventLog.id == event_log_id)
        .first()
    )

    if not event_log:
        return None

    if not event_log.blob_path:
        return None

    df = download_csv_to_dataframe(
        event_log.blob_path
    )

    return analyze_question(
        df,
        question
    )