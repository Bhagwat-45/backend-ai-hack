"""
Standalone verification script for the chat/conversation backend.

Run this after `pip install -r requirements.txt` (plus sqlalchemy and
python-multipart) and after dropping in database.py / models/db_models.py /
services/chat_service.py / router/. It does NOT need a running server -
it drives the FastAPI app directly via TestClient.

By default it MOCKS the Azure OpenAI call, so this works even without
real credentials in .env. Set TEST_REAL_AZURE=1 to also send one real
request to Azure using your .env credentials, to confirm those are wired
up correctly too.

Usage:
    cd backend
    pip install --break-system-packages pytest  # only if you don't have it
    python test_backend.py
"""
import os
import sys
import io
import contextlib

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_verify.db")

import pandas as pd

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f" - {detail}" if detail and status == FAIL else ""))
    return condition


def make_sample_csv(path: str):
    import random
    random.seed(7)
    activities = ["Receive Request", "Review", "Approve", "Reject", "Close"]
    resources = ["Alice", "Bob", "Carol"]
    rows = []
    for case_id in range(1, 31):
        t = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=case_id)
        rows.append([case_id, "Receive Request", t, random.choice(resources)])
        t += pd.Timedelta(hours=random.randint(1, 5))
        rows.append([case_id, "Review", t, random.choice(resources)])
        t += pd.Timedelta(hours=random.randint(1, 5))
        rows.append([case_id, random.choice(["Approve", "Reject"]), t, random.choice(resources)])
        t += pd.Timedelta(hours=random.randint(1, 3))
        rows.append([case_id, "Close", t, random.choice(resources)])
    pd.DataFrame(rows, columns=["case_id", "activity", "timestamp", "resource"]).to_csv(path, index=False)


def main():
    # Remove any leftover test DB from a previous run
    if os.path.exists("test_verify.db"):
        try:
            os.remove("test_verify.db")
        except PermissionError:
            print("(Note: a previous test_verify.db is still locked - close any other "
                  "process using it, e.g. a DB browser tool, or just delete it by hand)")

    try:
        import main as app_main  # noqa
        import database  # noqa - needed to dispose the engine for cleanup later
    except Exception as e:
        check("Backend imports without errors", False, str(e))
        print_summary()
        return

    check("Backend imports without errors", True)

    from fastapi.testclient import TestClient
    csv_path = "test_sample_log.csv"
    make_sample_csv(csv_path)

    noisy = io.StringIO()
    with contextlib.redirect_stdout(noisy):  # swallow pm4py's verbose prints
        with TestClient(app_main.app) as client:

            r = client.get("/")
            check("Health check (GET /)", r.status_code == 200, r.text)

            with open(csv_path, "rb") as f:
                r = client.post("/simulate/upload", files={"file": ("sample.csv", f, "text/csv")})
            ok = check("Upload CSV -> 200 with event_log_id", r.status_code == 200 and "event_log_id" in r.json(), r.text)
            event_log_id = r.json().get("event_log_id") if ok else None

            r = client.post("/api/conversation", json={"event_log_id": event_log_id})
            ok = check("Create conversation -> 200 with id", r.status_code == 200 and "id" in r.json(), r.text)
            conversation_id = r.json().get("id") if ok else None

            r = client.get("/api/conversation/does-not-exist")
            check("Unknown conversation -> 404 (not a crash)", r.status_code == 404, r.text)

            if os.environ.get("TEST_REAL_AZURE") == "1":
                r = client.post("/api/chat", json={
                    "conversation_id": conversation_id,
                    "message": "What's the biggest bottleneck in this process?",
                })
                check("Chat with REAL Azure credentials -> 200", r.status_code == 200, r.text)
            else:
                from unittest.mock import patch
                with patch("router.chat.call_azure_chat", return_value="Mocked response.") as mock_call:
                    r = client.post("/api/chat", json={
                        "conversation_id": conversation_id,
                        "message": "What's the biggest bottleneck in this process?",
                    })
                    check("Chat (mocked Azure) -> 200", r.status_code == 200, r.text)

                    if mock_call.call_args:
                        sent_prompt = mock_call.call_args[0][0][0]["content"]
                        check(
                            "Prompt stays bounded (< 12000 chars even on bigger logs)",
                            len(sent_prompt) < 12000,
                            f"length was {len(sent_prompt)}",
                        )

            r = client.get(f"/api/conversation/{conversation_id}")
            history = r.json() if r.status_code == 200 else []
            check("History has both user + assistant messages", len(history) == 2, str(history))

            r = client.post("/simulate/whatif", json={
                "event_log_id": event_log_id,
                "scenario_patch": {"resource_capacities": {"Alice": 2}},
                "sim_days": 5,
                "num_runs": 3,
            })
            check("What-if simulation runs against persisted event_log_id", r.status_code == 200, r.text)

            r = client.post("/simulate/whatif", json={"event_log_id": 999999, "scenario_patch": {}})
            check("What-if with bogus event_log_id -> 404 (not a crash)", r.status_code == 404, r.text)

    os.remove(csv_path)

    # On Windows, SQLite keeps the file handle open via SQLAlchemy's
    # connection pool until the engine is explicitly disposed - deleting
    # the file before that raises PermissionError (WinError 32). Disposing
    # first fixes it; the try/except is just a safety net in case anything
    # else (antivirus, an editor preview, etc.) is also holding the file.
    database.engine.dispose()
    if os.path.exists("test_verify.db"):
        try:
            os.remove("test_verify.db")
        except PermissionError:
            print("(Note: couldn't delete test_verify.db - safe to ignore or delete it by hand)")

    print_summary()


def print_summary():
    print()
    print("=" * 50)
    failed = [r for r in results if r[0] == FAIL]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for status, label, detail in failed:
            print(f"  - {label}: {detail}")
        sys.exit(1)
    else:
        print("Everything works.")


if __name__ == "__main__":
    main()