import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

STARTER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STARTER))

from agentic import config  # noqa: E402
from data.models import cultpass  # noqa: E402


def _move_dates_to_today(db_path: Path) -> None:
    """Notebook 01 dates the sample data relative to the day it ran. Shift the copy to today so the
    tests don't start failing once those dates pass: experiences start tomorrow (same order and
    spacing), and the seeded reservations count toward this month's quota."""
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        experiences = session.query(cultpass.Experience).all()
        now = datetime.now()
        shift = now + timedelta(days=1) - min(e.when for e in experiences)
        for e in experiences:
            e.when += shift
        for r in session.query(cultpass.Reservation).all():
            r.created_at = now
        session.commit()
    engine.dispose()


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    """Run every test against fresh copies of the databases built by notebooks 01 and 02."""
    for name, path in [("UDAHUB_DB_PATH", config.UDAHUB_DB_PATH), ("CULTPASS_DB_PATH", config.CULTPASS_DB_PATH)]:
        if not path.exists():
            pytest.skip(f"{path} missing: run 01_external_db_setup and 02_core_db_setup first")
        copy = tmp_path / path.name
        shutil.copy(path, copy)
        monkeypatch.setattr(config, name, copy)
    _move_dates_to_today(config.CULTPASS_DB_PATH)
    monkeypatch.setattr(config, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(config, "OUTBOX_PATH", tmp_path / "outbox.jsonl")
    yield
