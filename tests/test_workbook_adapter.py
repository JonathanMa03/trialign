from pathlib import Path

import pandas as pd

from trialign.data import load_corpus


def test_workbook_adapter_reports_review_state(tmp_path: Path):
    workbook = tmp_path / "review.xlsx"
    frame = pd.DataFrame([{
        "source": "Age >= 18 years",
        "parent_statement": "Age >= 18 years",
        "criterion_type": "inclusion",
        "domain": "person",
        "entity": "Age",
        "comparator": ">=",
        "value": 18,
        "unit": "years",
        "lower_bound": 18,
        "lower_inclusive": True,
        "negated": False,
        "review_status": "accepted",
        "validation_issues": "",
        "readiness_blockers": "",
        "trial_id": "NCT00000001",
        "criterion_id": "criterion-1",
        "parent_statement_id": "statement-1",
    }])
    with pd.ExcelWriter(workbook) as writer:
        frame.to_excel(writer, sheet_name="Human Review", index=False)
    rows, report = load_corpus(workbook)
    assert len(rows) == 1
    assert report.invalid_rows == 0
    assert report.review_statuses == {"accepted": 1}

