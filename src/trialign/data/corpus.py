"""Modern AutoCrit workbook adapter and immutable corpus export helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import ValidationError

from trialign.schemas import CriterionRecord


REQUIRED_COLUMNS = {
    "source",
    "parent_statement",
    "criterion_type",
    "domain",
    "entity",
    "review_status",
    "trial_id",
    "criterion_id",
    "parent_statement_id",
}
LIST_COLUMNS = {"validation_issues", "readiness_blockers"}
BOOLEAN_COLUMNS = {"lower_inclusive", "upper_inclusive", "negated", "export_ready"}


@dataclass(frozen=True)
class CorpusValidationReport:
    path: str
    sheet: str
    rows_read: int
    valid_rows: int
    invalid_rows: int
    trials: int
    review_statuses: dict[str, int]
    errors: list[dict[str, Any]]
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _missing_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_bool(value: Any) -> bool | None:
    value = _missing_to_none(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _as_list(value: Any) -> list[str]:
    value = _missing_to_none(value)
    if value is None:
        return []
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def normalize_workbook_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: _missing_to_none(value) for key, value in row.items()}
    for column in LIST_COLUMNS:
        normalized[column] = _as_list(normalized.get(column))
    for column in BOOLEAN_COLUMNS:
        if column in normalized:
            normalized[column] = _as_bool(normalized.get(column))
    normalized["review_status"] = str(normalized.get("review_status") or "pending").casefold()
    normalized["criterion_type"] = str(normalized.get("criterion_type") or "").casefold()
    normalized["logical_operator"] = str(
        normalized.get("logical_operator") or "standalone"
    ).casefold()
    return normalized


def load_corpus(
    path: str | Path,
    *,
    sheet: str = "Human Review",
    reviewed_only: bool = False,
    reviewed_statuses: Iterable[str] = ("accepted", "corrected"),
    fail_on_invalid: bool = False,
) -> tuple[list[CriterionRecord], CorpusValidationReport]:
    """Load and validate a Modern AutoCrit review workbook.

    Invalid rows are reported rather than silently dropped. Set ``fail_on_invalid``
    when producing a frozen dataset.
    """
    source = Path(path)
    frame = pd.read_excel(source, sheet_name=sheet, dtype=object)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {', '.join(missing)}")

    accepted = {status.casefold() for status in reviewed_statuses}
    records: list[CriterionRecord] = []
    errors: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    for excel_row, raw in enumerate(frame.to_dict(orient="records"), start=2):
        row = normalize_workbook_row(raw)
        status = row["review_status"]
        statuses[status] = statuses.get(status, 0) + 1
        if reviewed_only and status not in accepted:
            continue
        try:
            records.append(CriterionRecord.model_validate(row))
        except ValidationError as exc:
            errors.append({
                "excel_row": excel_row,
                "criterion_id": row.get("criterion_id"),
                "trial_id": row.get("trial_id"),
                "errors": exc.errors(include_url=False),
            })

    if fail_on_invalid and errors:
        raise ValueError(f"{len(errors)} workbook rows failed schema validation")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = CorpusValidationReport(
        path=str(source.resolve()),
        sheet=sheet,
        rows_read=len(frame),
        valid_rows=len(records),
        invalid_rows=len(errors),
        trials=len({record.trial_id for record in records}),
        review_statuses=statuses,
        errors=errors,
        source_sha256=digest,
    )
    return records, report


def write_corpus(
    records: Iterable[CriterionRecord],
    output: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write an immutable-style Parquet corpus and adjacent metadata manifest."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.model_dump(mode="json") for record in records]
    pd.DataFrame(rows).to_parquet(destination, index=False)
    manifest = {
        "schema_version": "0.1.0",
        "rows": len(rows),
        "trials": len({row["trial_id"] for row in rows}),
        "criterion_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row["criterion_id"] for row in rows)).encode()
        ).hexdigest(),
        **(metadata or {}),
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_parquet_corpus(path: str | Path) -> list[CriterionRecord]:
    """Read a corpus previously written by :func:`write_corpus`."""
    frame = pd.read_parquet(path)
    records = []
    for row in frame.to_dict(orient="records"):
        normalized = normalize_workbook_row(row)
        records.append(CriterionRecord.model_validate(normalized))
    return records

