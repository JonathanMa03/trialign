"""Isolated uploaded datasets with flexible columns and explicit source provenance."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from trialign.data.corpus import normalize_workbook_row
from trialign.schemas import CriterionRecord

ALIASES = {
    "source": [
        "criterion",
        "criteria",
        "eligibility criteria",
        "eligibility criterion",
        "text",
        "requirement",
        "criterion text",
        "inclusion criteria",
        "exclusion criteria",
    ],
    "trial_id": ["trial", "trial id", "nct id", "nct number", "study", "study id", "protocol id"],
    "criterion_type": ["type", "criteria type", "eligibility type", "section", "category"],
    "entity": ["concept", "clinical concept"],
    "parent_statement": ["parent", "parent text"],
    "review_status": ["human review status"],
}


def key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def tables(filename, content):
    suffix = Path(filename).suffix.casefold()
    if suffix == ".xlsx":
        with pd.ExcelFile(io.BytesIO(content)) as workbook:
            sheets = workbook.sheet_names
            # Read every sheet; prefer reviewed rows when repeated IDs occur.
            sheets = sorted(sheets, key=lambda s: s not in {"Human Review", "Validated Criteria"})
            return [(s, pd.read_excel(workbook, sheet_name=s, dtype=object)) for s in sheets]
    if suffix == ".csv":
        return [("", pd.read_csv(io.BytesIO(content), dtype=object))]
    if suffix == ".parquet":
        frame = pd.read_parquet(io.BytesIO(content))
        for column in ["value", "value_raw", "value_normalized"]:
            if column in frame:
                frame[column] = frame[column].map(decode_scalar)
        return [("", frame)]
    if suffix in {".json", ".jsonl"}:
        raw = (
            [json.loads(line) for line in content.decode().splitlines() if line.strip()]
            if suffix == ".jsonl"
            else json.loads(content)
        )
        if isinstance(raw, dict):
            raw = raw.get("criteria", raw.get("records", raw.get("rows")))
        if not isinstance(raw, list):
            raise ValueError("JSON must contain a criteria/records/rows array.")
        return [("", pd.DataFrame(raw))]
    raise ValueError("Upload XLSX, Parquet, JSON, JSONL or CSV criteria files.")


def decode_scalar(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def expanded_tables(filename, content, text_column):
    for sheet, frame in tables(filename, content):
        columns = {key(c): c for c in frame.columns}
        if not text_column and "inclusioncriteria" in columns and "exclusioncriteria" in columns:
            for field, role in [
                ("inclusioncriteria", "inclusion"),
                ("exclusioncriteria", "exclusion"),
            ]:
                expanded = frame.copy()
                expanded["source"] = frame[columns[field]]
                expanded["criterion_type"] = role
                yield sheet, expanded
        elif not text_column and "triala" in columns and "trialb" in columns:
            for field, title in [("triala", "Trial A"), ("trialb", "Trial B")]:
                yield sheet, pd.DataFrame({"source": frame[columns[field]], "trial_id": title})
        else:
            yield sheet, frame


def parse_upload(files, *, text_column="", trial_column="", role=""):
    if not files or len(files) > 10:
        raise ValueError("Upload between 1 and 10 files.")
    if role not in {"", "inclusion", "exclusion"}:
        raise ValueError("Choose a valid eligibility role.")
    signature = hashlib.sha256()
    prepared, warnings = [], []
    originals, duplicate_count = {}, 0
    for item in files:
        filename = Path(item["filename"]).name
        content = item["content"]
        side = item.get("side", "")
        if side not in {"", "A", "B"}:
            raise ValueError("Invalid upload side.")
        if len(content) > 15_000_000:
            raise ValueError("Upload limit is 15 MB per file.")
        digest = hashlib.sha256(content).hexdigest()
        signature.update(
            json.dumps([filename, digest, side, text_column, trial_column, role]).encode()
        )
        found = 0
        for sheet, frame in expanded_tables(filename, content, text_column):
            columns = {key(c): c for c in frame.columns}
            mapped = {}
            for field in [
                "source",
                "trial_id",
                "criterion_type",
                "entity",
                "parent_statement",
                "review_status",
            ]:
                override = (
                    text_column
                    if field == "source"
                    else trial_column
                    if field == "trial_id"
                    else ""
                )
                choices = [override] if override else [field, *ALIASES.get(field, [])]
                mapped[field] = next((columns[key(c)] for c in choices if key(c) in columns), None)
            if mapped["source"] is None:
                if len(frame.columns) == 1 and not text_column:
                    mapped["source"] = frame.columns[0]
                else:
                    warnings.append(
                        f"Skipped {filename} / {sheet or 'table'}: no criterion-text column. "
                        f"Available columns: {', '.join(map(str, frame.columns))[:240]}."
                    )
                    continue
            source_header = key(mapped["source"])
            for index, raw in enumerate(frame.to_dict("records"), 2):
                row = normalize_workbook_row(raw)
                for field, column in mapped.items():
                    if column is not None:
                        row[field] = normalize_workbook_row({field: raw.get(column)}).get(field)
                source = str(row.get("source") or "").strip()
                if not source:
                    continue
                row["source"] = source
                supplied_id = str(row.get("trial_id") or "").strip()
                trial = supplied_id or (sheet if sheet else Path(filename).stem)
                identity = (
                    "side-" + side
                    if side
                    else (
                        "trial-" + supplied_id
                        if supplied_id
                        else f"file-{digest[:10]}-sheet-{sheet}"
                    )
                )
                explicit_role = str(row.get("criterion_type") or "").strip().casefold()
                scope_hint = key(sheet) + " " + source_header
                if explicit_role.startswith("inclusion"):
                    explicit_role = "inclusion"
                elif explicit_role.startswith("exclusion"):
                    explicit_role = "exclusion"
                elif role:
                    explicit_role = role
                elif "exclusioncriteria" in scope_hint or re.match(r"^exclusion\s*:", source, re.I):
                    explicit_role = "exclusion"
                elif "inclusioncriteria" in scope_hint or re.match(r"^inclusion\s*:", source, re.I):
                    explicit_role = "inclusion"
                unknown_role = explicit_role not in {"inclusion", "exclusion"}
                row["criterion_type"] = explicit_role if not unknown_role else "unspecified"
                if unknown_role:
                    row["source_quality_issues"] = [
                        "Eligibility role is unspecified in this upload; "
                        "clarify inclusion/exclusion before assigning a relationship."
                    ]
                    row["role_unspecified"] = True
                row.setdefault("domain", "unspecified")
                row["domain"] = row.get("domain") or "unspecified"
                row["entity"] = row.get("entity") or "unspecified"
                row["parent_statement"] = row.get("parent_statement") or source
                row["original_criterion_id"] = row.get("criterion_id")
                row["original_trial_id"] = supplied_id or None
                row["upload_file"] = filename
                row["upload_sheet"] = sheet
                row["upload_row"] = index
                row["source_version"] = row.get("source_version") or "uploaded-" + digest[:12]
                found += 1
                original_id = row.get("original_criterion_id")
                if original_id:
                    duplicate_key = (identity, str(original_id))
                    previous = originals.get(duplicate_key)
                    if previous:
                        same = all(
                            previous.get(k) == row.get(k)
                            for k in ["source", "parent_statement", "criterion_type"]
                        )
                        reviewed_view = previous["upload_file"] == filename and previous[
                            "upload_sheet"
                        ] in {"Human Review", "Validated Criteria"}
                        if not same and not reviewed_view:
                            raise ValueError(
                                f"Conflicting repeated criterion ID in {filename}. "
                                "Supply one reviewed version per trial."
                            )
                        duplicate_count += 1
                        continue
                    originals[duplicate_key] = row
                prepared.append((identity, trial, side, row))
        if not found:
            detail = warnings[-1] if warnings else "No nonempty criterion text found."
            raise ValueError(
                f"No criteria found in {filename}. {detail} "
                "Set the criterion-text column under column mapping if needed."
            )
    if len(prepared) > 10000:
        raise ValueError("Upload at most 10,000 criteria.")
    if duplicate_count:
        warnings.append(
            f"Skipped {duplicate_count} repeated criterion IDs; "
            "reviewed sheets take priority within a workbook."
        )
    dataset_id = "upload-" + signature.hexdigest()[:32]
    records, titles, seen = [], {}, set()
    side_ids = {
        side: {
            row.get("original_trial_id")
            for _, _, s, row in prepared
            if s == side and row.get("original_trial_id")
        }
        for side in ["A", "B"]
    }
    if any(len(ids) > 1 for ids in side_ids.values()):
        raise ValueError(
            "A separate-trial upload contains multiple trial IDs. "
            "Use combined upload, or supply one trial per side."
        )
    for identity, title, side, row in prepared:
        trial_id = "uploaded-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        titles[trial_id] = (f"Trial {side}: " if side else "") + title
        row["trial_id"] = trial_id
        criterion = hashlib.sha256(
            json.dumps(
                [
                    identity,
                    row.get("original_criterion_id"),
                    row["source"],
                    row["parent_statement"],
                    row["criterion_type"],
                    row.get("upload_file"),
                    row.get("upload_sheet"),
                    row["upload_row"],
                ]
            ).encode()
        ).hexdigest()[:24]
        row["criterion_id"] = dataset_id + "-" + criterion
        row["parent_statement_id"] = (
            dataset_id
            + "-parent-"
            + hashlib.sha256((identity + row["parent_statement"]).encode()).hexdigest()[:24]
        )
        try:
            record = CriterionRecord.model_validate(normalize_workbook_row(row))
        except Exception as exc:
            raise ValueError(
                f"{row['upload_file']} / {row['upload_sheet'] or 'table'}, "
                f"row {row['upload_row']}: invalid criterion fields ({type(exc).__name__})."
            ) from exc
        if record.criterion_id not in seen:
            seen.add(record.criterion_id)
            records.append(record.model_dump(mode="json"))
    json.dumps(records, allow_nan=False)
    counts = Counter(r["trial_id"] for r in records)
    trials = [
        {
            "trial_id": t,
            "title": titles[t],
            "criteria": count,
            "preview": [r["source"] for r in records if r["trial_id"] == t][:2],
        }
        for t, count in counts.items()
    ]
    missing_roles = sum(bool(r.get("role_unspecified")) for r in records)
    if missing_roles:
        warnings.append(
            f"{missing_roles} criteria have no inclusion/exclusion role. "
            "Set a role when known, or keep these for source review."
        )
    return {
        "dataset_id": dataset_id,
        "records": records,
        "trials": trials,
        "warnings": warnings,
        "files": [Path(f["filename"]).name for f in files],
    }
