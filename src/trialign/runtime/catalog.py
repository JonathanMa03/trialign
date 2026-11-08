"""Reviewed corpus/import adapters; never reads pair labels for inference."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from trialign.data.corpus import load_parquet_corpus, normalize_workbook_row
from trialign.schemas import CriterionRecord

# Source defects established during the source audit; these are not pair labels.
SOURCE_BLOCKS = {
    "criterion-72ee99074cca2884": "Registry age endpoint conflicts with detailed prose.",
    "criterion-5df50ca2b63ccbea": "Lesion-measurability example is not a positive inclusion rule.",
}


def text_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\", "")).casefold().strip()


def local_context(source: str, full_text: str) -> tuple[str, str]:
    # Context is source text only, never annotations or model judgments.
    lines = full_text.splitlines()
    selected = text_key(source)
    for index, line in enumerate(lines):
        if selected and selected in text_key(line):
            excerpt = "\n".join(lines[max(0, index - 14) : index + 7])[:7000]
            indent = len(line) - len(line.lstrip())
            governing = []
            for previous in reversed(lines[:index]):
                if not previous.strip():
                    continue
                depth = len(previous) - len(previous.lstrip())
                if depth < indent:
                    if previous.rstrip().endswith(":") or re.search(
                        r"\b(unless|except|defined as|following)\b", previous, re.I
                    ):
                        governing.append(previous)
                    indent = depth
                if indent == 0:
                    break
            return excerpt, "\n".join(reversed(governing))
    return "", ""


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        self.records = {}
        self.titles = {}
        self.input_hashes = {}
        path = root / "data/processed/trial100_v1.parquet"
        if path.is_file():
            manifest = json.loads(path.with_suffix(".parquet.manifest.json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["parquet_sha256"]:
                raise ValueError("Frozen corpus checksum failed.")
            self.add(load_parquet_corpus(path), origin="frozen trial100_v1")
            self.input_hashes["frozen_corpus"] = manifest["parquet_sha256"]
        inventory = root / "data/processed/family_audit_v1/trial_inventory.json"
        if inventory.is_file():
            for trial_id, row in json.loads(inventory.read_text()).items():
                self.titles[trial_id] = row.get("identification", {}).get("briefTitle", trial_id)
                full = row.get("frozen_protocol_text", "")
                for record in self.records.values():
                    if record.trial_id == trial_id:
                        excerpt, governing = local_context(record.source, full)
                        record.retrieved_context = excerpt
                        record.governing_context = governing
        # Explicit source-role/heading defects in the frozen source release.
        for record in self.records.values():
            issues = []
            if record.criterion_id in SOURCE_BLOCKS:
                issues.append(SOURCE_BLOCKS[record.criterion_id])
            if record.trial_id == "NCT00091377" and record.source.strip() == "Chemotherapy":
                issues.append("Chemotherapy is a section heading, not an independent rule.")
            if (
                record.trial_id == "NCT07584967"
                and "strong inhibitors of CYP3A4" in record.source
                and record.criterion_type == "inclusion"
            ):
                issues.append("Stored inclusion role conflicts with source exclusion heading.")
            if issues:
                record.source_quality_issues = issues

    def add(self, records, *, origin):
        incoming = list(records)
        ids = [row.criterion_id for row in incoming]
        if len(ids) != len(set(ids)):
            raise ValueError("Imported criterion IDs are duplicated.")
        for row in incoming:
            prior = self.records.get(row.criterion_id)
            if prior and any(
                getattr(prior, k) != getattr(row, k)
                for k in [
                    "source",
                    "parent_statement",
                    "trial_id",
                    "criterion_type",
                    "source_version",
                    "protocol_hash",
                ]
            ):
                raise ValueError("An imported ID conflicts with an existing source record.")
        for row in incoming:
            if row.criterion_id not in self.records:
                row.import_origin = origin
                self.records[row.criterion_id] = row
        return len(incoming)

    def import_file(self, filename: str, content: bytes):
        if len(content) > 15_000_000:
            raise ValueError("Import limit is 15 MB.")
        suffix = Path(filename).suffix.casefold()
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                raw = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
            else:
                raw = json.loads(content)
                if isinstance(raw, dict):
                    raw = raw.get("criteria", raw.get("records", raw.get("rows")))
            if not isinstance(raw, list):
                raise ValueError("JSON must contain a criteria/records/rows array.")
        elif suffix == ".xlsx":
            workbook = pd.ExcelFile(io.BytesIO(content))
            sheet = next(
                (
                    s
                    for s in ["Human Review", "Validated Criteria", "Criteria"]
                    if s in workbook.sheet_names
                ),
                workbook.sheet_names[0],
            )
            raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object).to_dict("records")
        elif suffix == ".parquet":
            raw = pd.read_parquet(io.BytesIO(content)).to_dict("records")
            # Frozen Parquet encodes only these potentially heterogeneous scalars as JSON.
            for row in raw:
                for key in ["value", "value_raw", "value_normalized"]:
                    value = row.get(key)
                    if isinstance(value, str):
                        try:
                            row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
        elif suffix == ".csv":
            raw = pd.read_csv(io.BytesIO(content), dtype=object).to_dict("records")
        else:
            raise ValueError("Import XLSX, Parquet, JSON, JSONL or CSV criteria files.")
        if not raw or len(raw) > 10000:
            raise ValueError("Import between 1 and 10,000 criterion records.")
        records = []
        for index, row in enumerate(raw, 1):
            try:
                records.append(CriterionRecord.model_validate(normalize_workbook_row(row)))
            except Exception as exc:
                raise ValueError(
                    f"Import row {index} fails the criterion schema; no rows imported."
                ) from exc
        digest = hashlib.sha256(content).hexdigest()
        try:
            json.dumps([r.model_dump(mode="json") for r in records], allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("Import contains non-finite or unserializable values.") from exc
        self.add(records, origin="import:" + digest[:12])
        self.input_hashes["import:" + digest[:12]] = digest
        return {
            "records": len(records),
            "sha256": digest,
            "trials": len({r.trial_id for r in records}),
        }

    def trials(self):
        counts = Counter(r.trial_id for r in self.records.values())
        return [
            {"trial_id": t, "title": self.titles.get(t, t), "criteria": count}
            for t, count in sorted(counts.items())
        ]

    def for_trial(self, trial_id: str):
        result = sorted(
            (r for r in self.records.values() if r.trial_id == trial_id),
            key=lambda r: r.criterion_id,
        )
        if not result:
            raise ValueError("Selected trial has no imported criteria.")
        return result
