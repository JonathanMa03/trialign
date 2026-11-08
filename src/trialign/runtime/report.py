"""Readable comparison exports generated from the immutable saved run snapshot."""

from __future__ import annotations

import re
import textwrap
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def readable(value):
    return str(value or "").replace("_", " ").capitalize()


def comparison_workbook(job):
    """Return an XLSX report; model proposals and human reviews stay distinct."""
    wb = Workbook()
    wb.remove(wb.active)

    def sheet(name, headers, rows, widths):
        ws = wb.create_sheet(name)
        ws.append(headers)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        for col, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(col)].width = width
        for row in ws:
            for cell in row:
                if isinstance(cell.value, str):
                    # Keep source text literal, including strings beginning with '='.
                    cell.value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cell.value)
                    if len(cell.value) > 32767:
                        cell.value = cell.value[:32650] + (
                            "… [Excel cell limit reached; full text remains in the saved run.]"
                        )
                    cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.font = Font(name="Calibri", size=11)
            if row[0].row > 1:
                line_counts = []
                for cell, width in zip(row, widths, strict=True):
                    wrapped = [
                        textwrap.wrap(line, width=max(12, int(width) - 3))
                        for line in str(cell.value or "").split("\n")
                    ]
                    line_counts.append(sum(max(1, len(parts)) for parts in wrapped))
                lines = max(line_counts)
                ws.row_dimensions[row[0].row].height = min(409, max(45, lines * 16 + 10))
                if row[0].row % 2 == 0:
                    for cell in row:
                        cell.fill = PatternFill("solid", fgColor="F4F6F7")
        for cell in ws[1]:
            cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="941D38")
        ws.row_dimensions[1].height = 32
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = "landscape"
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.print_title_rows = "1:1"
        return ws

    summary = job["summary"]
    sheet(
        "Summary",
        ["Item", "Value"],
        [
            ["Report", "TriAlign comparison evidence and review"],
            ["Trial A", f"{job['trial_a']} · {job.get('title_a', '')}"],
            ["Trial B", f"{job['trial_b']} · {job.get('title_b', '')}"],
            ["Criteria A / B", f"{len(job['criteria_a'])} / {len(job['criteria_b'])}"],
            ["Run status", job["status"]],
            ["Run message", job.get("message", "")],
            ["Assigned candidate pairs", summary["candidate_pairs"]],
            ["Examined pairs", summary["compared_pairs"]],
            ["Evidence-supported proposals", summary["supported_pairs"]],
            ["Pairs needing review", summary["needs_review"]],
            ["Pairs with saved human review", summary["reviewed_pairs"]],
            [
                "Interpretation",
                "Model/tool labels are proposals. Human review is shown separately. "
                "Unexamined pairs have no relationship decision; unmatched criteria are not "
                "proof of a clinical difference.",
            ],
            [
                "Matching policy",
                job.get("matching_policy", "Historical matcher; rerun with current policy"),
            ],
            ["Input source", job.get("input_origin", "")],
            ["Run ID", job["run_id"]],
            ["Created (UTC)", job["created_at"]],
            ["Updated (UTC)", job.get("updated_at", "")],
            ["Judge model", job.get("model") or "No model used"],
            ["Critic model", job.get("critic_model") or "No model used"],
            ["Runtime version", job.get("runtime_version", "")],
            ["Input SHA-256", job.get("input_sha256", "")],
        ],
        [32, 115],
    )
    records = {r["criterion_id"]: r for r in job["criteria_a"] + job["criteria_b"]}
    results = {(r["criterion_a_id"], r["criterion_b_id"]): r for r in job["results"]}
    pair_rows = []
    for candidate in job["candidates"]:
        a, b = records[candidate["a"]], records[candidate["b"]]
        result = results.get((candidate["a"], candidate["b"]), {})
        history = job["reviews"].get(result.get("pair_id"), [])
        latest = history[-1] if history else {}
        critic = result.get("critic", {})
        pair_rows.append(
            [
                a["entity"],
                a["source"],
                b["entity"],
                b["source"],
                readable(result.get("label")) or "No decision",
                "Not examined"
                if not result
                else "Needs review"
                if result["should_escalate"]
                else "Evidence-supported proposal",
                result.get("rationale", "No relationship evaluated before the run stopped."),
                result.get("common_frame", ""),
                "\n".join([critic.get("rationale", ""), *critic.get("issues", [])]).strip(),
                readable(latest.get("final_label")) or "Not reviewed",
                latest.get("rationale", ""),
                latest.get("reviewer_id", ""),
                a["criterion_type"],
                b["criterion_type"],
                a["criterion_id"],
                b["criterion_id"],
                result.get("pair_id", ""),
                "\n".join(f"[{e['side']}] {e['quote']}" for e in result.get("evidence", [])),
            ]
        )
    sheet(
        "Pair decisions",
        [
            "A concept",
            "A criterion source",
            "B concept",
            "B criterion source",
            "Proposed relationship",
            "Decision status",
            "Explanation",
            "Comparison frame",
            "Critic feedback",
            "Latest human label",
            "Human rationale",
            "Reviewer",
            "A eligibility role",
            "B eligibility role",
            "A criterion ID",
            "B criterion ID",
            "Pair ID",
            "Cited evidence",
        ],
        pair_rows,
        [25, 65, 25, 65, 27, 30, 80, 60, 70, 27, 65, 24, 22, 22, 36, 36, 32, 75],
    )
    coverage_rows = []
    for side in ["a", "b"]:
        for row in job["criteria_" + side]:
            coverage_rows.append(
                [
                    side.upper(),
                    row["trial_id"],
                    row["entity"],
                    row["source"],
                    "No counterpart assigned"
                    if row.get("alignment_status") == "no_candidate_found"
                    else readable(row.get("alignment_status")),
                    row.get("alignment_reason", ""),
                    row["criterion_type"],
                    row["criterion_id"],
                    row.get("parent_statement", ""),
                    row.get("source_uri", ""),
                ]
            )
    sheet(
        "Criterion coverage",
        [
            "Side",
            "Trial ID",
            "Concept",
            "Criterion source",
            "Coverage status",
            "Coverage explanation",
            "Eligibility role",
            "Criterion ID",
            "Parent statement",
            "Source URL",
        ],
        coverage_rows,
        [9, 30, 28, 80, 30, 70, 22, 40, 90, 55],
    )
    history_rows = []
    for pair_id, decisions in job["reviews"].items():
        for index, decision in enumerate(decisions, 1):
            history_rows.append(
                [
                    pair_id,
                    index,
                    readable(decision.get("final_label")),
                    decision.get("rationale", ""),
                    decision.get("reviewer_id", ""),
                    decision.get("created_at", ""),
                    decision.get("status", ""),
                ]
            )
    sheet(
        "Review history",
        [
            "Pair ID",
            "Version",
            "Human label",
            "Rationale",
            "Reviewer",
            "Reviewed (UTC)",
            "Review status",
        ],
        history_rows,
        [35, 12, 30, 90, 28, 30, 24],
    )
    stream = BytesIO()
    wb.save(stream)
    return stream.getvalue()
