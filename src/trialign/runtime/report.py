"""Readable comparison exports generated from the immutable saved run snapshot."""

from __future__ import annotations

import json
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

    settings = job.get("provider_settings", {})
    metrics = job.get("metrics", {})
    cost = job.get("cost_estimate", {})
    metadata_rows = [
        ["Run note", job.get("note") or "None"],
        ["Display status", job.get("display_status") or job["status"]],
        ["Run filename", job.get("file_name", "Not recorded")],
        ["Resolved API models", ", ".join(job.get("resolved_models", [])) or "None reported"],
        ["Request timeout (seconds)", settings.get("request_timeout_seconds", "Not recorded")],
        ["Maximum output tokens per call", settings.get("max_output_tokens", "Not recorded")],
        ["Configured judge model", settings.get("judge_model", "Not recorded")],
        ["Configured critic model", settings.get("critic_model", "Not recorded")],
        ["Retrieval model", settings.get("retrieval_model", job.get("encoder", "Not recorded"))],
        [
            "Encoder configuration and revision",
            json.dumps(job.get("encoder_metadata", {}), sort_keys=True),
        ],
        [
            "Relationship artifact",
            json.dumps(settings.get("relationship_artifact"), sort_keys=True),
        ],
        ["Contextual models enabled", job.get("use_llm")],
        [
            "Maximum model calls",
            job.get("max_model_calls") if job.get("max_model_calls") is not None else "Unlimited",
        ],
        ["Time budget (seconds)", job.get("max_seconds")],
        ["Candidate-pair limit", job.get("requested_pair_limit") or "All assigned counterparts"],
        ["Total elapsed (seconds)", metrics.get("elapsed_seconds", "Not recorded")],
        ["Retrieval (seconds)", metrics.get("retrieval_seconds", "Not recorded")],
        ["Comparison execution (seconds)", metrics.get("execution_seconds", "Not recorded")],
        ["Local worker CPU (seconds)", metrics.get("worker_thread_cpu_seconds", "Not recorded")],
        ["API requests (seconds)", metrics.get("api_request_seconds", "Not recorded")],
        ["Input tokens", job.get("usage", {}).get("input_tokens", "Not recorded")],
        ["Output tokens", job.get("usage", {}).get("output_tokens", "Not recorded")],
        [
            "Estimated API cost (USD)",
            cost.get("estimated_api_cost_usd")
            if cost.get("estimated_api_cost_usd") is not None
            else "Unavailable",
        ],
        ["Known API subtotal (USD)", cost.get("known_api_cost_usd", "Not recorded")],
        ["Cost status", cost.get("status", "Not recorded")],
        ["Cost limitations", " ".join(cost.get("reasons", []))],
        ["Cost scope", cost.get("scope", "No cost metadata recorded for this historical run")],
        ["Local compute", metrics.get("local_device", "Not recorded")],
        [
            "Platform / architecture",
            f"{metrics.get('platform', '')} {metrics.get('machine', '')}".strip() or "Not recorded",
        ],
        ["Python version", metrics.get("python_version", "Not recorded")],
    ]
    for agent, rates in settings.get("pricing", {}).get("rates", {}).items():
        for field, value in rates.items():
            metadata_rows.append(
                [f"{agent} {field} (USD)", value if value is not None else "Unknown"]
            )
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
        ]
        + metadata_rows,
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
                (
                    readable((result.get("model_prediction") or {}).get("proposed_label"))
                    or "Abstained"
                )
                if result.get("model_prediction")
                else "Not run",
                (result.get("model_prediction") or {}).get("relevance_score"),
                (result.get("model_prediction") or {}).get("artifact_sha256", ""),
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
            "Learned proposal (uncalibrated)",
            "Learned relevance score",
            "Learned artifact SHA-256",
        ],
        pair_rows,
        [25, 65, 25, 65, 27, 30, 80, 60, 70, 27, 65, 24, 22, 22, 36, 36, 32, 75, 40, 24, 75],
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
    usage_rows = []
    for entry in job.get("usage_ledger", []):
        usage = entry.get("usage") or {}
        usage_rows.append(
            [
                entry["agent"],
                entry["role"],
                entry["requested_model"],
                entry.get("resolved_model"),
                usage.get("input_tokens"),
                usage.get("cached_input_tokens"),
                usage.get("output_tokens"),
                usage.get("cache_write_tokens"),
                entry["elapsed_seconds"],
                entry.get("response_status"),
                entry.get("error_type"),
            ]
        )
    sheet(
        "API usage",
        [
            "Agent",
            "Request role",
            "Requested model",
            "Resolved model",
            "Input tokens",
            "Cached input tokens",
            "Output tokens",
            "Cache-write tokens",
            "API seconds",
            "Response status",
            "Error type",
        ],
        usage_rows,
        [15, 35, 35, 35, 18, 22, 18, 22, 18, 22, 30],
    )
    stream = BytesIO()
    wb.save(stream)
    return stream.getvalue()
