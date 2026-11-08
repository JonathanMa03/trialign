"""Closed tool registry with before/after execution hooks and provenance."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict

from trialign.tools.structured import VERSION, compare_structured, frame_issues


class ToolRegistry:
    names = ("inspect_frame", "compare_constraints", "cite_sources")
    version = "comparison-tools-1"

    def execute(self, name, a, b, *, budget, emit):
        if name not in self.names:
            raise ValueError("Tool is not registered")
        budget.before_tool()
        emit("hook.before_tool", {"tool": name, "call": budget.tool_calls})
        started = time.monotonic()
        if name == "inspect_frame":
            result = {"a": frame_issues(a), "b": frame_issues(b)}
        elif name == "compare_constraints":
            result = asdict(compare_structured(a, b))
            result["label"] = result["label"].value if result["label"] else None
        else:
            result = {
                side: {
                    "criterion_id": row.criterion_id,
                    "trial_id": row.trial_id,
                    "source": row.source,
                    "parent_statement": row.parent_statement,
                    "source_uri": row.source_uri,
                    "source_version": row.source_version,
                    "source_sha256": hashlib.sha256(row.source.encode()).hexdigest(),
                }
                for side, row in [("A", a), ("B", b)]
            }
        emit(
            "tool.completed",
            {
                "tool": name,
                "version": VERSION,
                "result": result,
                "seconds": round(time.monotonic() - started, 4),
            },
        )
        emit("hook.after_tool", {"tool": name, "status": "validated"})
        return result
