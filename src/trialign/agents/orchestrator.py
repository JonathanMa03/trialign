"""Bounded planner -> tools -> judge -> critic -> repair/escalation workflow."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from uuid import uuid4

from trialign.agents.contracts import Critique, Evidence, Judgment, ModelPrediction, PairResult
from trialign.agents.hooks import BudgetExceeded, RuntimeBudget, SourceNeedsReview, validate_inputs
from trialign.agents.roles import CriticAgent, JudgeAgent, PlannerAgent
from trialign.schemas import RelationshipLabel
from trialign.tools.correspondence import NoClinicalCounterpart, correspondence
from trialign.tools.registry import ToolRegistry
from trialign.tools.source_quality import assess_source

VERSION = "agent-runtime-8"
AgentResult = PairResult


class AgentOrchestrator:
    def __init__(self, provider=None, *, enable_critic=True, relationship_model=None):
        self.provider = provider
        self.enable_critic = enable_critic
        self.planner = PlannerAgent()
        self.tools = ToolRegistry()
        self.judge = JudgeAgent()
        self.critic = CriticAgent()
        self.relationship_model = relationship_model

    def compare(self, a, b, *, budget=None, on_event=None, run_id=None):
        budget = budget or RuntimeBudget()
        started = time.monotonic()
        pair_id = (
            "pair-"
            + hashlib.sha256(f"{a.criterion_id}\x1f{b.criterion_id}".encode()).hexdigest()[:20]
        )
        trace_id = "trace-" + uuid4().hex[:20]
        trace = []

        def emit(stage, payload):
            event = {
                "sequence": len(trace) + 1,
                "stage": stage,
                "data": payload,
                "pair_id": pair_id,
                "trace_id": trace_id,
                "run_id": run_id,
                "time": datetime.now(UTC).isoformat(),
            }
            trace.append(event)
            if on_event:
                on_event(event)

        validate_inputs(a, b)
        emit(
            "hook.input_validation",
            {
                "status": "passed",
                "version": VERSION,
                "criterion_ids": [a.criterion_id, b.criterion_id],
            },
        )
        skill = self.planner.plan(a, b)
        emit("agent.planner", skill)
        proposal = None
        route = "context_review"
        critique = Critique(verdict="escalate", issues=[], rationale="Not yet reviewed.")
        tools = {}
        critic_assessment = {}
        disposition = "needs_review"
        model_prediction = None
        try:
            compatible, reason = correspondence(a, b)
            emit("hook.clinical_correspondence", {"allowed": compatible, "reason": reason})
            if not compatible:
                raise NoClinicalCounterpart(reason)
            source_assessments = {"A": assess_source(a), "B": assess_source(b)}
            emit(
                "hook.source_scope",
                {side: result.as_dict() for side, result in source_assessments.items()},
            )
            blockers = [
                f"{side}: {issue}"
                for side, result in source_assessments.items()
                for issue in result.blockers
            ]
            if blockers:
                raise SourceNeedsReview("; ".join(blockers))
            for tool in skill["tools"]:
                tools[tool] = self.tools.execute(tool, a, b, budget=budget, emit=emit)
            source_issues = list(getattr(a, "source_quality_issues", []) or [])
            source_issues += list(getattr(b, "source_quality_issues", []) or [])
            if source_issues:
                raise SourceNeedsReview("; ".join(source_issues))
            exact = tools.get("compare_constraints", {})
            warnings = any(result.warnings for result in source_assessments.values())
            if exact.get("label") and not exact.get("requires_context") and not warnings:
                route = "exact_tools"
                proposal = Judgment(
                    label=exact["label"],
                    rationale=" ".join(exact["reasons"]),
                    common_frame=(
                        f"Selected {a.entity} predicates with source-supported roles and units."
                    ),
                    uncertainty="low",
                    assumptions=["No whole-trial population claim."],
                    evidence=[
                        Evidence(side="A", quote=a.source[:400]),
                        Evidence(side="B", quote=b.source[:400]),
                    ],
                )
                critique = self.critic.critique(
                    proposal, a, b, provider=None, budget=budget, emit=emit, exact=True
                )
                emit("agent.critic.rule_audit", critique.model_dump())
            else:
                if self.relationship_model is not None:
                    budget.before_tool()
                    emit("hook.before_tool", {"tool": "predict_relationship"})
                    model_prediction = ModelPrediction.model_validate(
                        self.relationship_model.predict_pair(a, b)
                    ).model_dump(mode="json")
                    budget.check()
                    tools["predict_relationship"] = model_prediction
                    emit(
                        "tool.completed",
                        {"tool": "predict_relationship", "result": model_prediction},
                    )
                    emit(
                        "hook.after_tool",
                        {"tool": "predict_relationship", "status": "proposal_only"},
                    )
                if self.provider is None:
                    route = "trained_model_review" if model_prediction else "tools_only"
                    raise BudgetExceeded(
                        "Contextual review unavailable; any trained prediction "
                        "is a proposal requiring review."
                    )
                if budget.remaining_model_calls < (3 if self.enable_critic else 1):
                    raise BudgetExceeded(
                        "Insufficient call budget for judge, independent assessment "
                        "and critic review."
                    )
                if model_prediction:
                    route = "model_assisted_context_review"
                proposal = self.judge.judge(
                    a, b, tools, skill, provider=self.provider, budget=budget, emit=emit
                )
                if self.enable_critic:
                    critique = self.critic.critique(
                        proposal,
                        a,
                        b,
                        provider=self.provider,
                        budget=budget,
                        emit=emit,
                        assessment_cache=critic_assessment,
                    )
                    if critique.verdict == "revise" and budget.remaining_model_calls >= 2:
                        emit("orchestrator.revision", {"attempt": 1, "issues": critique.issues})
                        proposal = self.judge.judge(
                            a,
                            b,
                            tools,
                            skill,
                            provider=self.provider,
                            budget=budget,
                            emit=emit,
                            feedback=critique.model_dump(),
                        )
                        critique = self.critic.critique(
                            proposal,
                            a,
                            b,
                            provider=self.provider,
                            budget=budget,
                            emit=emit,
                            assessment_cache=critic_assessment,
                        )
                else:
                    # Harness-only ablation: still enforce the source-grounding hook.
                    from trialign.agents.hooks import validate_evidence

                    issues = validate_evidence(proposal, a, b)
                    critique = Critique(
                        verdict="escalate" if issues else "accept",
                        issues=issues,
                        rationale=(
                            "Harness ablation: LLM critic disabled; evidence hooks remain active."
                        ),
                    )
        except NoClinicalCounterpart as exc:
            disposition = "no_counterpart"
            route = "correspondence_gate"
            critique = Critique(
                verdict="escalate",
                issues=[str(exc)],
                rationale="No relationship evaluated: no supported common clinical concept.",
            )
        except SourceNeedsReview as exc:
            disposition = "source_review"
            emit("hook.source_validation", {"issues": [str(exc)]})
            critique = Critique(
                verdict="escalate",
                issues=[str(exc)],
                rationale="Source correction/clarification is required.",
            )
        except BudgetExceeded as exc:
            disposition = "not_evaluated"
            emit("hook.budget_stop", {"reason": str(exc)})
            critique = Critique(verdict="escalate", issues=[str(exc)], rationale=str(exc))
        except Exception as exc:
            disposition = "operational_failure"
            # Never persist SDK request headers, credentials or raw exception bodies.
            emit("orchestrator.failure", {"error_type": type(exc).__name__})
            critique = Critique(
                verdict="escalate",
                issues=[f"Context review failed ({type(exc).__name__})."],
                rationale="The provider/tool step did not produce a validated result.",
            )

        accepted = proposal is not None and critique.verdict == "accept"
        if accepted:
            disposition = "tool_supported" if route == "exact_tools" else "critic_accepted"
            label = proposal.label
            rationale = proposal.rationale
            frame = proposal.common_frame
            evidence = proposal.evidence
            uncertainty = proposal.uncertainty
            assumptions = proposal.assumptions
        else:
            label = RelationshipLabel.AMBIGUOUS if disposition == "source_review" else None
            rationale = critique.rationale + " " + "; ".join(critique.issues)
            frame = f"Selected A/B criteria concerning {a.entity} and {b.entity}."
            evidence = [
                Evidence(side="A", quote=a.source[:400]),
                Evidence(side="B", quote=b.source[:400]),
            ]
            uncertainty, assumptions = "high", []
        escalates = (
            not accepted
            or label in {RelationshipLabel.AMBIGUOUS, RelationshipLabel.RELATED_NOT_COMPARABLE}
            or any(r.review_status not in {"accepted", "corrected"} for r in [a, b])
        )
        emit(
            "hook.before_finalize",
            {
                "accepted_by_critic": accepted,
                "should_escalate": escalates,
                "label": label.value if label else None,
                "disposition": disposition,
            },
        )
        return PairResult(
            pair_id=pair_id,
            trace_id=trace_id,
            criterion_a_id=a.criterion_id,
            criterion_b_id=b.criterion_id,
            label=label,
            rationale=rationale.strip(),
            common_frame=frame,
            uncertainty=uncertainty,
            assumptions=assumptions,
            evidence=evidence,
            route=route,
            disposition=disposition,
            should_escalate=escalates,
            critic=critique,
            trace=trace,
            elapsed_seconds=round(time.monotonic() - started, 3),
            confidence=1.0 if route == "exact_tools" and accepted else 0.0,
            model_prediction=model_prediction,
        )
