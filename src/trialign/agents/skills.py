"""Versioned executable skill selection; instructions are runtime inputs."""

from __future__ import annotations

import hashlib
from importlib.resources import files

import yaml

from trialign.tools.structured import frame_issues


class SkillRegistry:
    def __init__(self):
        text = files("trialign.agents").joinpath("skills.yaml").read_text()
        payload = yaml.safe_load(text)
        self.version = payload["version"]
        self.digest = hashlib.sha256(text.encode()).hexdigest()
        self.skills = payload["skills"]

    def select(self, a, b) -> dict:
        flags = frame_issues(a) + frame_issues(b)
        if getattr(a, "source_quality_issues", []) or getattr(b, "source_quality_issues", []):
            key = "source_resolution"
        elif flags:
            key = "clinical_context"
        elif a.comparator and b.comparator:
            key = "numeric"
        else:
            key = "clinical_context"
        return {
            "id": key,
            "version": self.version,
            "sha256": self.digest,
            **self.skills[key],
            "route_reasons": flags,
        }
