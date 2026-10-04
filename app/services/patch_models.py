from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChangeType(str, Enum):
    REPLACE = "replace"
    CREATE = "create"


@dataclass(slots=True)
class FileChange:
    """One safe source-code edit proposed by the Patch Agent."""

    path: str
    old_text: str
    new_text: str
    change_type: ChangeType = ChangeType.REPLACE
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileChange":
        raw_type = str(data.get("change_type", ChangeType.REPLACE.value)).lower()
        try:
            change_type = ChangeType(raw_type)
        except ValueError as exc:
            raise ValueError(f"Unsupported change_type: {raw_type}") from exc

        path = data.get("path")
        old_text = data.get("old_text", "")
        new_text = data.get("new_text", "")

        if not isinstance(path, str) or not path.strip():
            raise ValueError("Each change needs a non-empty path.")
        if not isinstance(old_text, str):
            raise ValueError("old_text must be a string.")
        if not isinstance(new_text, str):
            raise ValueError("new_text must be a string.")

        if change_type is ChangeType.CREATE and old_text:
            raise ValueError("CREATE changes must use an empty old_text.")

        return cls(
            path=path,
            old_text=old_text,
            new_text=new_text,
            change_type=change_type,
            reason=str(data.get("reason", "")),
        )


@dataclass(slots=True)
class PatchPlan:
    """Structured repair plan returned by an LLM."""

    summary: str
    changes: list[FileChange] = field(default_factory=list)
    tests_to_focus: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatchPlan":
        raw_changes = data.get("changes", [])
        if not isinstance(raw_changes, list):
            raise ValueError("changes must be a list.")

        changes = [FileChange.from_dict(item) for item in raw_changes if isinstance(item, dict)]
        if not changes:
            raise ValueError("Patch plan contains no valid changes.")

        raw_tests = data.get("tests_to_focus", [])
        tests = [str(item) for item in raw_tests] if isinstance(raw_tests, list) else []

        return cls(
            summary=str(data.get("summary", "")),
            changes=changes,
            tests_to_focus=tests,
        )


@dataclass(slots=True)
class AppliedFile:
    path: str
    change_type: ChangeType
    before: str | None
    after: str


@dataclass(slots=True)
class VerificationStep:
    name: str
    passed: bool
    output: str = ""


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    steps: list[VerificationStep] = field(default_factory=list)
    new_diagnostics: list[str] = field(default_factory=list)
    remaining_diagnostics: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Verification: {status}"]
        for step in self.steps:
            marker = "PASS" if step.passed else "FAIL"
            lines.append(f"[{marker}] {step.name}")
            if step.output:
                lines.append(step.output.strip())
        if self.new_diagnostics:
            lines.append("New diagnostics:")
            lines.extend(f"- {item}" for item in self.new_diagnostics)
        if self.remaining_diagnostics:
            lines.append("Remaining diagnostics:")
            lines.extend(f"- {item}" for item in self.remaining_diagnostics)
        return "\n".join(lines)


@dataclass(slots=True)
class PatchRunResult:
    success: bool
    summary: str
    diff: str = ""
    attempts: int = 0
    verification: VerificationResult | None = None
    errors: list[str] = field(default_factory=list)
    plan: PatchPlan | None = None
