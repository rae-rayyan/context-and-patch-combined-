from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.services.context import ContextPacket
from app.services.llm import PatchLLM
from app.services.patch_models import (
    AppliedFile,
    ChangeType,
    PatchPlan,
    PatchRunResult,
    VerificationResult,
)
from app.services.verification import RepositoryVerifier


SYSTEM_CONSTRAINTS = """
You repair a software repository from a compact Context Packet.

Rules:
1. Only edit files present in the supplied context unless the issue clearly
   requires a new test/supporting file.
2. Prefer the smallest correct change.
3. NEVER output shell commands to execute.
4. For replace operations, old_text MUST be copied exactly from the supplied
   source, including indentation and blank lines.
5. Do not rewrite an entire file when a local change is sufficient.
6. Preserve existing public APIs unless the bug explicitly requires a change.
7. Add or update a focused regression test when practical.
8. Return ONLY JSON with this shape:
{
  "summary": "one sentence",
  "changes": [
    {
      "path": "relative/path.py",
      "change_type": "replace" | "create",
      "old_text": "exact existing text, empty for create",
      "new_text": "replacement/new file content",
      "reason": "why this change fixes the issue"
    }
  ],
  "tests_to_focus": ["relative/test/path.py"]
}
""".strip()


@dataclass(slots=True)
class PatchAgent:
    """Generate, apply, verify, and rollback repository patches."""

    repo_path: str | Path
    llm: PatchLLM
    verifier: RepositoryVerifier | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path).resolve()
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Repository does not exist: {self.repo_path}")
        self.verifier = self.verifier or RepositoryVerifier(self.repo_path)

    def run(self, context: ContextPacket) -> PatchRunResult:
        errors: list[str] = []
        last_verification: VerificationResult | None = None
        last_plan: PatchPlan | None = None

        feedback = ""

        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_prompt(context, feedback)

            try:
                raw = self.llm.generate_patch(prompt)
                plan = PatchPlan.from_dict(raw)
                last_plan = plan
            except Exception as exc:  # model/schema errors are recoverable
                errors.append(f"Attempt {attempt}: patch generation failed: {exc}")
                feedback = f"Previous attempt failed before applying a patch: {exc}"
                continue

            try:
                baseline = self.verifier.snapshot(
                    [change.path for change in plan.changes]
                )
                applied, diff = self._apply_plan(plan)
            except Exception as exc:
                errors.append(f"Attempt {attempt}: patch application failed: {exc}")
                feedback = (
                    "Patch application failed. Fix the plan and return JSON again.\n"
                    f"Application error: {exc}"
                )
                continue

            verification = self.verifier.verify(
                changed_files=[item.path for item in applied],
                baseline=baseline,
            )
            last_verification = verification

            if verification.passed:
                return PatchRunResult(
                    success=True,
                    summary=plan.summary or "Patch generated and verified.",
                    diff=diff,
                    attempts=attempt,
                    verification=verification,
                    errors=errors,
                    plan=plan,
                )

            self._rollback(applied)
            feedback = (
                "The patch was applied but verification failed. Return a corrected"
                " patch using the exact current source from the original context.\n"
                f"\n{verification.summary()}"
            )

        return PatchRunResult(
            success=False,
            summary="Patch Agent could not produce a verified patch.",
            diff="",
            attempts=self.max_attempts,
            verification=last_verification,
            errors=errors,
            plan=last_plan,
        )

    def _build_prompt(self, context: ContextPacket, feedback: str) -> str:
        packet = context.render()
        sections = [SYSTEM_CONSTRAINTS, "", packet]
        if feedback:
            sections.extend(["", "## Previous Attempt Feedback", feedback])
        sections.extend(
            [
                "",
                "Produce the smallest safe repair now. "
                "Remember that old_text must match the repository exactly.",
            ]
        )
        return "\n".join(sections)

    def _apply_plan(self, plan: PatchPlan) -> tuple[list[AppliedFile], str]:
        if not plan.changes:
            raise ValueError("Patch plan has no changes.")

        applied: list[AppliedFile] = []
        before_after: list[tuple[str, str, str]] = []

        try:
            for change in plan.changes:
                path = self._safe_path(change.path)

                if change.change_type is ChangeType.CREATE:
                    if path.exists():
                        raise ValueError(f"Cannot create existing file: {change.path}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(change.new_text, encoding="utf-8")
                    applied.append(
                        AppliedFile(
                            path=change.path,
                            change_type=change.change_type,
                            before=None,
                            after=change.new_text,
                        )
                    )
                    before_after.append((change.path, "", change.new_text))
                    continue

                if not path.exists() or not path.is_file():
                    raise ValueError(f"Target file does not exist: {change.path}")
                if path.is_symlink():
                    raise ValueError(f"Refusing to modify symlink: {change.path}")

                before = path.read_text(encoding="utf-8")
                occurrences = before.count(change.old_text)
                if occurrences != 1:
                    raise ValueError(
                        f"old_text in {change.path} matched {occurrences} times; expected exactly 1."
                    )

                after = before.replace(change.old_text, change.new_text, 1)
                if after == before:
                    raise ValueError(f"Change for {change.path} produces no modification.")

                path.write_text(after, encoding="utf-8")
                applied.append(
                    AppliedFile(
                        path=change.path,
                        change_type=change.change_type,
                        before=before,
                        after=after,
                    )
                )
                before_after.append((change.path, before, after))

            return applied, _build_diff(before_after)

        except Exception:
            self._rollback(applied)
            raise

    def _rollback(self, applied: list[AppliedFile]) -> None:
        for item in reversed(applied):
            path = self._safe_path(item.path)
            if item.change_type is ChangeType.CREATE:
                if path.exists() and path.is_file():
                    path.unlink()
                continue
            if item.before is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item.before, encoding="utf-8")

    def _safe_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"Absolute paths are forbidden: {relative}")

        resolved = (self.repo_path / candidate).resolve()
        try:
            resolved.relative_to(self.repo_path)
        except ValueError as exc:
            raise ValueError(f"Path escapes repository: {relative}") from exc

        return resolved


def _build_diff(items: list[tuple[str, str, str]]) -> str:
    chunks: list[str] = []
    for path, before, after in items:
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)
