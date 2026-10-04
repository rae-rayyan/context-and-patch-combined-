from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.services.static_analysis import AnalysisResult, Diagnostic, StaticAnalyzer
from app.services.patch_models import VerificationResult, VerificationStep


@dataclass(slots=True)
class VerificationSnapshot:
    diagnostics_by_file: dict[str, list[Diagnostic]]


class RepositoryVerifier:
    """Deterministic post-patch verification.

    No commands are accepted from the model. The verifier only runs a fixed,
    repository-local set of checks.
    """

    def __init__(
        self,
        repo_path: str | Path,
        timeout: int = 60,
        run_tests: bool = True,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.run_tests = run_tests
        self.analyzer = StaticAnalyzer(self.repo_path, timeout=timeout)

    def snapshot(self, files: Iterable[str]) -> VerificationSnapshot:
        diagnostics_by_file: dict[str, list[Diagnostic]] = {}
        for file_path in files:
            if Path(file_path).suffix != ".py":
                continue
            result = self.analyzer.analyze_file(file_path)
            diagnostics_by_file[file_path] = result.diagnostics
        return VerificationSnapshot(diagnostics_by_file)

    def verify(
        self,
        changed_files: Iterable[str],
        baseline: VerificationSnapshot,
    ) -> VerificationResult:
        changed = list(dict.fromkeys(changed_files))
        steps: list[VerificationStep] = []

        python_files = [item for item in changed if Path(item).suffix == ".py"]

        # 1) Python compilation is cheap and deterministic.
        if python_files:
            compile_output = self._run(
                [sys.executable, "-m", "py_compile", *python_files]
            )
            steps.append(
                VerificationStep(
                    name="Python syntax compilation",
                    passed=compile_output.returncode == 0,
                    output=compile_output.output,
                )
            )

        # 2) Targeted static analysis. Compare with the pre-patch baseline so
        # pre-existing findings do not block an otherwise clean repair.
        after_diagnostics: list[Diagnostic] = []
        new_keys: set[tuple[str, str, str | None, int | None, int | None]] = set()
        remaining_keys: set[tuple[str, str, str | None, int | None, int | None]] = set()

        for path in python_files:
            result = self.analyzer.analyze_file(path)
            after_diagnostics.extend(result.diagnostics)
            before = baseline.diagnostics_by_file.get(path, [])
            before_keys = {_diag_key(item) for item in before}
            after_keys = {_diag_key(item) for item in result.diagnostics}
            new_keys.update(after_keys - before_keys)
            remaining_keys.update(after_keys & before_keys)

        new_keys = set(new_keys)
        remaining_keys = set(remaining_keys)

        steps.append(
            VerificationStep(
                name="Targeted Ruff/Bandit analysis",
                passed=not new_keys,
                output=(
                    f"{len(after_diagnostics)} finding(s) after patch; "
                    f"{len(new_keys)} new finding(s)."
                ),
            )
        )

        # 3) Run repository tests only when pytest is present and the repo
        # contains tests. This avoids failing projects that simply do not use it.
        test_output = ""
        test_passed = True
        if self.run_tests and python_files and self._has_pytest_tests():
            if shutil.which("pytest"):
                result = self._run(["pytest", "-q"])
                test_output = result.output
                test_passed = result.returncode == 0
            else:
                test_output = "pytest not installed; skipped."

            steps.append(
                VerificationStep(
                    name="Repository tests",
                    passed=test_passed,
                    output=test_output,
                )
            )

        passed = all(item.passed for item in steps) and not new_keys
        return VerificationResult(
            passed=passed,
            steps=steps,
            new_diagnostics=[_format_key(item) for item in new_keys],
            remaining_diagnostics=[_format_key(item) for item in remaining_keys],
        )

    def _has_pytest_tests(self) -> bool:
        return any(
            path.name.startswith("test_") and path.suffix == ".py"
            for path in self.repo_path.rglob("*.py")
            if ".git" not in path.parts and "site-packages" not in path.parts
        )

    def _run(self, command: list[str]) -> "CommandResult":
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            output = "\n".join(
                part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
            )
            return CommandResult(completed.returncode, output)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(124, f"Command timed out after {self.timeout}s: {exc}")
        except OSError as exc:
            return CommandResult(126, str(exc))


@dataclass(slots=True)
class CommandResult:
    returncode: int
    output: str


def _diag_key(item: Diagnostic) -> tuple[str, str, str | None, int | None, int | None]:
    return (
        item.tool,
        item.message,
        item.code,
        item.line,
        item.column,
    )


def _format_key(item: tuple[str, str, str | None, int | None, int | None]) -> str:
    tool, message, code, line, column = item
    location = ""
    if line is not None:
        location = f":{line}"
        if column is not None:
            location += f":{column}"
    code_text = f" [{code}]" if code else ""
    return f"{tool}{location}{code_text}: {message}"
