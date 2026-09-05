from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# Data models
# ============================================================


@dataclass
class Diagnostic:
    """
    One static-analysis finding.

    This is intentionally tool-independent so that the rest
    of Varmintor does not need to know whether a finding came
    from Ruff or Bandit.
    """

    tool: str
    message: str
    file_path: str | None = None
    line: int | None = None
    column: int | None = None

    code: str | None = None
    severity: str | None = None

    category: str | None = None

    def render(self) -> str:
        """Render one diagnostic as readable text."""

        location = ""

        if self.file_path:
            location = self.file_path

            if self.line is not None:
                location += f":{self.line}"

                if self.column is not None:
                    location += f":{self.column}"

        parts = [self.tool]

        if location:
            parts.append(location)

        if self.code:
            parts.append(f"[{self.code}]")

        if self.severity:
            parts.append(
                f"severity={self.severity}"
            )

        if self.category:
            parts.append(
                f"category={self.category}"
            )

        prefix = " ".join(parts)

        return f"{prefix}: {self.message}"


@dataclass
class AnalysisResult:
    """
    Complete result of Ruff + Bandit analysis.
    """

    diagnostics: list[Diagnostic] = field(
        default_factory=list
    )

    ruff_available: bool = True
    bandit_available: bool = True

    ruff_error: str | None = None
    bandit_error: str | None = None

    def has_findings(self) -> bool:
        return bool(self.diagnostics)

    def render(self) -> str:
        """
        Convert analysis results into compact text suitable
        for ContextPacket.diagnostics.
        """

        sections: list[str] = []

        # ----------------------------------------------------
        # Ruff
        # ----------------------------------------------------

        ruff_findings = [
            item
            for item in self.diagnostics
            if item.tool == "ruff"
        ]

        if ruff_findings:
            sections.append(
                "## Ruff Findings\n"
                + "\n".join(
                    item.render()
                    for item in ruff_findings
                )
            )
        elif self.ruff_error:
            sections.append(
                "## Ruff\n"
                f"Ruff analysis error: "
                f"{self.ruff_error}"
            )
        elif self.ruff_available:
            sections.append(
                "## Ruff Findings\n"
                "No Ruff findings."
            )
        else:
            sections.append(
                "## Ruff\n"
                "Ruff is not available."
            )

        # ----------------------------------------------------
        # Bandit
        # ----------------------------------------------------

        bandit_findings = [
            item
            for item in self.diagnostics
            if item.tool == "bandit"
        ]

        if bandit_findings:
            sections.append(
                "## Bandit Findings\n"
                + "\n".join(
                    item.render()
                    for item in bandit_findings
                )
            )
        elif self.bandit_error:
            sections.append(
                "## Bandit\n"
                f"Bandit analysis error: "
                f"{self.bandit_error}"
            )
        elif self.bandit_available:
            sections.append(
                "## Bandit Findings\n"
                "No Bandit findings."
            )
        else:
            sections.append(
                "## Bandit\n"
                "Bandit is not available."
            )

        return "\n\n".join(sections)


# ============================================================
# Static analyzer
# ============================================================


class StaticAnalyzer:
    """
    Runs deterministic static analysis before the Patch Agent.

    Tools:
        Ruff  -> linting / code-quality diagnostics
        Bandit -> Python security diagnostics

    This class does NOT modify source code.
    """

    def __init__(
        self,
        repo_path: str | Path,
        timeout: int = 60,
    ) -> None:

        self.repo_path = Path(
            repo_path
        ).resolve()

        self.timeout = timeout

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Repository does not exist: "
                f"{self.repo_path}"
            )

        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: "
                f"{self.repo_path}"
            )

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        run_ruff: bool = True,
        run_bandit: bool = True,
    ) -> AnalysisResult:
        """
        Run all requested static analyzers.

        The method intentionally continues if one analyzer
        fails or is unavailable.
        """

        result = AnalysisResult()

        if run_ruff:
            ruff_findings, error, available = (
                self.run_ruff()
            )

            result.diagnostics.extend(
                ruff_findings
            )

            result.ruff_error = error
            result.ruff_available = available

        if run_bandit:
            bandit_findings, error, available = (
                self.run_bandit()
            )

            result.diagnostics.extend(
                bandit_findings
            )

            result.bandit_error = error
            result.bandit_available = available

        return result

    # ========================================================
    # Ruff
    # ========================================================

    def run_ruff(
        self,
    ) -> tuple[
        list[Diagnostic],
        str | None,
        bool,
    ]:
        """
        Run Ruff using JSON output.

        Returns:

            diagnostics,
            error,
            executable_available
        """

        command = [
            "ruff",
            "check",
            str(self.repo_path),
            "--output-format",
            "json",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

        except FileNotFoundError:
            return (
                [],
                None,
                False,
            )

        except subprocess.TimeoutExpired:
            return (
                [],
                (
                    f"Ruff timed out after "
                    f"{self.timeout} seconds."
                ),
                True,
            )

        except OSError as exc:
            return (
                [],
                f"Could not execute Ruff: {exc}",
                True,
            )

        # ----------------------------------------------------
        # Ruff exit codes:
        #
        # 0 = no violations
        # 1 = violations found
        # 2 = error
        #
        # We therefore parse JSON even when returncode == 1.
        # ----------------------------------------------------

        if completed.returncode == 0:
            return (
                [],
                None,
                True,
            )

        findings = self._parse_ruff(
            completed.stdout
        )

        if completed.returncode == 1:
            return (
                findings,
                None,
                True,
            )

        error = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"Ruff exited with code "
            f"{completed.returncode}."
        )

        return (
            findings,
            error,
            True,
        )

    def _parse_ruff(
        self,
        output: str,
    ) -> list[Diagnostic]:
        """Convert Ruff JSON output into Diagnostic objects."""

        if not output.strip():
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        diagnostics: list[Diagnostic] = []

        for item in data:

            if not isinstance(item, dict):
                continue

            location = item.get(
                "location",
                {},
            )

            if not isinstance(location, dict):
                location = {}

            diagnostics.append(
                Diagnostic(
                    tool="ruff",
                    message=str(
                        item.get(
                            "message",
                            "Ruff violation.",
                        )
                    ),
                    file_path=self._relative_path(
                        item.get("filename")
                    ),
                    line=self._safe_int(
                        location.get("row")
                    ),
                    column=self._safe_int(
                        location.get("column")
                    ),
                    code=self._extract_ruff_code(
                        item
                    ),
                    severity="error",
                    category="lint",
                )
            )

        return diagnostics

    @staticmethod
    def _extract_ruff_code(
        item: dict[str, Any],
    ) -> str | None:
        """
        Ruff normally stores the rule code in `code`.

        Keep this separate because the JSON schema can evolve.
        """

        code = item.get("code")

        if code is None:
            return None

        return str(code)

    # ========================================================
    # Bandit
    # ========================================================

    def run_bandit(
        self,
    ) -> tuple[
        list[Diagnostic],
        str | None,
        bool,
    ]:
        """
        Run Bandit recursively over the repository.

        Bandit uses exit code 1 when findings exist, so,
        just like Ruff, that is NOT treated as execution failure.
        """

        command = [
            "bandit",
            "-r",
            str(self.repo_path),
            "-f",
            "json",
            "-q",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

        except FileNotFoundError:
            return (
                [],
                None,
                False,
            )

        except subprocess.TimeoutExpired:
            return (
                [],
                (
                    f"Bandit timed out after "
                    f"{self.timeout} seconds."
                ),
                True,
            )

        except OSError as exc:
            return (
                [],
                f"Could not execute Bandit: {exc}",
                True,
            )

        findings = self._parse_bandit(
            completed.stdout
        )

        # ----------------------------------------------------
        # Bandit normally:
        #
        # 0 = no findings
        # 1 = findings
        #
        # Other codes indicate an execution/configuration
        # problem.
        # ----------------------------------------------------

        if completed.returncode in {
            0,
            1,
        }:
            return (
                findings,
                None,
                True,
            )

        error = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"Bandit exited with code "
            f"{completed.returncode}."
        )

        return (
            findings,
            error,
            True,
        )

    def _parse_bandit(
        self,
        output: str,
    ) -> list[Diagnostic]:
        """Convert Bandit JSON output into Diagnostic objects."""

        if not output.strip():
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []

        results = data.get(
            "results",
            [],
        )

        if not isinstance(results, list):
            return []

        diagnostics: list[Diagnostic] = []

        for item in results:

            if not isinstance(item, dict):
                continue

            diagnostics.append(
                Diagnostic(
                    tool="bandit",
                    message=str(
                        item.get(
                            "issue_text",
                            "Bandit security finding.",
                        )
                    ),
                    file_path=self._relative_path(
                        item.get("filename")
                    ),
                    line=self._safe_int(
                        item.get("line_number")
                    ),
                    column=self._safe_int(
                        item.get("col_offset")
                    ),
                    code=self._safe_string(
                        item.get(
                            "test_id"
                        )
                    ),
                    severity=self._safe_string(
                        item.get(
                            "issue_severity"
                        )
                    ),
                    category=self._safe_string(
                        item.get(
                            "issue_confidence"
                        )
                    ),
                )
            )

        return diagnostics

    # ========================================================
    # Targeted analysis
    # ========================================================

    def analyze_file(
        self,
        file_path: str | Path,
    ) -> AnalysisResult:
        """
        Analyze a single Python file.

        Useful after the Context Agent identifies a target file.

        This avoids repeatedly analyzing the entire repository
        during future Patch → QA iterations.
        """

        path = self._safe_repo_path(
            file_path
        )

        if path is None:
            raise ValueError(
                f"File is outside repository or "
                f"does not exist: {file_path}"
            )

        result = AnalysisResult()

        # ----------------------------------------------------
        # Ruff
        # ----------------------------------------------------

        ruff_findings, ruff_error, available = (
            self._run_ruff_file(path)
        )

        result.diagnostics.extend(
            ruff_findings
        )

        result.ruff_error = ruff_error
        result.ruff_available = available

        # ----------------------------------------------------
        # Bandit
        # ----------------------------------------------------

        bandit_findings, bandit_error, available = (
            self._run_bandit_file(path)
        )

        result.diagnostics.extend(
            bandit_findings
        )

        result.bandit_error = bandit_error
        result.bandit_available = available

        return result

    def _run_ruff_file(
        self,
        path: Path,
    ) -> tuple[
        list[Diagnostic],
        str | None,
        bool,
    ]:

        command = [
            "ruff",
            "check",
            str(path),
            "--output-format",
            "json",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

        except FileNotFoundError:
            return [], None, False

        except subprocess.TimeoutExpired:
            return (
                [],
                (
                    f"Ruff timed out after "
                    f"{self.timeout} seconds."
                ),
                True,
            )

        except OSError as exc:
            return (
                [],
                f"Could not execute Ruff: {exc}",
                True,
            )

        findings = self._parse_ruff(
            completed.stdout
        )

        if completed.returncode in {
            0,
            1,
        }:
            return findings, None, True

        return (
            findings,
            (
                completed.stderr.strip()
                or f"Ruff exited with code "
                f"{completed.returncode}."
            ),
            True,
        )

    def _run_bandit_file(
        self,
        path: Path,
    ) -> tuple[
        list[Diagnostic],
        str | None,
        bool,
    ]:

        command = [
            "bandit",
            str(path),
            "-f",
            "json",
            "-q",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )

        except FileNotFoundError:
            return [], None, False

        except subprocess.TimeoutExpired:
            return (
                [],
                (
                    f"Bandit timed out after "
                    f"{self.timeout} seconds."
                ),
                True,
            )

        except OSError as exc:
            return (
                [],
                f"Could not execute Bandit: {exc}",
                True,
            )

        findings = self._parse_bandit(
            completed.stdout
        )

        if completed.returncode in {
            0,
            1,
        }:
            return findings, None, True

        return (
            findings,
            (
                completed.stderr.strip()
                or f"Bandit exited with code "
                f"{completed.returncode}."
            ),
            True,
        )

    # ========================================================
    # Security helpers
    # ========================================================

    def _safe_repo_path(
        self,
        file_path: str | Path,
    ) -> Path | None:
        """
        Prevent analysis from operating outside the cloned repo.
        """

        candidate = Path(
            file_path
        )

        if not candidate.is_absolute():
            candidate = (
                self.repo_path / candidate
            )

        try:
            candidate = candidate.resolve()
        except OSError:
            return None

        try:
            candidate.relative_to(
                self.repo_path
            )
        except ValueError:
            return None

        if not candidate.is_file():
            return None

        return candidate

    def _relative_path(
        self,
        file_path: Any,
    ) -> str | None:
        """
        Convert analyzer paths into repository-relative paths.

        This also prevents external absolute paths from being
        propagated into the Context Agent.
        """

        if file_path is None:
            return None

        candidate = Path(
            str(file_path)
        )

        if not candidate.is_absolute():
            candidate = (
                self.repo_path / candidate
            )

        try:
            candidate = candidate.resolve()
            relative = candidate.relative_to(
                self.repo_path
            )
        except (
            OSError,
            ValueError,
        ):
            return None

        return relative.as_posix()

    # ========================================================
    # Generic helpers
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:

        if value is None:
            return None

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _safe_string(
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        return str(value)


# ============================================================
# Convenience API
# ============================================================


def analyze_repository(
    repo_path: str | Path,
    timeout: int = 60,
) -> AnalysisResult:
    """
    Convenience function for repository-wide analysis.
    """

    analyzer = StaticAnalyzer(
        repo_path=repo_path,
        timeout=timeout,
    )

    return analyzer.analyze()


def analyze_file(
    repo_path: str | Path,
    file_path: str | Path,
    timeout: int = 60,
) -> AnalysisResult:
    """
    Convenience function for analyzing one file.
    """

    analyzer = StaticAnalyzer(
        repo_path=repo_path,
        timeout=timeout,
    )

    return analyzer.analyze_file(
        file_path
    )


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run Ruff and Bandit against a "
            "Python repository."
        )
    )

    parser.add_argument(
        "repo",
        help="Path to repository",
    )

    parser.add_argument(
        "--file",
        help=(
            "Analyze only one file instead "
            "of the whole repository."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    analyzer = StaticAnalyzer(
        repo_path=args.repo,
        timeout=args.timeout,
    )

    if args.file:
        result = analyzer.analyze_file(
            args.file
        )
    else:
        result = analyzer.analyze()

    print(
        result.render()
    )