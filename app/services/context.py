from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.services.graph import CodeGraph, GraphNode


# ============================================================
# Data models
# ============================================================


@dataclass
class FunctionExtractResult:
    """Exact source extracted for one Python function/method."""

    name: str
    source: str
    start_line: int
    end_line: int
    file_path: str
    qualified_name: str


@dataclass
class ContextItem:
    """One piece of source code included in the final context."""

    file_path: str
    symbol: str
    source: str
    start_line: int | None = None
    end_line: int | None = None
    reason: str = ""


@dataclass
class ContextPacket:
    """
    Compact context sent to the Patch Agent.

    The Patch Agent should receive this rather than the
    entire repository.
    """

    issue: str

    target_file: str | None = None
    target_symbol: str | None = None

    target_code: str | None = None

    related_code: list[ContextItem] = field(
        default_factory=list
    )

    diagnostics: str | None = None

    total_lines: int = 0

    truncated: bool = False

    def render(self) -> str:
        """
        Convert the ContextPacket into text suitable for
        an LLM prompt.
        """

        sections: list[str] = []

        sections.append(
            "## GitHub Issue\n"
            f"{self.issue}"
        )

        if self.target_file:
            sections.append(
                "## Target File\n"
                f"{self.target_file}"
            )

        if self.target_symbol:
            sections.append(
                "## Target Symbol\n"
                f"{self.target_symbol}"
            )

        if self.target_code:
            sections.append(
                "## Target Code\n"
                f"```python\n"
                f"{self.target_code}\n"
                f"```"
            )

        if self.related_code:
            related_sections = []

            for item in self.related_code:
                location = item.file_path

                if (
                    item.start_line is not None
                    and item.end_line is not None
                ):
                    location += (
                        f":{item.start_line}"
                        f"-{item.end_line}"
                    )

                related_sections.append(
                    f"### {item.symbol}\n"
                    f"Location: {location}\n"
                    f"Reason: {item.reason}\n\n"
                    f"```python\n"
                    f"{item.source}\n"
                    f"```"
                )

            sections.append(
                "## Related Code\n"
                + "\n\n".join(related_sections)
            )

        if self.diagnostics:
            sections.append(
                "## Diagnostics\n"
                f"```\n"
                f"{self.diagnostics}\n"
                f"```"
            )

        if self.truncated:
            sections.append(
                "## Context Note\n"
                "Context was limited to the configured "
                "budget. Do not assume omitted code is "
                "irrelevant."
            )

        return "\n\n".join(sections)


# ============================================================
# AST helpers
# ============================================================


def _iter_function_defs(
    tree: ast.AST,
) -> Iterable[
    tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]
]:
    """
    Yield:

        function_name
        ClassName.method_name
        Outer.inner

    for both def and async def.
    """

    def walk(
        node: ast.AST,
        parents: list[str],
    ):
        for child in ast.iter_child_nodes(node):

            if isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                name = ".".join(
                    parents + [child.name]
                )

                yield name, child

                yield from walk(
                    child,
                    parents + [child.name],
                )

            elif isinstance(child, ast.ClassDef):
                yield from walk(
                    child,
                    parents + [child.name],
                )

            else:
                yield from walk(
                    child,
                    parents,
                )

    yield from walk(tree, [])


# ============================================================
# Exact source extraction
# ============================================================


def extract_function(
    file_path: str | Path,
    function_name: str,
    include_decorators: bool = True,
) -> FunctionExtractResult | None:
    """
    Extract one exact function/method from a Python file.

    Supports:

        login
        UserService.login
        Outer.inner
        async functions
    """

    path = Path(file_path)

    if not path.exists():
        return None

    try:
        source = path.read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Important:
        # If the repository contains broken code, AST parsing
        # may fail. The graph should still be able to exist,
        # but exact AST extraction cannot happen here.
        return None

    lines = source.splitlines()

    matches: list[
        tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]
    ] = list(
        _iter_function_defs(tree)
    )

    # --------------------------------------------------------
    # Prefer exact qualified-name match.
    # --------------------------------------------------------

    selected = None

    for qualified_name, node in matches:
        if qualified_name == function_name:
            selected = (
                qualified_name,
                node,
            )
            break

    # --------------------------------------------------------
    # Fall back to bare function-name match.
    # --------------------------------------------------------

    if selected is None:

        for qualified_name, node in matches:

            if node.name == function_name:
                selected = (
                    qualified_name,
                    node,
                )
                break

    if selected is None:
        return None

    qualified_name, node = selected

    if node.lineno is None:
        return None

    start_line = node.lineno

    # --------------------------------------------------------
    # Include decorators.
    # --------------------------------------------------------

    if (
        include_decorators
        and node.decorator_list
    ):
        decorator_lines = [
            decorator.lineno
            for decorator in node.decorator_list
            if decorator.lineno is not None
        ]

        if decorator_lines:
            start_line = min(
                start_line,
                *decorator_lines,
            )

    end_line = node.end_lineno

    if end_line is None:
        return None

    extracted = lines[
        start_line - 1 : end_line
    ]

    return FunctionExtractResult(
        name=node.name,
        source="\n".join(extracted),
        start_line=start_line,
        end_line=end_line,
        file_path=str(path),
        qualified_name=qualified_name,
    )


def list_functions(
    file_path: str | Path,
) -> list[str]:
    """Return all function/method names in a Python file."""

    path = Path(file_path)

    if not path.exists():
        return []

    try:
        source = path.read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    return [
        qualified_name
        for qualified_name, _ in
        _iter_function_defs(tree)
    ]


# ============================================================
# Context Agent
# ============================================================


class ContextBuilder:
    """
    Builds compact repository context for the Patch Agent.

    Responsibilities:

        GitHub issue
             ↓
        CodeGraph
             ↓
        relevant graph nodes
             ↓
        exact source extraction
             ↓
        ContextPacket

    It deliberately does NOT use RAG.
    """

    def __init__(
        self,
        repo_path: str | Path,
        max_lines: int = 150,
        graph_depth: int = 1,
        max_related_nodes: int = 10,
    ) -> None:

        self.repo_path = Path(
            repo_path
        ).resolve()

        self.max_lines = max_lines
        self.graph_depth = graph_depth
        self.max_related_nodes = (
            max_related_nodes
        )

        self.graph = CodeGraph(
            self.repo_path
        )

    # ========================================================
    # Public API
    # ========================================================

    def build_context(
        self,
        issue: str,
        target_symbol: str | None = None,
        target_file: str | Path | None = None,
        diagnostics: str | None = None,
    ) -> ContextPacket:
        """
        Main entry point.

        If target_symbol is known:

            use it as the graph starting point.

        Otherwise:

            search the graph using the issue text.
        """

        # ----------------------------------------------------
        # Ensure graph exists.
        # ----------------------------------------------------

        self._ensure_graph()

        # ----------------------------------------------------
        # Determine target.
        # ----------------------------------------------------

        target_node: GraphNode | None = None

        if target_symbol:
            target_node = self.graph.find_node(
                target_symbol
            )

        # ----------------------------------------------------
        # If no explicit target exists, search issue.
        # ----------------------------------------------------

        if target_node is None:

            candidates = (
                self.graph.find_nodes(
                    issue,
                    limit=1,
                )
            )

            if candidates:
                target_node = candidates[0]

        # ----------------------------------------------------
        # Create packet.
        # ----------------------------------------------------

        packet = ContextPacket(
            issue=issue,
            diagnostics=diagnostics,
        )

        if target_node is None:
            return packet

        packet.target_file = (
            target_node.source_file
        )

        packet.target_symbol = (
            target_node.label
        )

        # ----------------------------------------------------
        # Extract target source.
        # ----------------------------------------------------

        target_result = (
            self._extract_graph_node(
                target_node
            )
        )

        if target_result:
            packet.target_code = (
                target_result.source
            )

            packet.total_lines += (
                self._line_count(
                    target_result.source
                )
            )

        # ----------------------------------------------------
        # Get related graph nodes.
        # ----------------------------------------------------

        related_nodes = (
            self.graph.related_nodes(
                target_node.id,
                depth=self.graph_depth,
                relations={
                    "calls",
                    "imports",
                    "inherits",
                },
            )
        )

        # ----------------------------------------------------
        # Add related source until budget is reached.
        # ----------------------------------------------------

        for node in related_nodes:

            if len(
                packet.related_code
            ) >= self.max_related_nodes:
                packet.truncated = True
                break

            result = (
                self._extract_graph_node(
                    node
                )
            )

            if result is None:
                continue

            source_lines = (
                self._line_count(
                    result.source
                )
            )

            if (
                packet.total_lines
                + source_lines
                > self.max_lines
            ):
                packet.truncated = True
                break

            packet.related_code.append(
                ContextItem(
                    file_path=(
                        result.file_path
                    ),
                    symbol=(
                        result.qualified_name
                    ),
                    source=result.source,
                    start_line=(
                        result.start_line
                    ),
                    end_line=(
                        result.end_line
                    ),
                    reason=(
                        "Related through the "
                        "repository code graph."
                    ),
                )
            )

            packet.total_lines += (
                source_lines
            )

        return packet

    # ========================================================
    # Graph node → source
    # ========================================================

    def _extract_graph_node(
        self,
        node: GraphNode,
    ) -> FunctionExtractResult | None:
        """
        Convert a Graphify node into exact source.

        Graphify tells us WHERE the code is.
        context.py extracts WHAT the code is.
        """

        if not node.source_file:
            return None

        file_path = (
            self._safe_repo_path(
                node.source_file
            )
        )

        if file_path is None:
            return None

        function_name = node.label.strip()

        if function_name.startswith("."):
            function_name = function_name[1:]

        if function_name.endswith("()"):
            function_name = function_name[:-2]

        result = extract_function(
            file_path=file_path,
            function_name=function_name,
        )

        if result:
            return result

        # ----------------------------------------------------
        # Graph labels may not exactly match the qualified
        # Python name. If the label contains a module prefix,
        # try the final component.
        # ----------------------------------------------------

        if "." in function_name:

            short_name = (
                function_name.rsplit(
                    ".",
                    1,
                )[-1]
            )

            result = extract_function(
                file_path=file_path,
                function_name=short_name,
            )

            if result:
                return result

        return None

    # ========================================================
    # Graph initialization
    # ========================================================

    def _ensure_graph(self) -> None:
        """Load an existing graph or build one."""

        if self.graph.graph is not None:
            return

        if self.graph.graph_path.exists():
            self.graph.load()
            return

        self.graph.build()
        self.graph.load()

    # ========================================================
    # Security
    # ========================================================

    def _safe_repo_path(
        self,
        file_path: str | Path,
    ) -> Path | None:
        """
        Ensure Graphify cannot make us read files outside
        the cloned repository.
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

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _line_count(
        source: str,
    ) -> int:
        if not source:
            return 0

        return len(
            source.splitlines()
        )


ContextAgent = ContextBuilder


# ============================================================
# Convenience function
# ============================================================


def build_context(
    repo_path: str | Path,
    issue: str,
    target_symbol: str | None = None,
    target_file: str | Path | None = None,
    diagnostics: str | None = None,
    max_lines: int = 150,
) -> ContextPacket:
    """
    Convenience wrapper for the Context Agent.
    """

    builder = ContextBuilder(
        repo_path=repo_path,
        max_lines=max_lines,
    )

    return builder.build_context(
        issue=issue,
        target_symbol=target_symbol,
        target_file=target_file,
        diagnostics=diagnostics,
    )


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build compact repository context "
            "for Varmintor."
        )
    )

    parser.add_argument(
        "repo",
        help="Path to repository",
    )

    parser.add_argument(
        "--issue",
        required=True,
        help="Issue description",
    )

    parser.add_argument(
        "--symbol",
        help="Known target symbol",
    )

    parser.add_argument(
        "--file",
        help="Known target file",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=150,
    )

    args = parser.parse_args()

    packet = build_context(
        repo_path=args.repo,
        issue=args.issue,
        target_symbol=args.symbol,
        target_file=args.file,
        max_lines=args.max_lines,
    )

    print(
        packet.render()
    )