from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class GraphNode:
    """A code entity discovered by Graphify."""

    id: str
    label: str
    source_file: str | None = None
    file_type: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class GraphEdge:
    """A relationship between two graph nodes."""

    source: str
    target: str
    relation: str | None = None
    confidence: str | None = None


class CodeGraph:
    """
    Adapter around a Graphify-generated code graph.

    Responsibilities:
    - build/update the Graphify graph
    - load graph.json
    - find relevant symbols
    - inspect callers/callees
    - traverse a limited number of hops
    - return source locations for Context Agent

    This class does NOT extract source code.
    That remains the responsibility of context.py.
    """

    def __init__(
        self,
        repo_path: str | Path,
        graph_path: str | Path | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()

        if graph_path is None:
            graph_path = self.repo_path / "graphify-out" / "graph.json"

        self.graph_path = Path(graph_path).resolve()

        self.graph: nx.Graph | nx.DiGraph | None = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> Path:
        """
        Run Graphify against the repository.

        Returns:
            Path to generated graph.json.
        """

        command = [
            "graphify",
            "extract",
            str(self.repo_path),
            "--code-only",
            "--out",
            str(self.repo_path),
        ]

        if force:
            command.append("--force")

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Graphify is not installed or the 'graphify' command "
                "is not available on PATH."
            ) from exc

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Graphify failed.\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            ) from exc

        if not self.graph_path.exists():
            raise RuntimeError(
                f"Graphify completed but graph was not created: {self.graph_path}"
            )

        return self.graph_path

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load graph.json into NetworkX."""

        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file does not exist: {self.graph_path}")

        with self.graph_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Graphify can produce either "links" or "edges".
        # Normalize the latter to NetworkX's expected key.
        if "links" not in data and "edges" in data:
            data = {
                **data,
                "links": data["edges"],
            }

        try:
            self.graph = nx.node_link_graph(
                data,
                edges="links",
            )
        except TypeError:
            # Compatibility with older NetworkX versions.
            self.graph = nx.node_link_graph(data)

    def _ensure_loaded(self) -> None:
        """Load graph lazily."""

        if self.graph is None:
            self.load()

    # ------------------------------------------------------------------
    # Node search
    # ------------------------------------------------------------------

    def find_nodes(
        self,
        query: str,
        limit: int = 10,
    ) -> list[GraphNode]:
        """
        Find graph nodes whose labels match the query.

        This is deliberately simple.

        Later you can improve the ranking using:
        - issue keywords
        - traceback symbols
        - filenames
        - function names
        - class names
        """

        self._ensure_loaded()

        assert self.graph is not None

        query_terms = {term.lower() for term in query.split() if len(term) >= 2}

        scored: list[tuple[int, str]] = []

        for node_id, data in self.graph.nodes(data=True):
            label = str(data.get("label", node_id)).lower()

            score = sum(1 for term in query_terms if term in label)

            if score > 0:
                scored.append((score, node_id))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results: list[GraphNode] = []

        for _, node_id in scored[:limit]:
            results.append(self._node_from_id(node_id))

        return results

    def find_node(self, symbol: str):
        """
        Resolve a human-readable symbol to a Graphify node.

        Supported forms:
            login
            AuthService.login
            app_auth_authservice_login

        Returns:
            GraphNode or None
        """

        symbol = symbol.strip()

        if not symbol:
            return None

        self._ensure_loaded()

        assert self.graph is not None

        # ---------------------------------------------------------
        # 1. Exact Graphify node ID
        # ---------------------------------------------------------

        for node_id, attrs in self.graph.nodes(data=True):
            if node_id == symbol:
                return self._node_from_id(node_id)

        # ---------------------------------------------------------
        # 2. Exact label match
        # ---------------------------------------------------------

        symbol_lower = symbol.lower()

        for node_id, attrs in self.graph.nodes(data=True):
            label = str(attrs.get("label", "")).strip()

            if label.lower() == symbol_lower:
                return self._node_from_id(node_id)

        # ---------------------------------------------------------
        # 3. Method/class qualified name
        #
        # Example:
        #     AuthService.login
        #
        # Graphify:
        #     label       = .login()
        #     source_file = app/auth.py
        # ---------------------------------------------------------

        if "." in symbol:
            class_name, method_name = symbol.rsplit(".", 1)

            method_name = method_name.lower()

            for node_id, attrs in self.graph.nodes(data=True):
                label = str(attrs.get("label", "")).strip()
                source_file = str(attrs.get("source_file", "")).strip()

                # Graphify method labels look like:
                # .login()
                label_method = label.removeprefix(".").removesuffix("()").lower()

                if label_method != method_name:
                    continue

                # The class name is generally encoded in the Graphify ID.
                normalized_id = node_id.lower()

                if class_name.lower() in normalized_id:
                    return self._node_from_id(node_id)

        # ---------------------------------------------------------
        # 4. Bare function/method name
        # ---------------------------------------------------------

        matches = []

        for node_id, attrs in self.graph.nodes(data=True):
            label = str(attrs.get("label", "")).strip()

            label_name = label.removeprefix(".").removesuffix("()").lower()

            if label_name == symbol_lower:
                matches.append(self._node_from_id(node_id))

        # Only return an unqualified match if it is unambiguous.
        if len(matches) == 1:
            return matches[0]

        return None

    # ------------------------------------------------------------------
    # Node information
    # ------------------------------------------------------------------

    def get_node(
        self,
        node_id: str,
    ) -> GraphNode | None:
        """Return a node by its graph ID."""

        self._ensure_loaded()

        assert self.graph is not None

        if node_id not in self.graph.nodes:
            return None

        return self._node_from_id(node_id)

    def _node_from_id(
        self,
        node_id: str,
    ) -> GraphNode:
        """Convert NetworkX node data into our domain object."""

        assert self.graph is not None

        data = self.graph.nodes[node_id]

        return GraphNode(
            id=str(node_id),
            label=str(data.get("label", node_id)),
            source_file=data.get("source_file"),
            file_type=data.get("file_type"),
            start_line=self._get_line(
                data,
                "start_line",
            ),
            end_line=self._get_line(
                data,
                "end_line",
            ),
        )

    @staticmethod
    def _get_line(
        data: dict[str, Any],
        key: str,
    ) -> int | None:
        value = data.get(key)

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        relation: str | None = None,
    ) -> list[GraphNode]:
        """
        Return nodes connected to a given node.

        Optional relation filtering:

            graph.neighbors(
                node_id,
                relation="calls",
            )
        """

        self._ensure_loaded()

        assert self.graph is not None

        if node_id not in self.graph:
            return []

        results: list[GraphNode] = []

        for neighbor in self.graph.neighbors(node_id):
            edge_data = self._edge_data(
                node_id,
                neighbor,
            )

            if relation is not None:
                if edge_data.get("relation") != relation:
                    continue

            results.append(self._node_from_id(neighbor))

        return results

    def callees(
        self,
        node_id: str,
    ) -> list[GraphNode]:
        """
        Return functions/classes called by this node.
        """

        return self.neighbors(
            node_id,
            relation="calls",
        )

    def callers(
        self,
        node_id: str,
    ) -> list[GraphNode]:
        """
        Return nodes that call this node.
        """

        self._ensure_loaded()

        assert self.graph is not None

        results: list[GraphNode] = []

        for source in self.graph.nodes:
            for target in self.graph.successors(source):
                if target != node_id:
                    continue

                edge_data = self._edge_data(
                    source,
                    target,
                )

                if edge_data.get("relation") == "calls":
                    results.append(self._node_from_id(source))

        return results

    def imports(
        self,
        node_id: str,
    ) -> list[GraphNode]:
        """Return nodes related through imports."""

        return self.neighbors(
            node_id,
            relation="imports",
        )

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def related_nodes(
        self,
        node_id: str,
        depth: int = 1,
        relations: set[str] | None = None,
    ) -> list[GraphNode]:
        """
        Return nodes within N graph hops.

        Example:

            related_nodes(
                "UserService.login",
                depth=2,
                relations={"calls"},
            )

        Important:
        Keep depth small. The Context Agent should not dump
        the entire repository into the Patch Agent.
        """

        self._ensure_loaded()

        assert self.graph is not None

        if node_id not in self.graph:
            return []

        visited: set[str] = {node_id}

        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()

            for current in frontier:
                for neighbor in self.graph.neighbors(current):
                    edge_data = self._edge_data(
                        current,
                        neighbor,
                    )

                    if (
                        relations is not None
                        and edge_data.get("relation") not in relations
                    ):
                        continue

                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)

            frontier = next_frontier

            if not frontier:
                break

        visited.remove(node_id)

        return [self._node_from_id(node) for node in visited]

    # ------------------------------------------------------------------
    # Context-oriented retrieval
    # ------------------------------------------------------------------

    def get_context_nodes(
        self,
        query: str,
        depth: int = 1,
        limit: int = 5,
    ) -> list[GraphNode]:
        """
        Main method intended for Context Agent.

        Flow:

            issue description
                    ↓
              find_nodes()
                    ↓
             graph traversal
                    ↓
              source locations

        Returns a compact list of potentially relevant
        code entities.
        """

        starting_nodes = self.find_nodes(
            query,
            limit=limit,
        )

        if not starting_nodes:
            return []

        results: dict[str, GraphNode] = {}

        for node in starting_nodes:
            results[node.id] = node

            related = self.related_nodes(
                node.id,
                depth=depth,
                relations={
                    "calls",
                    "imports",
                    "inherits",
                },
            )

            for related_node in related:
                if len(results) >= limit * 5:
                    break

                results[related_node.id] = related_node

        return list(results.values())

    # ------------------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------------------

    def _edge_data(
        self,
        source: str,
        target: str,
    ) -> dict[str, Any]:
        """Safely retrieve edge metadata."""

        assert self.graph is not None

        data = self.graph.get_edge_data(
            source,
            target,
        )

        if not data:
            return {}

        # MultiGraph / MultiDiGraph
        if self.graph.is_multigraph():
            first_edge = next(
                iter(data.values()),
                {},
            )
            return dict(first_edge)

        return dict(data)

    # ------------------------------------------------------------------
    # Debugging / inspection
    # ------------------------------------------------------------------

    def describe(
        self,
        node_id: str,
    ) -> dict[str, Any]:
        """
        Return useful information about a node.

        Helpful while developing the Context Agent.
        """

        node = self.get_node(node_id)

        if node is None:
            return {}

        callees = self.callees(node_id)
        callers = self.callers(node_id)

        return {
            "node": node.__dict__,
            "callees": [item.__dict__ for item in callees],
            "callers": [item.__dict__ for item in callers],
        }
