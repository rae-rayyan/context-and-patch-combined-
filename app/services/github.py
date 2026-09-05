from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github import Auth, Github
from github.GithubException import GithubException


# ============================================================
# Data models
# ============================================================


@dataclass
class RepositoryInfo:
    """Basic information about a GitHub repository."""

    owner: str
    name: str
    full_name: str
    clone_url: str


@dataclass
class IssueInfo:
    """Information extracted from a GitHub issue."""

    number: int
    title: str
    body: str
    url: str
    owner: str
    repo: str

    @property
    def description(self) -> str:
        """
        Combined issue text used by the Context Agent.
        """

        if self.body.strip():
            return f"{self.title}\n\n{self.body}"

        return self.title


@dataclass
class PullRequestInfo:
    """Information about a created pull request."""

    number: int
    title: str
    url: str
    branch: str
    base_branch: str


# ============================================================
# GitHub service
# ============================================================


class GitHubService:
    """
    GitHub integration for Varmintor.

    Responsibilities:

        GitHub
          |
          +-- repository information
          +-- issue information
          +-- clone repository
          +-- create branch
          +-- commit changes
          +-- push branch
          +-- create pull request

    This class intentionally does NOT handle:

        - Graphify
        - AST extraction
        - Ruff
        - Bandit
        - Docker QA
        - LangGraph
        - LLM calls
    """

    def __init__(
        self,
        token: str | None = None,
        timeout: int = 60,
    ) -> None:

        self.token = token or os.getenv("GITHUB_TOKEN")

        if not self.token:
            raise ValueError(
                "GitHub token not provided. Set GITHUB_TOKEN or pass token explicitly."
            )

        self.timeout = timeout

        # PyGithub currently supports Auth.Token(...)
        # as the preferred authentication style.
        self.github = Github(
            auth=Auth.Token(self.token),
            timeout=timeout,
        )

    # ========================================================
    # Authentication
    # ========================================================

    def test_connection(self) -> str:
        """
        Verify that the GitHub token works.

        Returns:
            Authenticated GitHub username.
        """

        try:
            user = self.github.get_user()
            return user.login

        except GithubException as exc:
            raise RuntimeError(f"GitHub authentication failed: {exc}") from exc

    # ========================================================
    # Repository
    # ========================================================

    def get_repository(
        self,
        repo_url: str,
    ) -> RepositoryInfo:
        """
        Parse a GitHub repository URL and return repository info.

        Supports:

            https://github.com/owner/repo
            https://github.com/owner/repo.git
            git@github.com:owner/repo.git
        """

        owner, name = self._parse_repo_url(repo_url)

        return RepositoryInfo(
            owner=owner,
            name=name,
            full_name=f"{owner}/{name}",
            clone_url=(f"https://github.com/{owner}/{name}.git"),
        )

    def get_issue(
        self,
        repo_url: str,
        issue_number: int,
    ) -> IssueInfo:
        """
        Retrieve a GitHub issue.

        This is what the webhook/LangGraph entry point can use
        to turn a GitHub issue into the issue description consumed
        by ContextBuilder.
        """

        repo_info = self.get_repository(repo_url)

        try:
            repo = self.github.get_repo(repo_info.full_name)

            issue = repo.get_issue(number=issue_number)

        except GithubException as exc:
            raise RuntimeError(
                f"Could not retrieve issue {repo_info.full_name}#{issue_number}: {exc}"
            ) from exc

        return IssueInfo(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            url=issue.html_url,
            owner=repo_info.owner,
            repo=repo_info.name,
        )

    # ========================================================
    # Clone
    # ========================================================

    def clone_repository(
        self,
        repo_url: str,
        destination: str | Path,
        branch: str | None = None,
    ) -> Path:
        """
        Clone a GitHub repository.

        The token is supplied through the HTTPS clone URL.
        The URL is never printed or logged.
        """

        destination = Path(destination).resolve()

        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(f"Clone destination is not empty: {destination}")
        else:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        authenticated_url = _authenticated_clone_url(self, repo_url)

        command = ["git", "clone"]

        if branch:
            command.extend(
                [
                    "--branch",
                    branch,
                ]
            )

        command.extend(
            [
                authenticated_url,
                str(destination),
            ]
        )

        self._run_git(
            command,
            cwd=destination.parent,
        )

        return destination

    # ========================================================
    # Branch management
    # ========================================================

    def create_branch(
        self,
        repo_path: str | Path,
        branch_name: str,
        base_branch: str = "main",
    ) -> None:
        """
        Create and checkout a new working branch.

        The base branch is fetched first.
        """

        repo_path = self._validate_repo_path(repo_path)

        self._run_git(
            [
                "fetch",
                "origin",
                base_branch,
            ],
            cwd=repo_path,
        )

        self._run_git(
            [
                "checkout",
                "-B",
                branch_name,
                f"origin/{base_branch}",
            ],
            cwd=repo_path,
        )

    def get_current_branch(
        self,
        repo_path: str | Path,
    ) -> str:
        """Return the current Git branch."""

        repo_path = self._validate_repo_path(repo_path)

        result = self._run_git(
            [
                "branch",
                "--show-current",
            ],
            cwd=repo_path,
        )

        branch = result.stdout.strip()

        if not branch:
            raise RuntimeError("Could not determine current Git branch.")

        return branch

    # ========================================================
    # Change detection
    # ========================================================

    def has_changes(
        self,
        repo_path: str | Path,
    ) -> bool:
        """Return True if the working tree has changes."""

        repo_path = self._validate_repo_path(repo_path)

        result = self._run_git(
            [
                "status",
                "--porcelain",
            ],
            cwd=repo_path,
        )

        return bool(result.stdout.strip())

    def changed_files(
        self,
        repo_path: str | Path,
    ) -> list[str]:
        """
        Return repository-relative paths changed in the
        working tree.
        """

        repo_path = self._validate_repo_path(repo_path)

        result = self._run_git(
            [
                "status",
                "--porcelain",
            ],
            cwd=repo_path,
        )

        files: list[str] = []

        for line in result.stdout.splitlines():
            if not line.strip():
                continue

            # Porcelain format:
            #
            # XY filename
            #
            # We only need the path after the status columns.
            file_name = line[3:].strip()

            # Handle quoted paths conservatively.
            if file_name.startswith('"'):
                file_name = file_name.strip('"')

            # For rename:
            #
            # old.py -> new.py
            #
            if " -> " in file_name:
                file_name = file_name.split(
                    " -> ",
                    1,
                )[1]

            files.append(file_name)

        return files

    # ========================================================
    # Commit
    # ========================================================

    def commit_changes(
        self,
        repo_path: str | Path,
        message: str,
        files: list[str] | None = None,
    ) -> str:
        """
        Stage and commit changes.

        Returns:
            New commit SHA.
        """

        repo_path = self._validate_repo_path(repo_path)

        if not self.has_changes(repo_path):
            raise RuntimeError("No changes to commit.")

        if files:
            safe_files = []

            for file_path in files:
                safe_files.append(
                    self._validate_repo_file(
                        repo_path,
                        file_path,
                    )
                )

            self._run_git(
                [
                    "add",
                    "--",
                    *[str(path.relative_to(repo_path)) for path in safe_files],
                ],
                cwd=repo_path,
            )

        else:
            self._run_git(
                [
                    "add",
                    "-A",
                ],
                cwd=repo_path,
            )

        self._run_git(
            [
                "commit",
                "-m",
                message,
            ],
            cwd=repo_path,
        )

        result = self._run_git(
            [
                "rev-parse",
                "HEAD",
            ],
            cwd=repo_path,
        )

        return result.stdout.strip()

    # ========================================================
    # Push
    # ========================================================

    def push_branch(
        self,
        repo_path: str | Path,
        branch_name: str | None = None,
    ) -> None:
        """
        Push the working branch to origin.

        If branch_name is omitted, the current branch is used.
        """

        repo_path = self._validate_repo_path(repo_path)

        if branch_name is None:
            branch_name = self.get_current_branch(repo_path)

        self._run_git(
            [
                "push",
                "--set-upstream",
                "origin",
                branch_name,
            ],
            cwd=repo_path,
        )

    # ========================================================
    # Pull request
    # ========================================================

    def create_pull_request(
        self,
        repo_url: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> PullRequestInfo:
        """
        Create a GitHub pull request.

        The branch must already exist on the remote.
        """

        repo_info = self.get_repository(repo_url)

        try:
            repo = self.github.get_repo(repo_info.full_name)

            pull_request = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )

        except GithubException as exc:
            raise RuntimeError(f"Could not create pull request: {exc}") from exc

        return PullRequestInfo(
            number=pull_request.number,
            title=pull_request.title,
            url=pull_request.html_url,
            branch=head_branch,
            base_branch=base_branch,
        )

    # ========================================================
    # Complete Patch → PR workflow
    # ========================================================

    def publish_patch(
        self,
        repo_url: str,
        repo_path: str | Path,
        branch_name: str,
        commit_message: str,
        pr_title: str,
        pr_body: str,
        base_branch: str = "main",
    ) -> PullRequestInfo:
        """
        Complete GitHub publication step.

        Expected state:

            repository cloned
                ↓
            Patch Agent modified files
                ↓
            this method
                ↓
            commit
                ↓
            push
                ↓
            pull request
        """

        repo_path = self._validate_repo_path(repo_path)

        if not self.has_changes(repo_path):
            raise RuntimeError("Patch produced no repository changes.")

        current_branch = self.get_current_branch(repo_path)

        if current_branch != branch_name:
            raise RuntimeError(
                f"Expected branch "
                f"'{branch_name}', "
                f"but repository is currently on "
                f"'{current_branch}'."
            )

        self.commit_changes(
            repo_path=repo_path,
            message=commit_message,
        )

        self.push_branch(
            repo_path=repo_path,
            branch_name=branch_name,
        )

        return self.create_pull_request(
            repo_url=repo_url,
            title=pr_title,
            body=pr_body,
            head_branch=branch_name,
            base_branch=base_branch,
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def cleanup_repository(
        self,
        repo_path: str | Path,
    ) -> None:
        """
        Delete a temporary cloned repository.

        Only call this on repositories created by Varmintor.
        """

        path = Path(repo_path).resolve()

        if not path.exists():
            return

        if path == Path("/"):
            raise RuntimeError("Refusing to delete filesystem root.")

        shutil.rmtree(path)

    # ========================================================
    # Git command helper
    # ========================================================

    def _run_git(
        self,
        command: list[str],
        cwd: str | Path,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a Git command safely.

        Authentication is supplied through environment variables
        rather than putting the GitHub token directly into the
        command line.
        """

        env = os.environ.copy()

        # Prevent Git from unexpectedly opening an interactive
        # username/password prompt.
        env["GIT_TERMINAL_PROMPT"] = "0"

        # Use the GitHub token through an askpass helper.
        #
        # This avoids:
        #
        # https://TOKEN@github.com/...
        #
        # appearing in the process command line.
        askpass_script = "import os; print(os.environ['VARMINTOR_GITHUB_TOKEN'])"

        env["VAMINTOR_GIT_ASKPASS"] = askpass_script

        # Git expects GIT_ASKPASS to point to an executable
        # program, so we use the Python interpreter itself.
        env["GIT_ASKPASS"] = os.sys.executable

        env["VARMINTOR_GITHUB_TOKEN"] = self.token

        try:
            return subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
            )

        except FileNotFoundError as exc:
            raise RuntimeError(
                "Git is not installed or is not available on PATH."
            ) from exc

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Git command timed out after "
                f"{self.timeout} seconds: "
                f"{' '.join(command)}"
            ) from exc

        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""

            stdout = exc.stdout.strip() if exc.stdout else ""

            details = stderr or stdout or "unknown Git error"

            raise RuntimeError(
                f"Git command failed: {' '.join(command)}\n{details}"
            ) from exc

    # ========================================================
    # Path validation
    # ========================================================

    def _validate_repo_path(
        self,
        repo_path: str | Path,
    ) -> Path:
        """Validate a local Git repository path."""

        path = Path(repo_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Repository does not exist: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Repository is not a directory: {path}")

        if not ((path / ".git").exists()):
            raise ValueError(f"Not a Git repository: {path}")

        return path

    def _validate_repo_file(
        self,
        repo_path: Path,
        file_path: str | Path,
    ) -> Path:
        """
        Make sure a file passed to git add is inside the repo.
        """

        candidate = Path(file_path)

        if not candidate.is_absolute():
            candidate = repo_path / candidate

        candidate = candidate.resolve()

        try:
            candidate.relative_to(repo_path)
        except ValueError as exc:
            raise ValueError(f"File is outside repository: {file_path}") from exc

        if not candidate.exists():
            raise FileNotFoundError(f"File does not exist: {candidate}")

        return candidate

    # ========================================================
    # URL parsing
    # ========================================================

    @staticmethod
    def _parse_repo_url(
        repo_url: str,
    ) -> tuple[str, str]:
        """
        Parse common GitHub repository URL formats.
        """

        value = repo_url.strip()

        patterns = [
            # HTTPS:
            # https://github.com/owner/repo
            # https://github.com/owner/repo.git
            r"^https?://github\.com/"
            r"(?P<owner>[^/]+)/"
            r"(?P<repo>[^/#]+)"
            r"/?$",
            # SSH:
            # git@github.com:owner/repo.git
            r"^git@github\.com:"
            r"(?P<owner>[^/]+)/"
            r"(?P<repo>[^/#]+)"
            r"/?$",
        ]

        for pattern in patterns:
            match = re.match(
                pattern,
                value,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            owner = match.group("owner")

            repo = match.group("repo")

            if repo.endswith(".git"):
                repo = repo[:-4]

            if not owner or not repo:
                break

            return owner, repo

        raise ValueError(f"Unsupported GitHub repository URL: {repo_url}")


# ============================================================
# Convenience functions
# ============================================================


def clone_repository(
    repo_url: str,
    destination: str | Path,
    branch: str | None = None,
    token: str | None = None,
) -> Path:
    """Convenience wrapper around GitHubService.clone_repository."""

    service = GitHubService(token=token)

    return service.clone_repository(
        repo_url=repo_url,
        destination=destination,
        branch=branch,
    )


def get_issue(
    repo_url: str,
    issue_number: int,
    token: str | None = None,
) -> IssueInfo:
    """Convenience wrapper around GitHubService.get_issue."""

    service = GitHubService(token=token)

    return service.get_issue(
        repo_url=repo_url,
        issue_number=issue_number,
    )


def _authenticated_clone_url(
    self,
    repo_url: str,
) -> str:
    """
    Build an authenticated HTTPS GitHub URL.

    The returned value must never be logged.
    """

    owner, repo = self._parse_repo_url(repo_url)

    return f"https://x-access-token:{self.token}@github.com/{owner}/{repo}.git"


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=("GitHub utilities for Varmintor."))

    parser.add_argument(
        "--repo",
        help="GitHub repository URL.",
    )

    parser.add_argument(
        "--issue",
        type=int,
        help="GitHub issue number.",
    )

    parser.add_argument(
        "--test-auth",
        action="store_true",
        help="Test GitHub authentication.",
    )

    args = parser.parse_args()

    service = GitHubService()

    if args.test_auth:
        username = service.test_connection()
        print(f"Authenticated as: {username}")

    elif args.repo and args.issue:
        issue = service.get_issue(
            repo_url=args.repo,
            issue_number=args.issue,
        )

        print(f"Issue #{issue.number}: {issue.title}")

        print()
        print(issue.description)

    else:
        parser.error("Use --test-auth or provide --repo and --issue.")
