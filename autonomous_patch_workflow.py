from __future__ import annotations

import argparse
from pathlib import Path

from app.services.context import ContextBuilder
from app.services.github import GitHubService
from app.services.llm import OpenAIPatchLLM
from app.services.patch_agent import PatchAgent
from app.services.static_analysis import StaticAnalyzer


def run_issue_repair(
    repo_url: str,
    issue_number: int,
    workspace: str | Path,
    base_branch: str = "main",
    publish: bool = False,
) -> object:
    """Context Agent -> Patch Agent -> verification -> optional PR.

    The existing Context-Agent GitHubService owns clone/branch/commit/push/PR.
    The new PatchAgent owns only code modification and verification.
    """

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    github = GitHubService()
    issue = github.get_issue(repo_url, issue_number)
    repo_info = github.get_repository(repo_url)

    repo_path = workspace / repo_info.name
    if not repo_path.exists():
        github.clone_repository(repo_url, repo_path, branch=base_branch)

    branch_name = f"patch/issue-{issue_number}"
    github.create_branch(repo_path, branch_name, base_branch=base_branch)

    analysis = StaticAnalyzer(repo_path).analyze()
    context = ContextBuilder(repo_path, max_lines=200, graph_depth=1).build_context(
        issue=f"#{issue.number}: {issue.title}\n\n{issue.body}",
        diagnostics=analysis.render(),
    )

    patch_agent = PatchAgent(repo_path, llm=OpenAIPatchLLM(), max_attempts=3)
    result = patch_agent.run(context)

    if not result.success:
        return result

    if publish:
        pr = github.publish_patch(
            repo_url=repo_url,
            repo_path=repo_path,
            branch_name=branch_name,
            commit_message=f"fix: repair issue #{issue_number}",
            pr_title=result.summary,
            pr_body=(
                f"Automated repair for {issue.url}.\n\n"
                "Patch Agent verification passed.\n\n"
                f"Attempts: {result.attempts}\n\n"
                "```diff\n"
                f"{result.diff}\n"
                "```"
            ),
            base_branch=base_branch,
        )
        return pr

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Context -> Patch workflow")
    parser.add_argument("repo_url")
    parser.add_argument("issue_number", type=int)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    result = run_issue_repair(
        repo_url=args.repo_url,
        issue_number=args.issue_number,
        workspace=args.workspace,
        base_branch=args.base_branch,
        publish=args.publish,
    )
    print(result)


if __name__ == "__main__":
    main()
