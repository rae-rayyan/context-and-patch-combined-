from __future__ import annotations

import sys
import types
from pathlib import Path


# These stubs let this add-on be unit-tested without copying the upstream
# Context-Agent implementation into this bundle.
context_stub = types.ModuleType("app.services.context")


class ContextPacket:
    def __init__(self, issue, target_file=None, target_symbol=None, target_code=None, related_code=None):
        self.issue = issue
        self.target_file = target_file
        self.target_symbol = target_symbol
        self.target_code = target_code
        self.related_code = related_code or []
        self.diagnostics = None
        self.truncated = False

    def render(self):
        return self.target_code or self.issue


context_stub.ContextPacket = ContextPacket
sys.modules["app.services.context"] = context_stub

verification_stub = types.ModuleType("app.services.verification")


class NoopVerifier:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def snapshot(self, files):
        return None

    def verify(self, changed_files, baseline):
        from app.services.patch_models import VerificationResult, VerificationStep

        return VerificationResult(
            passed=True,
            steps=[VerificationStep("unit-test verification", True)],
        )


verification_stub.RepositoryVerifier = NoopVerifier
sys.modules["app.services.verification"] = verification_stub

from app.services.patch_agent import PatchAgent


class FakeLLM:
    def generate_patch(self, prompt: str):
        return {
            "summary": "Fix login boolean check",
            "changes": [
                {
                    "path": "app.py",
                    "change_type": "replace",
                    "old_text": "def login(password):\n    return password == 'secret'\n",
                    "new_text": "def login(password):\n    return bool(password) and password == 'secret'\n",
                    "reason": "Avoid accepting an empty value as a valid credential path.",
                }
            ],
            "tests_to_focus": [],
        }


def test_patch_agent_applies_safe_change(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "app.py"
    source.write_text("def login(password):\n    return password == 'secret'\n", encoding="utf-8")

    context = ContextPacket(
        issue="login bug",
        target_file="app.py",
        target_symbol="login",
        target_code="def login(password):\n    return password == 'secret'\n",
    )

    result = PatchAgent(repo, FakeLLM(), verifier=NoopVerifier(repo)).run(context)

    assert result.success
    assert "bool(password)" in source.read_text(encoding="utf-8")
    assert "a/app.py" in result.diff
