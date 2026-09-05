from pathlib import Path

print("=== CONTEXT TEST STARTED ===")

from app.services.graph import CodeGraph
print("Graph imported")

from app.services.context import ContextAgent
print("Context imported")

repo_path = Path("test_repo")

graph = CodeGraph(repo_path)
graph.load()
print("Graph loaded")

agent = ContextAgent(repo_path)
print("Context agent created")

symbol = "AuthService.login"
result = agent.build_context(
	issue="Resolve and inspect the login method.",
	target_symbol=symbol,
)

print()
print("=== CONTEXT RESULT ===")
print("Symbol:", symbol)
print("File:", result.target_file)
print()
print(result.target_code or "Context source not found")

print("=== CONTEXT TEST FINISHED ===")