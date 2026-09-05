from app.services.context import ContextBuilder
from app.services.static_analysis import StaticAnalyzer


repo = "test_repo"


# ---------------------------------
# Static analysis
# ---------------------------------

analyzer = StaticAnalyzer(repo)

analysis = analyzer.analyze()


# ---------------------------------
# Context
# ---------------------------------

builder = ContextBuilder(
    repo_path=repo,
    max_lines=150,
    graph_depth=1,
)


packet = builder.build_context(
    issue="Login fails when password is incorrect",
    target_symbol="AuthService.login",
    diagnostics=analysis.render(),
)


print(packet.render())