from app.services.graph import CodeGraph


graph = CodeGraph("test_repo")

graph.load()

print("=== LOGIN SEARCH ===")

nodes = graph.find_nodes(
    "login",
    limit=5,
)

for node in nodes:
    print(node)


print()
print("=== EXACT NODE ===")

node = graph.find_node(
    "AuthService.login"
)

print(node)


if node:
    print()
    print("=== CALLEES ===")

    for callee in graph.callees(node.id):
        print(callee)