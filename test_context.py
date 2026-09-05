from app.services.context import (
    extract_function,
    list_functions,
)


file = "test_repo/app/auth.py"


print("=== FUNCTIONS ===")

for function in list_functions(file):
    print(function)


print()
print("=== LOGIN FUNCTION ===")

result = extract_function(
    file,
    "AuthService.login",
)

if result:
    print(result.source)
    print()
    print("Start:", result.start_line)
    print("End:", result.end_line)
else:
    print("Function not found")