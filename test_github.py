from app.services.github import GitHubService


service = GitHubService()


path = service.clone_repository(
    repo_url="https://github.com/your_username/test_repo",
    destination="github_test/repo",
)


print("Cloned to:")
print(path)