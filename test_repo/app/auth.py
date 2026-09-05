class AuthService:

    def find_user(self, username):
        return {
            "username": username,
            "password": "secret",
        }

    def verify_password(self, password, stored_password):
        return password == stored_password

    def login(self, username, password):
        user = self.find_user(username)

        if not user:
            return False

        return self.verify_password(
            password,
            user["password"],
        )