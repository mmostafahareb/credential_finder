import os
import requests
import subprocess

# List of GitHub usernames
usernames = ["mmostafahareb","markodenic","codewithsadee","SimonHoiberg","SalimMersally","ralfaouad","iamamiramine","karimkishly","AmineJml","wissamfawaz"]

# Directory where the repositories will be cloned
base_dir = "./"

# GitHub API base URL
api_base_url = "https://api.github.com"

# Clone repositories for each username
for username in usernames:
    # Create a directory for the user's repositories
    user_dir = os.path.join(base_dir, username)
    os.makedirs(user_dir, exist_ok=True)

    # Retrieve the list of public repositories for the user
    try:
        url = f"{api_base_url}/users/{username}/repos"
        response = requests.get(url)
        response.raise_for_status()
        repositories = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch repositories for user: {username}")
        print(f"Error message: {str(e)}")
        continue

    # Clone each repository
    for repo in repositories:
        repo_name = repo["name"]
        repo_url = repo["clone_url"]
        repo_dir = os.path.join(user_dir, repo_name)

        print(f"Cloning repository: {repo_url}")
        try:
            subprocess.run(["git", "clone", repo_url, repo_dir])
        except subprocess.SubprocessError as e:
            print(f"Failed to clone repository: {repo_name}")
            print(f"Error message: {str(e)}")