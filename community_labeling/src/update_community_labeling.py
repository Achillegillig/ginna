import os
import requests
import json
from datetime import datetime

# GitHub settings
REPO = "Achillegillig/ginna" 
TOKEN = os.getenv("GITHUB_TOKEN")
API_URL = f"https://api.github.com/repos/{REPO}"

HEADERS = {"Authorization": f"token {TOKEN}"}

def fetch_issues():
    """Fetch all open issues from the repository."""
    issues = []
    url = f"{API_URL}/issues"
    while url:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            raise Exception(f"Error fetching issues: {response.json()}")
        issues.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return issues

def summarize_votes(issue):
    """Summarize votes from comments in an issue."""
    url = issue["comments_url"]
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching comments: {response.json()}")
    comments = response.json()

    summary = {"👍": 0, "👎": 0, "🧐": 0}

    for comment in comments:
        reactions_url = comment["reactions"]["url"]
        reactions_response = requests.get(reactions_url, headers=HEADERS)
        if reactions_response.status_code != 200:
            continue
        reactions = reactions_response.json()
        for reaction in summary.keys():
            summary[reaction] += reactions.get(reaction, 0)

    return summary

def update_voting_results(issues):
    """Update the voting results file."""
    results = {}
    for issue in issues:
        if "labels" in issue and any(label["name"] in ["assessment", "proposal"] for label in issue["labels"]):
            votes = summarize_votes(issue)
            results[issue["title"]] = {
                "url": issue["html_url"],
                "votes": votes
            }

    output_file = "voting_results/voting_summary.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Voting results updated: {output_file}")

if __name__ == "__main__":
    try:
        print("Fetching issues...")
        issues = fetch_issues()
        print("Summarizing votes...")
        update_voting_results(issues)
    except Exception as e:
        print(f"Error: {e}")
