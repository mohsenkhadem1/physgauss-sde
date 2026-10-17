"""Download one repository-owned Actions artifact using existing Git auth.

Credentials stay in memory and are sent only to api.github.com. Redirected
signed downloads receive no authorization header. Never prints credentials or
signed URLs. Refuses path traversal during ZIP extraction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import urllib.error
import zipfile


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None


def download(run_id, name, output, repository="mohsenkhadem1/physgauss-sde"):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    credential = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        text=True, capture_output=True, env=environment, check=False)
    if credential.returncode: raise RuntimeError("Existing GitHub credential unavailable.")
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token: raise RuntimeError("Existing GitHub credential has no API token.")
    def request(path):
        url = "https://api.github.com/"+path
        return urllib.request.Request(url, headers={"Authorization": "Bearer "+token,
            "Accept": "application/vnd.github+json", "User-Agent": "physgauss-reproducible-study"})
    with urllib.request.urlopen(request(f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100")) as response:
        artifacts = json.load(response)["artifacts"]
    selected = [a for a in artifacts if a["name"] == name and not a["expired"]]
    if len(selected) != 1: raise ValueError("Need exactly one complete, unexpired named artifact.")
    artifact = selected[0]
    opener = urllib.request.build_opener(NoRedirect())
    try: opener.open(request(f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"))
    except urllib.error.HTTPError as response:
        if response.code != 302: raise RuntimeError("GitHub artifact authorization/download failed.") from None
        location = response.headers["Location"]
    else: raise RuntimeError("Expected a signed archive redirect.")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    archive = output.parent/f"{name}-{artifact['id']}.zip"
    # No credentials accompany the storage request.
    with urllib.request.urlopen(location) as response, archive.open("wb") as stream:
        while block := response.read(1024*1024): stream.write(block)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = artifact.get("digest")
    if expected and expected != "sha256:"+digest: raise ValueError("Artifact digest mismatch.")
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (output/member.filename).resolve()
            if not target.is_relative_to(output): raise ValueError("Unsafe archive member.")
            if member.is_dir(): target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as destination:
                    while block := source.read(1024*1024): destination.write(block)
    metadata = {"repository": repository, "workflow_run_id": run_id, "artifact_name": name,
        "artifact_id": artifact["id"], "archive_sha256": digest, "archive_size_bytes": archive.stat().st_size,
        "workflow_url": f"https://github.com/{repository}/actions/runs/{run_id}"}
    (output/"artifact_download.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True); parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); download(args.run, args.name, args.output)
