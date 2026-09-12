"""Refresh the publication inventory from Git-visible files (not build output)."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = root / "manifest.json"
data = json.loads(manifest.read_text(encoding="utf-8"))
data["edition"] = "bilingual-gitbook-pages-20260912"
data["languages"] = ["zh", "en"]
data["reading_pages_per_language"] = {
    lang: len(re.findall(r"^\s*\* \[.+\]\(.+\)$", (root / lang / "SUMMARY.md").read_text(encoding="utf-8"), re.M))
    for lang in data["languages"]
}
data["reading_pages"] = sum(data["reading_pages_per_language"].values())
data["site_url"] = "https://honggui.github.io/sgd-net-science/"
result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root, capture_output=True, check=True)
paths = sorted(set(result.stdout.decode("utf-8").strip("\0").split("\0")))
data["files"] = []
for name in paths:
    if name == "manifest.json":
        continue
    path = root / name
    if not path.is_file():
        continue
    raw = path.read_bytes()
    data["files"].append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Manifest: {len(data['files'])} files; {data['reading_pages']} reading pages")
