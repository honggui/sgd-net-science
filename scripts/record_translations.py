"""Record a reviewed translation pair; never silently mark edits synchronized."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("paths", nargs="*", help="Reviewed paths relative to zh/ and en/")
parser.add_argument("--all", action="store_true", help="Record all pairs after reviewing the full edition")
args = parser.parse_args()
if not args.all and not args.paths:
    parser.error("Specify reviewed paths or --all; this command acknowledges translation review")
subprocess.run([sys.executable, str(ROOT / "scripts/check_bilingual.py"), "--skip-record"], check=True)
record = ROOT / "i18n/translations.json"
data = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {
    "source_language": "zh",
    "target_language": "en",
    "method": "AI-assisted complete translation with structural checks; not independent scientific peer review",
    "pages": [],
}
entries = {item["path"]: item for item in data["pages"]}
if args.all:
    entries = {}
paths = [p.relative_to(ROOT / "zh").as_posix() for p in (ROOT / "zh").rglob("*.md")] if args.all else args.paths
for name in paths:
    name = Path(name).as_posix()
    item = {"path": name}
    for language in ("zh", "en"):
        path = (ROOT / language / name).resolve()
        if not path.is_relative_to(ROOT / language) or not path.is_file():
            raise ValueError(f"Invalid translated path: {language}/{name}")
        item[language + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entries[name] = item
data["pages"] = [entries[name] for name in sorted(entries)]
record.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Recorded {len(paths)} reviewed translation pairs")
