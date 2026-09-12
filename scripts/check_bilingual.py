"""Fail publication when translations, equations, anchors, or links diverge."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def without_code(text):
    return re.sub(r"(?ms)^```.*?^```[^\n]*$", "", text)


def equations(text):
    return Counter(re.findall(r"\$\$(.*?)\$\$", without_code(text), re.S))


def inspect_pair(chinese, english):
    problems = []
    def local_links(text):
        return Counter(link for link in re.findall(r"\]\(([^)]+)\)", without_code(text))
                       if not urlsplit(link).scheme and not urlsplit(link).netloc)
    if local_links(chinese) != local_links(english):
        problems.append("local link targets differ")
    if re.findall(r"(?m)^(#{1,6}) ", without_code(chinese)) != re.findall(r"(?m)^(#{1,6}) ", without_code(english)):
        problems.append("heading order differs")
    if equations(chinese) != equations(english):
        problems.append("formula payloads differ")
    for label, pattern in (
        ("explicit anchors", r'\bid="([^"]+)"'),
        ("heading levels", r"(?m)^(#{1,6}) "),
        ("fenced block languages", r"(?m)^```([^`\n]*)$"),
        ("external link targets", r"\]\((https?://[^)]+)\)"),
        ("footnote identifiers", r"\[\^([^\]]+)\]"),
    ):
        if Counter(re.findall(pattern, chinese)) != Counter(re.findall(pattern, english)):
            problems.append(label + " differ")
    if len(re.findall(r"(?m)^\|", without_code(chinese))) != len(re.findall(r"(?m)^\|", without_code(english))):
        problems.append("table row counts differ")
    # Untranslated prose is easy to miss in long tables and navigation labels.
    prose = without_code(english)
    prose = re.sub(r"\$\$.*?\$\$", "", prose, flags=re.S)
    prose = re.sub(r"https?://\S+", "", prose)
    prose = re.sub(r"`+[^`\n]*`+", "", prose)
    if re.search(r"[\u3400-\u9fff]", prose):
        problems.append("Chinese prose remains outside code/math")
    diagrams = re.findall(r"(?ms)^```mermaid\n(.*?)^```", english)
    if any(re.search(r"[\u3400-\u9fff]", diagram) for diagram in diagrams):
        problems.append("Chinese display text remains in Mermaid")
    return problems


def main():
    errors = []
    chinese = {p.relative_to(ROOT / "zh").as_posix(): p for p in (ROOT / "zh").rglob("*.md")}
    english = {p.relative_to(ROOT / "en").as_posix(): p for p in (ROOT / "en").rglob("*.md")}
    if chinese.keys() != english.keys():
        errors.append(f"Missing English: {sorted(chinese.keys()-english.keys())}; missing Chinese: {sorted(english.keys()-chinese.keys())}")
    orders = {}
    for language, files in (("zh", chinese), ("en", english)):
        summary = ROOT / language / "SUMMARY.md"
        if summary.exists():
            order = re.findall(r"(?m)^\s*\* \[.+\]\(([^)]+)\)$", summary.read_text(encoding="utf-8"))
            orders[language] = order
            if len(order) != len(set(order)) or set(order) != files.keys() - {"SUMMARY.md"}:
                errors.append(language + ": SUMMARY must list every reading page exactly once")
    if "zh" in orders and "en" in orders and orders["zh"] != orders["en"]:
        errors.append("Chinese and English chapter order differs")
    for name in sorted(chinese.keys() & english.keys()):
        zh = chinese[name].read_text(encoding="utf-8")
        en = english[name].read_text(encoding="utf-8")
        errors.extend(name + ": " + issue for issue in inspect_pair(zh, en))
    for path in [*chinese.values(), *english.values()]:
        text = without_code(path.read_text(encoding="utf-8"))
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        for link in re.findall(r"\]\(([^)]+)\)", text):
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            if not url.path.endswith((".md", ".json")):
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if not target.is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing {link}")
            elif target.suffix == ".md" and url.fragment:
                anchors = re.findall(r'\bid="([^"]+)"', target.read_text(encoding="utf-8"))
                if unquote(url.fragment) not in anchors:
                    errors.append(f"{path.relative_to(ROOT)}: missing anchor {link}")
    record = ROOT / "i18n/translations.json"
    if record.exists() and "--skip-record" not in sys.argv:
        recorded = json.loads(record.read_text(encoding="utf-8"))["pages"]
        if {item["path"] for item in recorded} != chinese.keys():
            errors.append("Translation record does not cover the current page set")
        for item in recorded:
            for lang in ("zh", "en"):
                path = ROOT / lang / item["path"]
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != item[lang + "_sha256"]:
                    errors.append(f"{lang}/{item['path']}: changed since translation review; synchronize both editions and refresh translation record")
    elif not record.exists() and "--skip-record" not in sys.argv:
        errors.append("Translation review record is missing")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Bilingual checks passed: {len(chinese)} matched page pairs; formulas, headings, anchors, references and local links")


if __name__ == "__main__":
    main()
