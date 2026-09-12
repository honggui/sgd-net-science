"""Rebuild English labels and the root GitBook contents from paired pages."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "入门": "Getting started",
    "研究定位": "Research framing",
    "核心模型架构": "Core model architecture",
    "数学、学习与持续演化": "Mathematics, learning, and continual evolution",
    "领域模型与科学应用": "Domain models and scientific applications",
    "实现与实验验证": "Implementation and experimental validation",
    "附录": "Appendices",
}
chinese = (ROOT / "zh/SUMMARY.md").read_text(encoding="utf-8")
english = []
for line in chinese.splitlines():
    if line.startswith("## "):
        english.append("## " + GROUPS[line[3:]])
        continue
    match = re.fullmatch(r"(\s*\* )\[.+\]\(([^)]+)\)", line)
    if not match:
        english.append(line)
        continue
    path = ROOT / "en" / match[2]
    source = path.read_text(encoding="utf-8-sig")
    title = re.search(r"(?m)^# (.+)$", source)
    if not title:
        raise ValueError(f"No translated title: {path}")
    label = re.sub(r"<[^>]+>", "", title[1]).strip()
    if match[2] == "README.md":
        label = "Home"
    elif re.fullmatch(r"docs/\d{2}\.md", match[2]):
        label = path.stem + " · " + label
    english.append(match[1] + "[" + label + "](" + match[2] + ")")
english = "\n".join(english) + "\n"
(ROOT / "en/SUMMARY.md").write_text(english, encoding="utf-8", newline="\n")
combined = ["# Summary", "", "* [Languages / 语言入口](README.md)", ""]
for language, text in (("zh", chinese), ("en", english)):
    name = "中文" if language == "zh" else "English"
    combined.extend(["## " + name, ""])
    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            line = "## " + name + " · " + line[3:]
        line = re.sub(r"\]\(([^)]+)\)", lambda m: "](" + language + "/" + m[1] + ")", line)
        combined.append(line)
    combined.append("")
(ROOT / "SUMMARY.md").write_text("\n".join(combined).rstrip() + "\n", encoding="utf-8", newline="\n")
print("Synchronized English and combined bilingual tables of contents")
