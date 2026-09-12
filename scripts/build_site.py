"""Build Pages from the same Markdown and SUMMARY used by GitBook."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def navigation():
    nav, group, children = [], None, None
    for line in (ROOT / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            group = []
            nav.append({line[3:]: group})
            children = None
        match = re.fullmatch(r"( *)\* \[(.+)\]\(([^)]+)\)", line)
        if not match:
            continue
        indent, title, path = match.groups()
        target = (ROOT / path).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Invalid SUMMARY target: {path}")
        dest = nav if group is None else group
        if indent:
            if children is None:
                label, parent = next(iter(dest[-1].items()))
                children = [{"概览": parent}]
                dest[-1] = {label: children}
            children.append({title: path})
        else:
            dest.append({title: path})
            children = None
    return nav


def pages_math(source):
    """Adapt GitBook inline $$ to Arithmatex, leaving TeX and code unchanged."""
    result, fence, block_math = [], None, False
    for line in source.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            result.append(line)
            continue
        if fence is not None:
            result.append(line)
            continue
        if line.strip() == "$$":
            # Adjacent GitBook blocks need blank boundaries in Python-Markdown.
            result.append(line + "\n" if block_math else "\n" + line)
            block_math = not block_math
            continue
        if block_math:
            result.append(line)
            continue
        # Protect inline code, including spans delimited by multiple backticks.
        parts = re.split(r"(`+[^`]*`+)", line)
        def inline(match):
            tex = match[1]
            if line.lstrip().startswith("|"):
                # Raw math bars otherwise split Markdown table cells.
                tex = re.sub(r"(?<!\\)\|", lambda _: r"\vert ", tex)
            return r"\(" + tex + r"\)"
        for index in range(0, len(parts), 2):
            parts[index] = re.sub(r"\$\$(.+?)\$\$", inline, parts[index])
        result.append("".join(parts))
    return "".join(result)


def main():
    with tempfile.TemporaryDirectory(prefix=".site-source-", dir=ROOT) as folder:
        work = Path(folder)
        content = work / "content"
        content.mkdir()
        # Explicit public-content allowlist; never scan arbitrary workspace files.
        sources = list(ROOT.glob("*.md"))
        for directory in ("docs", "guide", "architecture"):
            sources.extend((ROOT / directory).rglob("*.md"))
        sources.extend((ROOT / "architecture").glob("*.json"))
        for source in sources:
            target = content / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".md":
                target.write_text(pages_math(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
            else:
                shutil.copyfile(source, target)
        shutil.copytree(ROOT / "site-assets", content / "assets", dirs_exist_ok=True)
        config = {
            "site_name": "SGD-Net Science",
            "site_description": "Architecture Research for Scientific and Physical AI · 科学与物理 AI 架构研究",
            "site_url": "https://honggui.github.io/sgd-net-science/",
            "repo_url": "https://github.com/honggui/sgd-net-science",
            "repo_name": "honggui/sgd-net-science",
            "edit_uri": "edit/main/",
            "docs_dir": str(content),
            "site_dir": str(ROOT / "site"),
            "theme": {
                "name": "material", "language": "zh", "font": False,
                "palette": [
                    {"scheme": "default", "primary": "indigo", "toggle": {"icon": "material/brightness-7", "name": "切换深色"}},
                    {"scheme": "slate", "primary": "indigo", "toggle": {"icon": "material/brightness-4", "name": "切换浅色"}},
                ],
                "features": ["navigation.top", "navigation.footer", "navigation.indexes", "search.highlight", "search.suggest", "content.code.copy"],
            },
            "plugins": [{"search": {"lang": ["zh", "en"]}}],
            "markdown_extensions": [
                "admonition", "attr_list", "tables", "footnotes",
                {"toc": {"permalink": True}},
                {"pymdownx.arithmatex": {"generic": True}},
                "fenced_code",
            ],
            "extra_javascript": ["assets/math.js", "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js", "assets/diagrams.js"],
            "extra_css": ["assets/reading.css"],
            "nav": navigation(),
            "validation": {"links": {"anchors": "info"}},
        }
        config_path = work / "mkdocs.yml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(config_path)], check=True)


if __name__ == "__main__":
    main()
