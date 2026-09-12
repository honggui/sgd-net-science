"""Build Pages from the same Markdown and SUMMARY used by GitBook."""
from pathlib import Path
import html
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://honggui.github.io/sgd-net-science/"


def navigation(language):
    nav, group, children = [], None, None
    for line in (ROOT / language / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            group = []
            nav.append({line[3:]: group})
            children = None
        match = re.fullmatch(r"( *)\* \[(.+)\]\(([^)]+)\)", line)
        if not match:
            continue
        indent, title, path = match.groups()
        target = (ROOT / language / path).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Invalid SUMMARY target: {path}")
        dest = nav if group is None else group
        if indent:
            if children is None:
                label, parent = next(iter(dest[-1].items()))
                children = [{"概览" if language == "zh" else "Overview": parent}]
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


def page_url(path):
    """MkDocs output path, including its special README-to-index convention."""
    path = Path(path)
    return (path.parent.as_posix().rstrip(".") + "/").lstrip("/") if path.name == "README.md" else path.with_suffix("").as_posix() + "/"


def stage_markdown(source, language):
    relative = source.relative_to(ROOT / language)
    text = source.read_text(encoding="utf-8")
    def link(match):
        url = urlsplit(match[1])
        if url.scheme or url.netloc or not url.path.endswith((".md", ".json")):
            return match[0]
        target = (source.parent / url.path).resolve()
        fragment = "#" + url.fragment if url.fragment else ""
        if target.is_relative_to(ROOT / "architecture") and target.suffix == ".json":
            # Shared canonical registries are copied into each generated edition.
            local = "architecture/" + target.name
            return "](" + posixpath.relpath(local, relative.parent.as_posix()) + fragment + ")"
        other = "en" if language == "zh" else "zh"
        if target.is_relative_to(ROOT / other):
            return "](" + SITE_URL + other + "/" + page_url(target.relative_to(ROOT / other)) + fragment + ")"
        return match[0]
    text = re.sub(r"\]\(([^)]+)\)", link, text)
    # Keep translated TOC fragments interoperable without replacing old IDs.
    lines, fence, heading_index = [], False, 0
    for line in text.splitlines(keepends=True):
        if re.match(r"^```", line):
            fence = not fence
        if not fence and re.match(r"^#{1,6} ", line):
            heading_index += 1
            line = line.rstrip("\n") + f' <span class="language-anchor" id="language-heading-{heading_index:03d}"></span>\n'
        lines.append(line)
    text = "".join(lines)
    other = "en" if language == "zh" else "zh"
    title = "Read this page in English" if language == "zh" else "Read this page in Chinese"
    text += "\n\n---\n\n[" + title + "](" + SITE_URL + other + "/" + page_url(relative) + ")\n"
    return pages_math(text)


def build_edition(language, site):
    with tempfile.TemporaryDirectory(prefix=".site-source-", dir=ROOT) as folder:
        work = Path(folder)
        content = work / "content"
        content.mkdir()
        for source in (ROOT / language).rglob("*.md"):
            target = content / source.relative_to(ROOT / language)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(stage_markdown(source, language), encoding="utf-8", newline="\n")
        for source in (ROOT / "architecture").glob("*.json"):
            target = content / "architecture" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copytree(ROOT / "site-assets", content / "assets", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("landing.html"))
        chinese = language == "zh"
        config = {
            "site_name": "SGD-Net Science",
            "site_description": "科学与物理 AI 架构研究" if chinese else "Architecture Research for Scientific and Physical AI",
            "site_url": SITE_URL + language + "/",
            "repo_url": "https://github.com/honggui/sgd-net-science",
            "repo_name": "honggui/sgd-net-science",
            "edit_uri": "edit/main/" + language + "/",
            "docs_dir": str(content),
            "site_dir": str(site / language),
            "theme": {
                "name": "material", "language": language, "font": False,
                "palette": [
                    {"scheme": "default", "primary": "indigo", "toggle": {"icon": "material/brightness-7", "name": "切换深色" if chinese else "Switch to dark mode"}},
                    {"scheme": "slate", "primary": "indigo", "toggle": {"icon": "material/brightness-4", "name": "切换浅色" if chinese else "Switch to light mode"}},
                ],
                "features": ["navigation.top", "navigation.footer", "navigation.indexes", "search.highlight", "search.suggest", "content.code.copy"],
            },
            "extra": {"alternate": [
                {"name": "中文", "link": SITE_URL + "zh/", "lang": "zh"},
                {"name": "English", "link": SITE_URL + "en/", "lang": "en"},
            ]},
            "plugins": [{"search": {"lang": ["zh", "en"] if chinese else ["en"]}}],
            "markdown_extensions": [
                "admonition", "attr_list", "tables", "footnotes",
                {"toc": {"permalink": True}},
                {"pymdownx.arithmatex": {"generic": True}},
                "fenced_code",
            ],
            "extra_javascript": ["assets/math.js", "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js", "assets/diagrams.js", "assets/language.js"],
            "extra_css": ["assets/reading.css"],
            "nav": navigation(language),
            "validation": {"links": {"anchors": "info"}},
        }
        config_path = work / "mkdocs.yml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(config_path)], check=True)


def write_redirect(path, destination):
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = html.escape(destination, quote=True)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="refresh" content="0;url=' + escaped + '">'
        '<link rel="canonical" href="' + escaped + '">'
        '<title>SGD-Net Science — page moved</title>'
        '<p><a href="' + escaped + '">Continue reading / 继续阅读</a></p>'
        '<script>location.replace(' + json.dumps(destination) + '+location.search+location.hash)</script></html>',
        encoding="utf-8", newline="\n",
    )


def main():
    site = (ROOT / "site").resolve()
    if site.parent != ROOT or site.name != "site":
        raise ValueError("Refusing to clean a build directory outside this repository")
    if site.exists():
        shutil.rmtree(site)
    for language in ("zh", "en"):
        build_edition(language, site)
    shutil.copyfile(ROOT / "site-assets/landing.html", site / "index.html")
    shutil.copytree(ROOT / "architecture", site / "architecture", ignore=shutil.ignore_patterns("*.md"), dirs_exist_ok=True)
    legacy = json.loads((ROOT / "i18n/legacy-paths.json").read_text(encoding="utf-8"))
    for old, new in legacy.items():
        if old == "README.md":
            continue
        write_redirect(site / page_url(old) / "index.html", SITE_URL + "zh/" + page_url(new))
    (site / ".nojekyll").touch()
    print("Built Chinese and English editions with legacy URL redirects")


if __name__ == "__main__":
    main()
