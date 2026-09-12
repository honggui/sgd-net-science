"""Check generated local links and preservation of every source equation."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.math = set(), [], []
        self.active = None
        self.mermaid = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        classes = attrs.get("class", "").split()
        if "arithmatex" in classes:
            self.active = []
        if "language-mermaid" in classes:
            self.mermaid += 1

    def handle_data(self, data):
        if self.active is not None:
            self.active.append(data)

    def handle_endtag(self, tag):
        if self.active is not None and tag in ("span", "div"):
            self.math.append("".join(self.active))
            self.active = None


def canonical(tex):
    return tex.strip().replace(r"\vert ", "|")


pages = {p.resolve(): Page(p.read_text(encoding="utf-8")) for p in SITE.rglob("*.html")}
errors, checked = [], 0
for path, page in pages.items():
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc or link.startswith("/") or not link:
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        if target.is_dir():
            target /= "index.html"
        checked += 1
        if not target.exists():
            errors.append(f"{path.relative_to(SITE)}: missing {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f"{path.relative_to(SITE)}: missing anchor {link}")

sources = list(ROOT.glob("*.md"))
for directory in ("docs", "guide", "architecture"):
    sources.extend((ROOT / directory).rglob("*.md"))
for source in sources:
    relative = source.relative_to(ROOT)
    target = SITE / relative.parent / "index.html" if source.name == "README.md" else SITE / relative.with_suffix("") / "index.html"
    text = source.read_text(encoding="utf-8")
    # Delimiters mentioned as code are not equations.
    text = re.sub(r"(?ms)^```.*?^```[^\n]*$", "", text)
    text = re.sub(r"`+[^`\n]*`+", "", text)
    expected = Counter(canonical(m) for m in re.findall(r"\$\$(.*?)\$\$", text, re.S))
    actual = Counter(canonical(m[2:-2]) for m in pages[target.resolve()].math)
    if expected != actual:
        errors.append(f"{relative}: equation mismatch; missing={list((expected-actual).elements())[:2]}")

if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(pages)} HTML pages, {checked} local links, "
      f"{sum(len(p.math) for p in pages.values())} equations, "
      f"{sum(p.mermaid for p in pages.values())} Mermaid diagrams")
