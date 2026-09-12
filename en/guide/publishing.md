# Online reading and publishing maintenance

This project uses a bilingual GitHub Pages reading site with the GitHub repository as its content source. GitBook can connect to the same content, and GitHub Wiki serves as an entry point.

- [Online reading home](https://honggui.github.io/sgd-net-science/)
- [Source repository](https://github.com/honggui/sgd-net-science)
- [Automatic publication records](https://github.com/honggui/sgd-net-science/actions/workflows/pages.yml)

## 1. Publishing approach

GitHub Wiki uses a separate Git repository and does not directly host GitBook's website interface. This project uses Pages for chapter sidebars, search, language switching, light/dark themes, and mobile reading. The Wiki contains only an entry point, avoiding a second full copy.[GitHub Wiki documentation](https://docs.github.com/en/communities/documenting-your-project-with-wikis/adding-or-editing-wiki-pages), [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## 2. Content sources and rendering

zh/ and en/ contain the complete texts, using identical numbered paths. Each language's SUMMARY.md determines its table-of-contents order. Shared architecture/ JSON, scripts/ build tools, and site-assets/ styling are maintained only once.

Pages uses MkDocs Material. During the build, math delimiters, adjacent equation blocks, and table bars are adapted temporarily, preserving mathematical meaning and the source files. Browser scripts for MathJax and Mermaid come from fixed versions on the jsDelivr CDN, which must be reachable on first reading. Search runs in the browser.[Math rendering configuration](https://squidfunk.github.io/mkdocs-material/reference/math/)

## 3. Automatic publishing and local checks

After a push to main, Actions checks bilingual correspondence, equations, and translation versions, builds both languages, checks generated links, and finally publishes site/. The Source under Settings → Pages should be GitHub Actions. Failed builds are not deployed.[Publishing Pages with GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

Use Python 3.12 or a compatible version locally:

```bash
python -m venv .venv-docs
# After activating the virtual environment:
pip install -r requirements-docs.txt
python scripts/check_bilingual.py
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --directory site
```

Open `http://localhost:8000/` and check the rendering test pages, contents, search, and corresponding-chapter language switches in both editions. Do not commit site/, virtual environments, or temporary directories. Before committing, run `python scripts/update_manifest.py` and inspect git status.

## 4. Legacy links and the Wiki

i18n/legacy-paths.json maps old Chinese paths to the new numbered paths. The site build generates redirects for the old addresses and preserves the browser's chapter anchors. Repository source files have moved to zh/ and en/; new references should use the new paths.

If the Wiki is uninitialized, save the first page through the [Wiki web interface](https://github.com/honggui/sgd-net-science/wiki) before cloning its separate repository. You can use the [Home.md template](https://github.com/honggui/sgd-net-science/blob/main/wiki-entry/Home.md). The template's existence does not mean that the Wiki has been published.

## 5. GitBook and research status

See [GitBook setup](../GITBOOK_SETUP.md) for a single-space bilingual contents layout and multi-space language variants. A live Pages site does not imply a connected GitBook account.

The added tools publish documentation; they are not an SGD-Net training or inference implementation. See [research status](status.md), [NOTICE](../NOTICE.md), and [bilingual maintenance](bilingual.md) for research status, licensing, and translation synchronization respectively.
