# Bilingual documentation structure and maintenance

Chinese is the current source edition of the research documentation, and English is its complete translation. Both editions should express the same assumptions, methods, limitations, and validation plans. Translation and format checks are not independent scientific peer review and do not add implemented capabilities.

## 1. Directory responsibilities

| Path | Responsibility |
|---|---|
| README.md / SUMMARY.md | Bilingual gateways for GitHub and a single-space GitBook setup |
| zh/ | Chinese text, contents, guides, and readable component tables |
| en/ | English edition with a one-to-one correspondence to Chinese numbered paths |
| architecture/ | Component registry JSON shared by both languages |
| i18n/ | Legacy URL mappings and per-page translation records |
| scripts/ / site-assets/ | Shared build tools, checks, and reading interface |
| wiki-entry/ | Wiki home-page entry template |

For example, `zh/docs/36.md` and `en/docs/36.md` introduce the same architecture; `docs/chapters/36/section-20.md` identifies the same chapter in both languages. Titles may be translated or revised while numbered paths and explicit anchors remain stable.

## 2. Translation conventions

Translate the full text, tables, diagram labels, and comments; do not replace full documents with summaries. Preserve TeX payloads, component IDs, code identifiers, reference URLs, and explicit anchors. Diagram display labels may be translated, but edges and computational relationships must not change.

Distinguish research proposals, conditional guarantees, and validated results; do not present candidate methods as validated capabilities. Equivariance and invariance are distinct. An SSM's internal state recurrence is not an implicit solve. Use growth, refinement, observation innovation, stability conditions, and related terms consistently with the [glossary](glossary.md).

## 3. Update a document pair

1. Edit the Chinese page and synchronize the corresponding full English text.
2. For a new page, update both SUMMARY.md files and use the same relative path in each language.
3. Review the meaning and run structural checks. Passing checks does not replace expert content review.
4. Explicitly record the translation version for the pair, then update the publication manifest.

```bash
python scripts/check_bilingual.py --skip-record
python scripts/record_translations.py docs/36.md
python scripts/update_manifest.py
python scripts/check_bilingual.py
python scripts/build_site.py
python scripts/check_site.py
```

Paths passed to `record_translations.py` are relative to zh/ and en/. Run it only after confirming synchronization; use `--all` for a whole-edition review. i18n/translations.json stores both file hashes. CI rejects unrecorded changes so that English does not silently become outdated after Chinese edits. Updating the publication manifest does not automatically acknowledge translation synchronization.

## 4. Check coverage and limitations

Automatic checks cover page completeness, equation payloads, heading-level counts, anchors, references, fenced-block languages, remaining Chinese prose, and local links. After the build, equations and generated links are checked again. Browser checks cover equations, Mermaid, search, mobile layout, and language switching.

These checks can detect structural omissions; they cannot establish that every scientific meaning was translated correctly. Important terminology and arguments should continue to receive domain-expert review. Apply corrections to both languages.

## 5. Publishing and reading

The [online reading home](https://honggui.github.io/sgd-net-science/) provides Chinese and English entry points. Each edition has its own contents and search; switching language preserves the numbered chapter. IDs in shared registries do not change with the language. See [publishing maintenance](publishing.md) for deployment and [GitBook setup](../GITBOOK_SETUP.md) for GitBook configuration.
