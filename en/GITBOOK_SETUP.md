# GitBook setup and maintenance

**On this page**

- [1. Repository and bilingual structure](#section-001)
- [2. Connect GitBook](#section-002)
- [3. Reading entry points](#section-003)
- [4. Equations, diagrams, and contents](#section-004)
- [5. Future updates](#section-005)

---

## 1. Repository and bilingual structure <a href="#section-001" id="section-001"></a>

The repository is [honggui/sgd-net-science](https://github.com/honggui/sgd-net-science), using the main branch. zh/ and en/ contain complete Chinese and English editions with corresponding numbered paths. The root README.md is the bilingual gateway, and the root SUMMARY.md includes both languages. Each language directory also has its own README.md, SUMMARY.md, and `.gitbook.yaml`.[GitBook content configuration](https://gitbook.com/docs/docs-as-code/git-sync/content-configuration)

## 2. Connect GitBook <a href="#section-002" id="section-002"></a>

Two configurations are available:

1. **One space with a bilingual table of contents**: connect Git Sync to the repository root and use the root `.gitbook.yaml`. Readers enter through the Chinese and English groups.
2. **Two spaces as language variants**: connect two spaces with zh/ and en/ as their respective content directories, then configure language variants within one documentation site. Each space uses its own SUMMARY.md. This option suits sites that need GitBook's native language selector.

Use GitHub as the source for the initial sync so that an empty space does not overwrite repository content. Do not repeat the same root-directory prefix in multiple settings; inspect the final Project directory and Content mapping paths in GitBook's current interface.[Official GitHub Sync steps](https://gitbook.com/docs/docs-as-code/git-sync/enabling-github-sync), [GitBook language variants](https://gitbook.com/docs/guides/content-organization-and-localization/localize-your-docs-with-variants-in-gitbook)

GitBook account connection and cloud publication must be completed in the relevant account. This repository retains sync-compatible configuration; publishing Pages does not mean that GitBook cloud has been connected.

## 3. Reading entry points <a href="#section-003" id="section-003"></a>

The [online reading home](https://honggui.github.io/sgd-net-science/) offers Chinese and English entry points. GitHub Pages publishes the primary reading site. If a GitBook site is published separately, add its actual generated URL as another entry point. See [publishing maintenance](guide/publishing.md) for deployment and the Wiki entry.

## 4. Equations, diagrams, and contents <a href="#section-004" id="section-004"></a>

Both languages retain GitBook's double-dollar math delimiters, stable explicit anchors, and Mermaid code blocks. English translations preserve equations, component IDs, and reference links; diagram display labels are translated into English. After the first import, check the [rendering test page](guide/rendering-check.md), then sample long equations and complex diagrams.[Math format](https://gitbook.com/docs/create-content/blocks/math-and-tex), [Mermaid format](https://gitbook.com/docs/create-content/blocks/mermaid-blocks)

Numbered pages 13 and 36 introduce the chapters, whose full text resides in docs/chapters/. Each language's SUMMARY.md registers each page in that language only once.

## 5. Future updates <a href="#section-005" id="section-005"></a>

Maintain the source text, contents, and corresponding translation together. Resolve merge differences when using bidirectional GitBook sync; do not replace chapter entry pages with old unsplit documents. See [bilingual maintenance](guide/bilingual.md) for updating translation checks and the publication manifest.

Shared component JSON files live in the root architecture/ directory. The component tables in each reading edition provide localized explanations. For separate GitBook language spaces, cross-space or shared-resource links may require configuration using the actual space URLs; the Pages build handles shared JSON resources automatically.

Do not commit local internal review records, ZIP archives, virtual environments, or build output. See NOTICE for licensing status. Local checks do not replace GitBook cloud previews.

---

[← Previous](NOTICE.md) · [Full contents](SUMMARY.md)
