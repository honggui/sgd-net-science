"""Regression checks for bilingual publication and math conversion."""
import unittest

import markdown

from build_site import page_url, pages_math
from check_bilingual import inspect_pair


def render(text):
    return markdown.markdown(pages_math(text), extensions=["tables", "pymdownx.arithmatex", "fenced_code"], extension_configs={"pymdownx.arithmatex": {"generic": True}})


class PublicationChecks(unittest.TestCase):
    def test_adjacent_equation_blocks_remain_separate(self):
        html = render("$$\nx=1\n$$\n$$\ny=2\n$$\n")
        self.assertEqual(html.count('class="arithmatex"'), 2)
        self.assertNotIn("$$", html)

    def test_math_bars_do_not_split_table_cells(self):
        html = render("| Variable | Value |\n|---|---|\n| Norm | $$|x|$$ |\n")
        self.assertEqual(html.count("<td>"), 2)
        self.assertEqual(html.count('class="arithmatex"'), 1)

    def test_code_is_not_rewritten_as_math(self):
        text = "```python\nlabel = '$$x$$'\n```\n\nUse `$$x$$` as a delimiter example.\n"
        self.assertEqual(pages_math(text), text)

    def test_changed_equation_or_missing_section_fails(self):
        zh = '# 方法\n## 约束 <a id="s1"></a>\n$$x=1$$\n'
        good = '# Method\n## Constraint <a id="s1"></a>\n$$x=1$$\n'
        self.assertFalse(inspect_pair(zh, good))
        self.assertIn("formula payloads differ", inspect_pair(zh, good.replace("x=1", "x=2")))
        self.assertIn("heading levels differ", inspect_pair(zh, good.replace("## Constraint", "Constraint")))

    def test_missing_reference_or_table_row_fails(self):
        zh = "[论文](https://example.org/paper)\n| A | B |\n|---|---|\n| 甲 | 乙 |\n"
        en = "[Paper](https://example.org/paper)\n| A | B |\n|---|---|\n| First | Second |\n"
        self.assertFalse(inspect_pair(zh, en))
        self.assertIn("external link targets differ", inspect_pair(zh, en.replace("https://example.org/paper", "https://example.org/other")))
        self.assertIn("table row counts differ", inspect_pair(zh, en.replace("| First | Second |\n", "")))

    def test_home_and_chapter_paths(self):
        self.assertEqual(page_url("README.md"), "")
        self.assertEqual(page_url("architecture/README.md"), "architecture/")
        self.assertEqual(page_url("docs/36.md"), "docs/36/")
        self.assertEqual(page_url("docs/chapters/36/section-20.md"), "docs/chapters/36/section-20/")

    def test_wrong_existing_chapter_is_not_a_translation(self):
        self.assertIn("local link targets differ", inspect_pair("[Next](03.md)", "[Next](04.md)"))

    def test_heading_order_supports_same_section_language_switch(self):
        self.assertIn("heading order differs", inspect_pair("# Title\n## A\n### B\n", "# Title\n### B\n## A\n"))


if __name__ == "__main__":
    unittest.main()
