// Convert syntax-highlighted Mermaid fences without changing the source book.
document.addEventListener("DOMContentLoaded", async () => {
  const blocks = document.querySelectorAll('code.language-mermaid');
  if (!blocks.length) return;
  const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.esm.min.mjs");
  mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "default" });
  for (const code of blocks) {
    const diagram = document.createElement("div");
    diagram.className = "mermaid";
    diagram.textContent = code.textContent;
    code.closest("pre").replaceWith(diagram);
  }
  await mermaid.run({ querySelector: ".mermaid" });
});
