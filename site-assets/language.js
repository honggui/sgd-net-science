// Both editions share stable page paths. Keep the current page and anchor.
function syncLanguageLinks() {
  const match = location.pathname.match(/^(.*\/)(zh|en)\/(.*)$/);
  if (!match) return;
  let fragment = location.hash;
  if (fragment) {
    let target;
    try { target = document.getElementById(decodeURIComponent(fragment.slice(1))); }
    catch (_) { target = null; }
    const heading = target && target.closest("h1,h2,h3,h4,h5,h6");
    const stable = heading && heading.querySelector(".language-anchor[id]");
    if (stable) fragment = "#" + stable.id;
  }
  for (const link of document.querySelectorAll(".md-select__link[hreflang]")) {
    const language = link.getAttribute("hreflang");
    if (language === "zh" || language === "en") {
      link.href = match[1] + language + "/" + match[3] + location.search + fragment;
    }
  }
}
document.addEventListener("DOMContentLoaded", syncLanguageLinks);
window.addEventListener("hashchange", syncLanguageLinks);
