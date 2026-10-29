# Layouts

## `BaseLayout.astro`
The HTML shell for every page. Props: `title`, `description`, `bodyClass`.
- Imports `../styles/site.css`, the entry for eight ordered Meridian modules
  (tokens through responsive overrides; see `styles/DESIGN_SYSTEM.md`).
- Preconnects to Google Fonts and loads the three families the system uses:
  **Fraunces** (variable `ital,opsz,wght` — display/headlines), **Source Serif
  4** (`ital` + weights — reading body), and **IBM Plex Mono** (instrument
  voice). Space Grotesk was removed in the redesign.
- Exposes named slots: `head` (per-page `<head>` extras, e.g. the MathJax config
  on math posts) and `scripts` (per-page JS, e.g. `home.js` on the index and
  `article.js` on essays with contents).

`bodyClass="public-page"` enables the faint paper graticule background.
