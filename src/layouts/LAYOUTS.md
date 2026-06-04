# Layouts

## `BaseLayout.astro`
The HTML shell for every page. Props: `title`, `description`, `bodyClass`.
- Imports the single stylesheet `../styles/site.css` (the "Meridian" system).
- Preconnects to Google Fonts and loads the three families the system uses:
  **Fraunces** (variable `ital,opsz,wght` — display/headlines), **Source Serif
  4** (`ital` + weights — reading body), and **IBM Plex Mono** (instrument
  voice). Space Grotesk was removed in the redesign.
- Exposes named slots: `head` (per-page `<head>` extras, e.g. the MathJax config
  on math posts) and `scripts` (per-page JS, e.g. `home.js` on the index).

`bodyClass="public-page"` enables the faint paper graticule background.
