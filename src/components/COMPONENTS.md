# Components

Reusable Astro partials shared across pages.

## `SiteNav.astro`
The sticky running head. Props: `activePage: "home" | "posts"`.
- **Wordmark** `Nima Manaf` + a mono `Gazetteer` sub-label (the atlas framing).
- **Nav** rendered as a mono coordinate strip (`Home · Articles · Contact`),
  active item underlined in brass.
- **Plate folio:** on the home page it renders `<span data-plate>Plate 01</span>`
  which `home.js` updates live as the journey scrolls; on other pages it shows a
  static `52.09° N` colophon instead.
- **Profile links** as small hairline glyph buttons (`@`, `in`, `</>`, `✦`).

## `SiteFooter.astro`
Wordmark + tagline (mono) on the left, social links as a mono index on the
right, separated by a hairline. Pure presentation, no props.

Both components read `src/data/site-data.json` (`homePage`, `socialLinks`).
All visual styling lives in `../styles/site.css` (the "Meridian" system).
