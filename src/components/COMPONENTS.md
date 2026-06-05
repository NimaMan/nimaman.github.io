# Components

Reusable Astro partials shared across pages.

## `SiteNav.astro`
The sticky running head. Props: `activePage: "home" | "posts" | "value_chains"`.
- **Wordmark** `Nima Manaf` + a mono `Gazetteer` sub-label (the atlas framing).
- **Nav** rendered as a mono coordinate strip (`Home · Articles · Value Chains ·
  Contact`), active item underlined in brass.
- **Plate folio:** on the home page it renders `<span data-plate>Plate 01</span>`
  which `home.js` updates live as the journey scrolls; on other pages it shows a
  static `52.09° N` colophon instead.
- **Profile links** as small hairline glyph buttons (`@`, `in`, `</>`, `✦`).

## `SeriesNav.astro`
Foot-of-article "continue the series" block, rendered only on posts whose
frontmatter declares a `series`. Props: `current: PostEntry`, `siblings:
PostEntry[]` (every entry in the series, via `getSeriesPosts`), and optional
`hubHref` / `hubLabel` (default `/value_chains` · `Value Chains`). Lists the
*other* parts of the series as brass-diamond plates — `series_label`, title,
dek, and a `Read →` cue — so a reader is handed straight to the next essay.

## `SiteFooter.astro`
Wordmark + tagline (mono) on the left, social links as a mono index on the
right, separated by a hairline. Pure presentation, no props.

`SiteNav` and `SiteFooter` read `src/data/site-data.json` (`homePage`,
`socialLinks`); `SeriesNav` reads only its props. All visual styling lives in
`../styles/site.css` (the "Meridian" system).
