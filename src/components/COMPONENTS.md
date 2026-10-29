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

## `SeriesNav.astro`
Foot-of-article "continue the series" block, rendered only on posts whose
frontmatter declares a `series`. Props: `current: PostEntry`, `siblings:
PostEntry[]` (every entry in the series, via `getSeriesPosts`), and optional
`hubHref` / `hubLabel` (default `/value_chains` · `Value Chains`). Lists the
*other* parts of the series as brass-diamond plates — `series_label`, title,
dek, and a `Read →` cue — so a reader is handed straight to the next essay.

## `SiteFooter.astro`
Spectrum signature, wordmark + tagline (mono) on the left, social links as a mono index on the
right, separated by a hairline. Pure presentation, no props.

## `SpectrumMark.astro`
Decorative Dombra signature: three 2px bars, left-aligned, with High violet on
top (60), Mid green (77), Low red at the bottom (86). Props: optional `onDark`
selects the bright dark-surface tokens; the default uses deep paper shades.
Used only in the article end mark and footer; the hero's signature text carries
the palette there. It has `aria-hidden` because
it conveys no navigation, category, or status information.

## `ArticleContents.astro`
Props: grouped `sections` from `articleContents()` and `variant` (`sidebar` or
`compact`). Renders nested native anchors inside either an aside or `<details>`.
CSS chooses the visible layout; `article.js` adds `aria-current="location"` to
matching links in both versions. The page decides whether the essay qualifies.
No link is current until the first section reaches the sticky header offset.

`SiteNav` and `SiteFooter` read `src/data/site-data.json` (`homePage`,
`socialLinks`); `SeriesNav`, `ArticleContents`, and `SpectrumMark` read only
their props. Shared styling lives in the ordered modules imported by
`../styles/site.css` (the "Meridian" system).
