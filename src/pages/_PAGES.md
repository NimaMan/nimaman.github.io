# Pages

Public routes (static output).

## `index.astro` — `/`
The home atlas. Reads `homePage`, `expertise`, `lifeTimeline`, `selectedWork`,
`socialLinks` from `site-data.json`, plus the 3 most recent posts.
Sections, in order:
1. **Hero** — coordinate kicker, Fraunces name + mono `PhD` stamp, statement,
   survey-marker ledger (current work / base / team), compass CTA, corner-ticked
   portrait with a coordinate caption.
2. **Story journey** (`#journey`) — the centrepiece. `lifeTimeline` is reversed
   to most-recent-first; each scene is given a real lat/long coordinate
   (`sceneCoords`) and a `plate` number. Renders the sticky navigator console +
   route rail and the seven full-colour chapter "rooms". Scroll behaviour is
   driven by `public/static/js/home.js`; markup exposes `data-story-*`
   attributes (incl. `data-story-coord`) the script reads.
3. **Current Work** — `expertise` as survey-station plates.
4. **Selected Work** (`#work`) — `selectedWork` as plotted destination rows.
5. **Writing** — 3 featured posts as field-note log cards.
6. **Contact** (`#contact`) — dark "deep water" plate; a route terminus marks
   *Present position*; social links as a mono index.

## `posts/index.astro` — `/posts`
"Index of entries": Fraunces editorial headline + standfirst, an **Article
Series** subsection for Value Chains, then all non-draft posts as ruled
log-entry plates (`getPublicPosts()`).

## `value_chains/index.astro` — `/value_chains`
Landing page for the **Value Chains** essay series. Lists every post whose
frontmatter sets `series: value_chains`, ordered by `series_order` via
`getSeriesPosts()`, each labelled by `series_label` (Part I · Companion · Part
II). Same ruled-plate listing styling as `/posts`. Linked from the articles
index subsection, from each series article's breadcrumb eyebrow, and from its
`SeriesNav` footer.

## `posts/[slug].astro` — `/posts/<slug>`
The reading view. Static-generated per post. Map chrome retreats to the
margins; prose renders in an opaque reading column. Loads MathJax (tex-svg)
only when a post's frontmatter sets `math: true`. Posts in a `series` also get a
breadcrumb eyebrow linking to the series hub and a `SeriesNav` footer that hands
the reader to the other parts.

Coordinates used in the journey are real and are kept accurate (the owner is a
data lead): Qaradag 38.70 N, Tehran 35.69 N, Istanbul 41.01 N, Stuttgart
48.78 N, Eindhoven 51.44 N, Utrecht 52.09 N.
