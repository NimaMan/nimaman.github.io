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
The articles index, rendered as a **category tree**. Buckets all non-draft
posts (`getPublicPosts()`) by the top-level segment of their `category`
frontmatter path (`categorySegments()[0]`) and renders one section per
category in a fixed editorial order — Value Chains, Optimization & Learning,
Mathematical Curiosities, Research & Teaching — then unexpected categories
alphabetically, then an "Uncategorized" bucket only if some post lacks a
category. The **Value Chains** section is special: it links out to the
`/value_chains` hub and nests one level deeper by `categorySegments()[1]`
(Evolution of Value Chains · Money), ordering those posts by `series_order`
so the series reads as a sequence; other categories list date-descending.
New posts join the tree through frontmatter alone — nothing is hardcoded.

## `value_chains/index.astro` — `/value_chains`
Landing page for the **Value Chains** essay series, organized by category.
Fetches every post with `series: value_chains` (ordered by `series_order` via
`getSeriesPosts()`) and groups them on the second segment of their `category`
path (`"Value Chains/<group>"` → `categorySegments(post)[1]`). Renders one
section per group in editorial order — **Evolution of Value Chains** (the
framework: The Law · The Pattern) then **Money** (the worked example: Part I ·
Companion · Part II) — each with a one-paragraph intro; unknown groups are
appended so future posts never silently disappear. Cards carry the
`series_label` chip and date. Same ruled-plate listing styling as `/posts`.
Linked from the articles index, from each series article's breadcrumb
eyebrow, and from its `SeriesNav` footer.

## `posts/[slug].astro` — `/posts/<slug>`
The reading view. Static-generated per post. Map chrome retreats to the
margins; prose renders in an opaque reading column. Loads MathJax (tex-svg)
only when a post's frontmatter sets `math: true`. Posts in a `series` also get a
breadcrumb eyebrow linking to the series hub and a `SeriesNav` footer that hands
the reader to the other parts.

Coordinates used in the journey are real and are kept accurate (the owner is a
data lead): Qaradag 38.70 N, Tehran 35.69 N, Istanbul 41.01 N, Stuttgart
48.78 N, Eindhoven 51.44 N, Utrecht 52.09 N.
