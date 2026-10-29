# Pages

Public routes (static output).

## `index.astro` — `/`
The home atlas. Reads `homePage`, `expertise`, `lifeTimeline`, `selectedWork`,
`socialLinks` from `site-data.json`, plus public posts for the latest essay,
three newest writing cards, and category counts (shared `postCategories()`).
Sections, in order:
1. **Hero** — coordinate kicker, Fraunces name + mono `PhD` stamp, italic
   **Building a Future to Believe In** in a Low → Mid → High Dombra gradient
   (solid ink for print, forced colours, and browsers without text clipping),
   primary **Read the articles** action, secondary
   **See timeline**, automatic latest-essay title/date, survey-marker ledger
   (current work / base / team), and a corner-ticked portrait with coordinates.
   On phones, the reading actions precede the ledger and portrait.
2. **Story journey** (`#journey`) — the centrepiece. `lifeTimeline` is reversed
   to most-recent-first; each scene is given a real lat/long coordinate
   (`sceneCoords`) and a `plate` number. Renders the sticky navigator console +
   route rail and the seven full-colour chapter "rooms". Scroll behaviour is
   driven by `public/static/js/home.js`; markup exposes `data-story-*`
   attributes (incl. `data-story-coord`) the script reads.
   Decorative paths and beacons are clipped to the right gutter. Room content
   stacks when two 18rem columns no longer fit, accommodating longer copy.
3. **Current Work** — `expertise` as survey-station plates.
4. **Selected Work** (`#work`) — `selectedWork` as plotted destination rows.
5. **Writing** — **All articles** action, a category/count jump row linking to
   `/posts#<category>`, and the 3 newest posts as field-note log cards. The heading
   and category strip share one upper rule.
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
(Evolution of Value Chains · Money · Music), ordering those posts by `series_order`
so the series reads as a sequence; other categories list date-descending.
New posts join the tree through frontmatter alone. A category/count jump strip
near the top and stable section IDs (slugified display names) support links from
the home page and article trails. Money and music are the worked examples.
The hero reads **Essays on value chains, music, learning, and mathematical
curiosities.** Its introduction and page metadata cover the same subjects.

## `value_chains/index.astro` — `/value_chains`
Landing page for the **Value Chains** essay series, organized by category.
Fetches every post with `series: value_chains` (ordered by `series_order` via
`getSeriesPosts()`) and groups them on the second segment of their `category`
path (`"Value Chains/<group>"` → `categorySegments(post)[1]`). Renders one
section per group in editorial order — **Evolution of Value Chains** (the
framework: The Law · The Pattern), **Money** (first worked example: Part I ·
Companion · Part II), then **Music** (second worked example: the DAW) — each
with a one-paragraph intro and a display-name section anchor; unknown groups are
appended so future posts never silently disappear. Cards carry the
`series_label` chip and date. Same ruled-plate listing styling as `/posts`.
Linked from the articles index, each series article's category trail, and its
`SeriesNav` footer. Empty groups are omitted until their public posts arrive.

## `posts/[slug].astro` — `/posts/<slug>`
The reading view, static-generated per public post. A category breadcrumb links
to `/posts`, its top-level category anchor, and the Value Chains hub's subgroup
anchor where applicable. Metadata shows date · author · estimated minutes.
`render(post)` supplies the contents headings: at least three h2 sections enable
a sticky sidebar at ≥1200px, with h3 children nested, and a native collapsible
details block at narrower widths. `article.js` progressively highlights the
current section, leaving all items unmarked before the first section is reached;
navigation also works without JS. Permalinks are injected by
the local rehype plugin. The pi and inventory posts preserve their wider figure
layouts and use compact contents only (when eligible).

Prose remains on opaque paper at a comfortable measure. Loads MathJax (tex-svg)
only for `math: true`. A spectrum end mark precedes **Keep reading**: series
posts retain `SeriesNav`; other posts show up to three newest public articles in
the same top-level category. Every article ends with **All articles**.

Coordinates used in the journey are real and are kept accurate (the owner is a
data lead): Qaradag 38.70 N, Tehran 35.69 N, Istanbul 41.01 N, Stuttgart
48.78 N, Eindhoven 51.44 N, Utrecht 52.09 N.
