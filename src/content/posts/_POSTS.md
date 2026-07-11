# Posts

Markdown articles loaded as the `posts` content collection (schema in
`src/content.config.ts`, helpers in `src/lib/posts.ts`). One file = one
article, served at `/posts/<filename-without-.md>`. File names are
descriptive snake_case and double as the URL slug — never rename casually.
Underscore-prefixed files (like this one) are excluded from the collection
by the loader's `**/[^_]*.md` glob, mirroring the `src/pages` convention.

## Frontmatter contract

- `author`, `title`, `date`, `description` — bylines and the excerpt shown on
  listing cards (a description that merely repeats the title is suppressed).
- `math: true` — loads MathJax (tex-svg) on the reading page.
- `draft: true` — hides the post from every listing and from the build.
- `category` — the post's position in the site's category tree: a
  "/"-separated path of display names, root first (e.g. `"Value
  Chains/Money"`). Segment text is used verbatim as headings. `/posts` groups
  on segment 0; the `/value_chains` hub groups on segment 1. Depth is
  unbounded so the tree can grow.
- `series`, `series_order`, `series_label` — membership in a named series
  (currently only `value_chains`), the post's position in the series' reading
  order (globally unique within the series), and the label chip shown on
  cards and breadcrumbs.

## Current category tree

- **Value Chains** (also `series: value_chains`; hub at `/value_chains`)
  - **Evolution of Value Chains** — the framework
    - `the_selection_pressure.md` — The Law (order 0)
    - `the_same_journey_everywhere.md` — The Pattern (order 1)
  - **Money** — the worked example
    - `battle_for_dollar_supremacy.md` — Part I (order 2)
    - `the_price_you_cannot_see.md` — Companion (order 3)
    - `the_spendable_form.md` — Part II (order 4)
- **Optimization & Learning**
  - `learning-to-control-inventory-management-systems.md`
  - `evolution_strategies.md` (draft)
- **Mathematical Curiosities**
  - `eingen_viz.md`, `pi.md`, `random_walk_mean_absorbing_time.md`
- **Research & Teaching**
  - `research_statement.md`, `learning_teaching_philosophy.md`,
    `phd_thesis.md`

## Conventions

- Long-form essays open with an HTML comment stating the OBJECTIVE of the
  piece and its ARGUMENT SPINE (the "algorithm" of the essay), so the file is
  self-describing before the prose starts.
- Series posts cross-link with root-relative URLs (`/posts/<slug>`); external
  claims carry inline links to their sources.
- When adding a series post, keep `series_order` globally unique within the
  series (it drives both the hub ordering and the SeriesNav footer) and give
  the post a `category` path under the series' root category.
