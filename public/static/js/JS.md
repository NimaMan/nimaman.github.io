<!-- Page script guide. Steps: document each script's selectors, state changes
and native fallbacks so optional enhancements stay accessible and routes keep
working without JavaScript. -->
# Page JavaScript

## `posts-filter.js`
Enhances `/posts` only; exits if `[data-post-listing]` is absent. Reads category
links from `[data-post-filter]`, then filters `[data-post-category]` entries
inside `[data-post-year]` sections using the existing category hash IDs.

On load and `hashchange`, it hides unmatched posts and empty years with `hidden`,
sets `aria-current="true"` on the selected link, and updates the polite live
`[data-post-count]` (e.g. **6 articles in Value Chains · Newest first**). The
**All** link uses `#all`; empty, unknown and malformed hashes also show everything.
It keeps the server-rendered newest-first order and never duplicates entries.
Native links supply URL updates, Back/Forward history, keyboard navigation and
landing targets for incoming home/breadcrumb links. No click interception or
custom scrolling is needed. With JavaScript off, all posts and years stay visible.
`footer-listings.css` explicitly preserves `hidden` against card display rules.

## `home.js`
Drives the home page **story journey** (and nothing else). Vanilla, no deps,
self-invoking; exits immediately if `[data-story-journey]` is absent.

What it does:
- Observes each `.story-chapter` with an `IntersectionObserver`
  (`rootMargin: -35% 0 -35% 0`) and computes the active chapter index.
- `setActive(index)` then:
  - toggles `.is-active` on the active chapter and `.is-active` / `.is-passed`
    on the stop indicators (drives the route-rail colour: brass active, navy
    visited, hollow ahead);
  - sets `--story-route-progress` on the journey, the
    `.story-journey__line-fill` height, and the `.story-route-console__meter`
    width — i.e. the **route fill** (pure CSS width/height, animated by the
    transition in `story-journey.css`, no JS animation loop);
  - swaps the console readout text (count, period, place, country, **coord**,
    context, body) from the active chapter's `data-story-*` attributes;
  - updates the running-head `[data-plate]` folio to `Plate NN`.
- Clicking a stop indicator scrolls to its chapter and sets it active. Smooth
  scrolling is used only for `prefers-reduced-motion: no-preference`.

It is additive and JS-optional: with JS off, the first chapter/console render
statically and the page is fully usable.

## `article.js`
Enhances article contents only; loaded when the rendered essay has at least
three h2 sections. Matches the contents anchors to h2/h3 IDs in `.article-prose`,
then marks the last heading above the sticky header as `aria-current="location"`
in both sidebar and compact lists. Scroll updates are passive and batched through
`requestAnimationFrame`; resize, hash changes, and load also refresh the state.
All items remain unmarked before the first heading reaches the header offset,
including when the reader scrolls back to the essay's introduction.
The script does not intercept links or animate scrolling. CSS uses brass for
the current location. Native anchors and `<details>` work with JavaScript off.
