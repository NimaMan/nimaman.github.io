# Page JavaScript

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
