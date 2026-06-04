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
    transition in `site.css`, no JS animation loop);
  - swaps the console readout text (count, period, place, country, **coord**,
    context, body) from the active chapter's `data-story-*` attributes;
  - updates the running-head `[data-plate]` folio to `Plate NN`.
- Clicking a stop indicator smooth-scrolls to its chapter and sets it active.

It is additive and JS-optional: with JS off, the first chapter/console render
statically and the page is fully usable.
