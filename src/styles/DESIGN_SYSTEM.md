<!-- Meridian style guide. Steps: define the visual language and tokens, map
the CSS modules, then record layout and interaction rules so changes preserve
the atlas identity and readable desktop/mobile pages. -->
# Styles — the "Meridian" design system

This folder holds the ordered CSS modules that drive the entire public site.
There is no CSS framework; hand-authored vanilla CSS is loaded once via
`BaseLayout.astro`, through the `site.css` entry stylesheet.

## Objective

Present Nima Manaf — someone who builds optimization, risk, and decision
systems — through the visual language of **measurement**. The site reads as a
**precision survey atlas of one working life**: the route from Qaradag to
Utrecht (five countries, four disciplines) is the navigational spine of the
page. Cartography *is* the credibility metaphor.

The look had to (a) be striking and unmistakably professional, and (b) be
first-class at both a wide desktop width and a 390px phone. Both were explicit
requirements.

## Signature device

**The route line + benchmark glyph.** A continuous cartographic thread runs the
page: it drops out of the hero CTA, becomes the spine of the journey rail
(solid brass = travelled, dashed navy = ahead), and terminates at the contact
band in a brass benchmark labelled *Present position*. The same survey glyph —
a small square rotated 45° (a benchmark/place-mark) — marks every kicker,
journey stop, prose `h2`, and active state. **Brass** remains the single rationed
interactive accent, used like a compass needle. Dombra's small spectrum signature
adds provenance without changing the atlas identity.

## Tokens (`:root`)

| Role | Token | Value |
| --- | --- | --- |
| Page paper | `--paper` | `#f1eadb` |
| Raised plate (cards, prose) | `--paper-high` | `#fbf6ec` |
| Alternate stock / soft section | `--paper-tint` | `#ece3d2` |
| Primary ink / display | `--ink` | `#16202b` |
| Secondary text / coordinates | `--graphite` | `#52596a` |
| Brand / links / route-ahead | `--navy` | `#0b4f6c` |
| Dark bands (contact, code, console) | `--abyssal` | `#0a1a24` |
| **Rationed accent (active only)** | `--brass` | `#b5611f` |
| Quiet secondary accent | `--verdigris` | `#3e7e73` |
| Text on dark | `--cream` | `#ece3d2` |
| Structural lines | `--hairline` / `--hairline-strong` | rgba ink |

### Dombra signature palette

| Band | Paper token / shade | Dark-surface token / shade |
| --- | --- | --- |
| Low | `--band-low` · `#b43d30` | `--band-low-on-dark` · `#ed6a5a` |
| Mid | `--band-mid` · `#257a43` | `--band-mid-on-dark` · `#72c787` |
| High | `--band-high` · `#6941ad` | `--band-high-on-dark` · `#a58ae6` |
| Strings | `--band-strings` · `#18222e` | `--band-strings-on-dark` · `#f1eddf` |

The bands are a set: **High violet above Mid green above Low red**, left-aligned
and lengthening downward at **60:77:86**. Never use these colours for status,
success/error, or categories. Use deep shades on paper, bright shades on dark
surfaces, and keep brass for actions and active navigation. The reusable
`SpectrumMark` uses 2px bars and 1px gaps (8px total) at the article end and
footer. Optional `onDark` selects the dark-surface set. The hero carries the
palette once, in **Building a Future to Believe In**: an italic Fraunces
signature at optical size 72, with a smooth Low red → Mid green → High violet
text gradient. Every gradient stop and transition clears 3:1 on paper at its
display size. Solid ink is the default without text clipping support, in print,
and in forced colours. The phrase balances into at most two lines at 390px.
A soft violet
`--band-high-soft` (`#e7dde3`, violet mixed into paper at roughly 14%) provides
an opaque text-selection fill that also keeps ink readable on dark surfaces.
Small brass text and hover states use `--brass-ink` on paper (5.33:1 or better)
or `--brass-on-dark` on dark bands; `--brass` remains the glyph/line colour.

The owner's explicit room treatment is the exception to the small signature:
**Qaradag** uses a `#6941ad` → `#482773` violet gradient, `#a58ae6` decorative
beacons, and ivory strings/lines. Small body/meta text uses full-opacity cream
instead of the lower-contrast violet accent. The active mobile console and
Qaradag chip use the same violet; route progress stays brass.
Every room clips its decorative route and beacon into the outer right gutter,
clear of the content even as copy grows. Room columns fit automatically, stacking
when two 18rem columns no longer fit; periods remain in the readable cartouche.

**Type:** `Fraunces` (variable `opsz`/`wght`/`ital`) for the name & all
headlines — an engraved literary serif used only at large optical sizes;
`Source Serif 4` for long-form reading; `IBM Plex Mono` (tracked, uppercase,
`tabular-nums`) as the instrument voice for all coordinates, labels, eyebrows,
meta, and code. (Space Grotesk was removed.)

**Geometry:** sharp `4px` radii, `--grid: 30px` graticule module, content shell
`min(1180px, 100% - 2.5rem)`.

## How the stylesheet is organised

`site.css` imports these files in the original cascade order:

| File | Responsibility |
| --- | --- |
| `tokens.css` | Palette, typography, geometry; original design rationale |
| `base-primitives.css` | Document defaults, paper graticule, labels, buttons, registration frames |
| `header-hero.css` | Running head, home hero, shared section scaffold |
| `story-journey.css` | Sticky console, route rail, stop indicators, chapter rooms |
| `home-sections.css` | Current work, selected work, writing cards, contact |
| `footer-listings.css` | Footer, archive and hub heroes, category navigation, dated archive rows and filter states |
| `article-reading.css` | Reading plate, prose, article navigation, embedded atlas figures |
| `motion-responsive.css` | Optional motion and viewport overrides |

The initial split was verified byte-for-byte: concatenating the modules in this
order, removing only their newly added header comments, exactly reproduced
`git show HEAD:src/styles/site.css` (empty diff, before feature changes).

## Things that must not break

- **The journey scroll interaction.** `home.js` toggles `.is-active` /
  `.is-passed` on `.story-chapter` / `.story-stop-indicator`, sets the
  `.story-journey__line-fill` height and `.story-route-console__meter` width
  (the route fill), and writes the running-head `[data-plate]` folio + console
  coordinate readout. The CSS only reskins those classes; it never refactors the
  observer/sticky logic. The route fill is pure CSS height driven by JS — no new
  animation code.
- **Per-chapter rooms** read `.story-chapter--<scene>` to set `--room` (a
  desaturated dark tint) and `--room-accent` (a brighter in-room accent). Body
  text in rooms is always cream for contrast.
- **Reading comfort.** The article prose column is opaque `--paper-high` so the
  page graticule never shows under text; equations (`mjx-container`), `pre`, and
  tables are `overflow-x:auto` so they never break the column.
  Long essays have a separate contents sidebar, preserving the 66ch prose
  measure. Headings have scroll offsets and labelled hover/focus permalinks.
  Pi and inventory retain their wider custom figure plates and compact contents.
  Contents has no current item before the first heading reaches the header offset.
  Atlas captions pair the industry and quiet italic serif want line with a
  wrapping verdict pill in a separate bounded column. Narrow atlas timelines
  wrap long chronology labels inside their date column. The home Writing heading
  supplies its category strip's upper rule; the archive strip keeps both rules.
- **Chronological archive.** `/posts` uses a single column of ruled article cards
  under year headings. Dates occupy a quiet left column on wide screens and
  stack above content at ≤760px. Category paths stay graphite, optional series
  labels use neutral outlined chips, and the active category link uses a brass
  underline plus `aria-current`. Category controls wrap with 44px minimum targets.
  Filtering must honor `hidden` for both entries and empty years, including their
  layout space, while preserving the date order and shared borders between the
  remaining entries. The series hub retains its own reading sequence.

## Responsive strategy

- **≥1200px:** eligible essays use a sticky 220px contents column beside the
  reading plate; below that, native `<details>` contents sits before the prose.
- **≤980px:** the home hero places its copy and reading actions before the
  portrait; the two-column journey collapses; the sticky left console becomes
  a slim **sticky top strip** (count · place · live coordinates · horizontal
  brass meter) above a horizontal-scroll row of benchmark stop-chips — the route
  identity "rotated 90°". Card triptychs/rows linearise.
- **≤760px:** the colour rooms go **full-bleed edge-to-edge** (more dramatic on
  a phone); header wraps to wordmark + folio with the nav as a second row.
  `html { overflow-x: clip }` makes the 100vw bleed safe without breaking
  sticky positioning.
- **≤520px:** shell tightens, CTA goes full-width, type clamps down so the name
  never overflows.

## Motion

Instrument-grade and cheap: staggered `rise` entrance, route fill off the JS
variable, hover underlines/arrows. Entrances and smooth document scrolling use
`@media (prefers-reduced-motion: no-preference)` with visible static defaults.
Reduced-motion overrides disable transitions/animations and hover movement;
journey stop clicks also respect this preference in `home.js`.
