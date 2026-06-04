# Styles — the "Meridian" design system

This folder holds the single stylesheet (`site.css`) that drives the entire
public site. There is no CSS framework; everything is hand-authored vanilla CSS
loaded once via `BaseLayout.astro`.

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
journey stop, prose `h2`, and active state. Colour is rationed to one **brass**
accent used like a compass needle, so **structure**, not colour, is the memory.

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

**Type:** `Fraunces` (variable `opsz`/`wght`/`ital`) for the name & all
headlines — an engraved literary serif used only at large optical sizes;
`Source Serif 4` for long-form reading; `IBM Plex Mono` (tracked, uppercase,
`tabular-nums`) as the instrument voice for all coordinates, labels, eyebrows,
meta, and code. (Space Grotesk was removed.)

**Geometry:** sharp `4px` radii, `--grid: 30px` graticule module, content shell
`min(1180px, 100% - 2.5rem)`.

## How the stylesheet is organised

`site.css` is ordered top-to-bottom as: tokens → base + paper graticule →
primitives (eyebrow/benchmark glyph, buttons, links, corner-tick frame) →
header/running-head → hero → **story journey (centrepiece)** → current work →
selected work → writing/article cards → contact → footer → articles index →
article reading view → motion → responsive. A long header comment in the file
restates this algorithm.

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

## Responsive strategy

- **≤980px:** the two-column journey collapses; the sticky left console becomes
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
variable, hover underlines/arrows. All entrance motion is gated behind
`@media (prefers-reduced-motion: no-preference)` with visible static defaults.
