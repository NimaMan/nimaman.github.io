<!-- Public content helper guide. Steps: describe query ordering and taxonomy,
then reading utilities so routes share stable links and use the intended date
or series sequence without duplicating helper logic. -->
# Public content helpers

`posts.ts` filters drafts, sorts public posts by date (or by `series_order` for a
series), formats dates, suppresses redundant excerpts, and splits category paths.
`categoryAnchor()` slugifies display names for links shared by the home page,
archive, article trails, and series hub. `postCategories()` derives counts and
editorial ordering, retaining unknown categories and a trailing Uncategorized bucket.
The `/posts` archive preserves the public query's date order under year headings,
uses the full category path as a label and its first segment as a filter key.
The Value Chains hub continues to use the explicit series reading sequence.

`article-reading.ts` estimates reading time at 230 words/minute, rounded up with a
one-minute minimum. It counts Unicode words after removing HTML/comments, scripts,
styles, fenced code, Markdown link destinations and formatting. It also groups
Astro's rendered h3 headings under their preceding h2, returning contents only
when there are at least three h2 sections. Initial orphan h3 headings are omitted.

`rehype-heading-anchors.mjs` is a dependency-free HAST visitor that appends labelled
permalinks to h2/h3 elements. `astro.config.mjs` first runs Astro's own exported
`rehypeHeadingIds` from its already-installed Markdown renderer so IDs match
`render(post).headings`, including duplicate titles. Links have no text children;
CSS supplies `#`, keeping heading text and contents labels clean.
