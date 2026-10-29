/* Category filtering for the newest-first articles archive.
   Steps: read the shared category ID from the hash, hide unmatched posts and
   empty years, then mark the current link and announce the visible count.
   Native hash links preserve history and incoming category URLs; without this
   enhancement every public post remains readable in chronological order. */
(() => {
  const listing = document.querySelector("[data-post-listing]");
  if (!listing) return;

  const filters = Array.from(listing.querySelectorAll("[data-post-filter]"));
  const years = Array.from(listing.querySelectorAll("[data-post-year]"))
    .map((section) => ({
      section,
      posts: Array.from(section.querySelectorAll("[data-post-category]")),
    }));
  const count = listing.querySelector("[data-post-count]");

  const applyFilter = () => {
    let category;
    try {
      category = decodeURIComponent(location.hash.slice(1));
    } catch {
      category = "";
    }
    const active = filters.find((link) => link.dataset.postFilter === category) ?? filters[0];
    const selected = active.dataset.postFilter;
    let visible = 0;

    for (const { section, posts } of years) {
      let yearCount = 0;
      for (const post of posts) {
        post.hidden = Boolean(selected) && post.dataset.postCategory !== selected;
        if (!post.hidden) yearCount += 1;
      }
      section.hidden = yearCount === 0;
      visible += yearCount;
    }

    for (const link of filters) {
      if (link === active) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }
    const label = selected ? ` in ${active.dataset.categoryName}` : "";
    count.textContent = `${visible} ${visible === 1 ? "article" : "articles"}${label} · Newest first`;
  };

  addEventListener("hashchange", applyFilter);
  applyFilter();
})();
