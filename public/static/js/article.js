/* Current-section enhancement for essay contents.
   Steps: match native contents links to headings, measure the last heading above
   the sticky header, then update aria-current in both layouts on scroll/resize.
   Leave every link unmarked until the first heading reaches that boundary.
   Native anchors and details remain usable when JavaScript is unavailable. */
(() => {
  const links = Array.from(document.querySelectorAll("[data-article-contents] a"));
  if (!links.length) return;

  const ids = new Set(links.map((link) => decodeURIComponent(link.hash.slice(1))));
  const headings = Array.from(document.querySelectorAll(".article-prose h2[id], .article-prose h3[id]"))
    .filter((heading) => ids.has(heading.id));
  if (!headings.length) return;

  const header = document.querySelector(".site-header");
  let currentId = null;
  let pending = false;
  const update = () => {
    pending = false;
    const boundary = (header?.getBoundingClientRect().bottom ?? 0) + 32;
    let active = null;
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > boundary) break;
      active = heading;
    }
    const activeId = active?.id ?? null;
    if (currentId === activeId) return;
    currentId = activeId;
    for (const link of links) {
      if (decodeURIComponent(link.hash.slice(1)) === currentId) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  };
  const schedule = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(update);
  };
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  addEventListener("hashchange", schedule);
  addEventListener("load", schedule, { once: true });
  update();
})();
