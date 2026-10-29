/* Section permalinks for rendered essays, using Astro's existing heading IDs.
   Steps: walk the HAST, find h2/h3 elements with IDs, then append an empty,
   labelled link. CSS supplies the visible glyph so contents labels stay clean. */
export default function rehypeHeadingAnchors() {
  return (tree) => {
    const textOf = (node) => node.type === "text"
      ? node.value
      : (node.children ?? []).map(textOf).join("");

    const visit = (node) => {
      if (node.type === "element" && /^h[23]$/.test(node.tagName) && node.properties?.id) {
        if (node.children.some((child) => child.properties?.className?.includes("heading-anchor"))) return;
        node.children.push({
          type: "element",
          tagName: "a",
          properties: {
            className: ["heading-anchor"],
            href: `#${node.properties.id}`,
            ariaLabel: `Link to section: ${textOf(node)}`,
          },
          children: [],
        });
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}
