// Static-site Markdown configuration.
// Steps: collect Astro's heading IDs before local permalink decoration, pin code
// colours, then emit static HTML so contents and section links also work without JS.
import { defineConfig } from "astro/config";
import { rehypeHeadingIds } from "@astrojs/markdown-remark";
import rehypeHeadingAnchors from "./src/lib/rehype-heading-anchors.mjs";

export default defineConfig({
  site: "https://nimamanaf.com",
  output: "static",
  markdown: {
    rehypePlugins: [rehypeHeadingIds, rehypeHeadingAnchors],
    // Pin the Shiki theme so code-token colours are stable; site.css then
    // lifts the low-contrast comment token (#6A737D) for AA on the dark block.
    shikiConfig: { theme: "github-dark" },
  },
});
