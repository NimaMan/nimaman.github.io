import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://nimamanaf.com",
  output: "static",
  markdown: {
    // Pin the Shiki theme so code-token colours are stable; site.css then
    // lifts the low-contrast comment token (#6A737D) for AA on the dark block.
    shikiConfig: { theme: "github-dark" },
  },
});
