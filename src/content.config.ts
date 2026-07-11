import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const posts = defineCollection({
  // [^_]* excludes underscore-prefixed files (e.g. _POSTS.md, the folder's
  // documentation) from the collection, mirroring the src/pages convention.
  loader: glob({ pattern: "**/[^_]*.md", base: "./src/content/posts" }),
  schema: z.object({
    author: z.string().optional(),
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    math: z.boolean().optional(),
    draft: z.boolean().optional(),
    // Tree position of the post in the site's category taxonomy: a "/"-separated
    // path of display names, root first (e.g. "Value Chains/Money"). Depth is
    // unbounded so the tree can grow; segment text is used verbatim as headings.
    category: z.string().optional(),
    series: z.string().optional(),
    series_order: z.coerce.number().optional(),
    series_label: z.string().optional(),
  }),
});

export const collections = { posts };
