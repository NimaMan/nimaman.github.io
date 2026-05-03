import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    author: z.string().optional(),
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    math: z.boolean().optional(),
    draft: z.boolean().optional(),
  }),
});

export const collections = { posts };
