import { getCollection, type CollectionEntry } from "astro:content";

export type PostEntry = CollectionEntry<"posts">;

export function postSlug(post: PostEntry): string {
  return post.id.replace(/\.md$/, "");
}

export function formatPostDate(date: Date): string {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    year: "numeric",
  }).format(date);
}

// Normalize for "is this just the title again?" comparison: trim, collapse
// whitespace, lowercase, and drop a leading article so a description that only
// echoes the headline is treated as empty (it adds no information).
function normalizeForCompare(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/^(a|an|the)\s+/, "");
}

export function postExcerpt(post: PostEntry): string {
  const description = (post.data.description || "").trim();
  if (!description) return "";
  if (normalizeForCompare(description) === normalizeForCompare(post.data.title)) {
    return "";
  }
  return description;
}

export async function getPublicPosts(): Promise<PostEntry[]> {
  const posts = await getCollection("posts", ({ data }) => data.draft !== true);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
