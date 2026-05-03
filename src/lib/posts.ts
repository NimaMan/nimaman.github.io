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

export function postExcerpt(post: PostEntry): string {
  return post.data.description || "";
}

export async function getPublicPosts(): Promise<PostEntry[]> {
  const posts = await getCollection("posts", ({ data }) => data.draft !== true);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
