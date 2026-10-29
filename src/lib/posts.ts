// Public post queries and taxonomy helpers.
// Steps: filter drafts, order posts, normalize category paths, then derive shared
// category anchors/counts and excerpts so every route presents the same archive.
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

// A post's `category` frontmatter is a "/"-separated path of display names,
// root first (e.g. "Value Chains/Money"). Returns the trimmed segments, or []
// for an uncategorized post. The /posts archive filters on segments[0] and
// displays the full path; the value-chains hub groups on segments[1].
export function categorySegments(post: PostEntry): string[] {
  return (post.data.category || "")
    .split("/")
    .map((segment) => segment.trim())
    .filter(Boolean);
}

// Display-name anchors are shared by the home strip, index, hub, and trails.
export function categoryAnchor(name: string): string {
  return name
    .normalize("NFKD")
    .replace(/\p{Mark}/gu, "")
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-|-$/g, "");
}

export function postCategories(posts: PostEntry[]) {
  const order = [
    "Value Chains",
    "Optimization & Learning",
    "Mathematical Curiosities",
    "Research & Teaching",
  ];
  const counts = new Map<string, number>();
  for (const post of posts) {
    const name = categorySegments(post)[0] ?? "Uncategorized";
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  const names = [
    ...order.filter((name) => counts.has(name)),
    ...[...counts.keys()]
      .filter((name) => !order.includes(name) && name !== "Uncategorized")
      .sort((a, b) => a.localeCompare(b)),
    ...(counts.has("Uncategorized") ? ["Uncategorized"] : []),
  ];
  return names.map((name) => ({ name, id: categoryAnchor(name), count: counts.get(name)! }));
}

// Posts belonging to a named series (e.g. "value_chains"), ordered by their
// explicit series_order so the collection reads as a sequence rather than by date.
export async function getSeriesPosts(series: string): Promise<PostEntry[]> {
  const posts = await getCollection(
    "posts",
    ({ data }) => data.draft !== true && data.series === series,
  );
  return posts.sort(
    (a, b) => (a.data.series_order ?? 0) - (b.data.series_order ?? 0),
  );
}
