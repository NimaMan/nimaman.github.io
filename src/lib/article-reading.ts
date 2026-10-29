// Build long-essay reading aids without changing the published Markdown.
// Steps: remove non-reading markup before estimating minutes, then group Astro's
// h3 headings beneath their preceding h2; show contents only for substantial essays.

interface Heading {
  depth: number;
  slug: string;
  text: string;
}

export interface ContentsSection {
  slug: string;
  text: string;
  children: { slug: string; text: string }[];
}

export function readingMinutes(markdown: string): number {
  const text = markdown
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, " ")
    .replace(/^\s*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\s*\1\s*$/gm, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/^\s*\[[^\]]+\]:\s+.*$/gm, " ")
    .replace(/!\[[^\]]*\]\([^\n]*?\)/g, " ")
    .replace(/\[([^\]]+)\]\((?:[^()\n]|\([^()\n]*\))*\)/g, "$1")
    .replace(/\[([^\]]+)\]\[[^\]]*\]/g, "$1")
    .replace(/https?:\/\/[^\s<>]+/g, " ")
    .replace(/&(?:#\d+|#x[\da-f]+|[a-z]+);/gi, " ")
    .replace(/\\[a-z]+/gi, " ")
    .replace(/[*_~`#>|{}[\]$\\]/g, " ");
  const words = text.match(/[\p{Letter}\p{Number}]+(?:['’–-][\p{Letter}\p{Number}]+)*/gu) ?? [];
  return Math.max(1, Math.ceil(words.length / 230));
}

export function articleContents(headings: Heading[]): ContentsSection[] {
  const sections: ContentsSection[] = [];
  for (const heading of headings) {
    if (heading.depth === 2) {
      sections.push({ slug: heading.slug, text: heading.text, children: [] });
    } else if (heading.depth === 3 && sections.length > 0) {
      sections[sections.length - 1].children.push({ slug: heading.slug, text: heading.text });
    }
  }
  return sections.length >= 3 ? sections : [];
}
