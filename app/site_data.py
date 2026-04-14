"""
Content helpers for the public nimamanafcom website.

This module keeps the public site self-contained:
- structured homepage and CV content lives here
- markdown posts are loaded from app/site_content/posts
- Hugo-specific asset paths are rewritten to the FastAPI static mount
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import markdown
import yaml


APP_DIR = Path(__file__).parent
POSTS_DIR = APP_DIR / "site_content" / "posts"

FRONT_MATTER_PATTERN = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


SOCIAL_LINKS = [
    {
        "label": "Email",
        "symbol": "@",
        "href": "mailto:nima.manaf8@gmail.com",
        "note": "nima.manaf8@gmail.com",
        "external": False,
    },
    {
        "label": "LinkedIn",
        "symbol": "in",
        "href": "https://www.linkedin.com/in/nima-manaf-phd-02174b74/",
        "note": "Professional profile",
        "external": True,
    },
    {
        "label": "GitHub",
        "symbol": "</>",
        "href": "https://github.com/nimaman",
        "note": "Code and experiments",
        "external": True,
    },
    {
        "label": "Scholar",
        "symbol": "✦",
        "href": "https://scholar.google.com/citations?user=oE_9bDkAAAAJ&hl=en",
        "note": "Publications and citations",
        "external": True,
    },
]


HOME_PAGE = {
    "name": "Nima Manafzadeh Dizbin",
    "short_name": "Nima Manaf",
    "role": "Data Analytics Lead",
    "tagline": "Applied machine learning, optimization, and analytics.",
    "description": (
        "Applied machine learning, optimization, and analytics across finance, operations research, and decision systems."
    ),
    "current_location": "Utrecht, Netherlands",
    "current_role": "Data Analytics Lead in Value Chain Finance Portfolio",
    "current_org": "Rabobank",
    "snapshot_summary": (
        "Working across credit risk, portfolio analytics, machine learning, and decision systems, "
        "after a route through operations research, industrial AI, and academia."
    ),
    "hero_image": "/static/images/rabo-photo.jpg",
    "journey_heading": "A route across cities, degrees, labs, and industries.",
    "journey_intro": (
        "Scroll through the route. The stage follows the active stop as the journey moves."
    ),
    "current_heading": "Current Work",
    "current_summary": (
        "The current chapter sits at the intersection of financial risk, optimization, "
        "machine learning, and decision systems used in practice."
    ),
    "writing_heading": "Selected Writing",
    "writing_intro": (
        "Research notes, visual experiments, and technical essays collected across different phases of the journey."
    ),
    "contact_heading": "If the problem is real and difficult, I am interested.",
    "contact_summary": (
        "I am especially drawn to work that combines data, uncertainty, and operational constraints."
    ),
}


LIFE_TIMELINE = [
    {
        "period": "Beginning",
        "scene": "qaradag",
        "place_short": "Qaradag",
        "place": "East Azerbaijan, Iran",
        "context": "Origin",
        "summary": "Born in Qaradag, in northwestern Iran.",
        "body": "The starting point: Qaradag, in the Azerbaijan region of Iran.",
    },
    {
        "period": "2010 - 2014",
        "scene": "tehran",
        "place_short": "Tehran",
        "place": "Iran",
        "context": "Sharif University of Technology",
        "summary": "Four undergraduate years in the capital.",
        "body": "Tehran meant four formative years at Sharif University of Technology.",
    },
    {
        "period": "2014 - 2019",
        "scene": "istanbul",
        "place_short": "Istanbul",
        "place": "Turkey",
        "context": "Koc University, MSc to PhD",
        "summary": "Master's study turned into a doctoral chapter.",
        "body": "Istanbul became the long academic chapter: first the MSc, then the PhD at Koc University.",
    },
    {
        "period": "2019 - 2020",
        "scene": "stuttgart",
        "place_short": "Stuttgart",
        "place": "Germany",
        "context": "Bosch Center for Artificial Intelligence",
        "summary": "One year inside industrial AI.",
        "body": "A year at Bosch AI working on scheduling, reinforcement learning, search, and industrial data.",
    },
    {
        "period": "2020",
        "scene": "istanbul-return",
        "place_short": "Istanbul",
        "place": "Turkey",
        "context": "Final PhD stretch",
        "summary": "Back to close the doctorate.",
        "body": "I returned to Istanbul to write, defend, and close the PhD.",
    },
    {
        "period": "2020 - 2022",
        "scene": "eindhoven",
        "place_short": "Eindhoven",
        "place": "Netherlands",
        "context": "Postdoc at TU/e",
        "summary": "Postdoc years, supervision, and the first master's student graduating.",
        "body": "Eindhoven brought the postdoc years, and by the second year the first master's student I supervised had graduated.",
    },
    {
        "period": "2022 - Present",
        "scene": "utrecht",
        "place_short": "Utrecht",
        "place": "Netherlands",
        "context": "Rabobank",
        "summary": "Research moved into finance, risk, and portfolio analytics.",
        "body": "Utrecht is the current chapter: Rabobank, value chain finance, and research habits translated into production decisions.",
    },
]


EXPERTISE = [
    {
        "title": "Financial Risk and Portfolio Analytics",
        "summary": (
            "Portfolio monitoring, challenger modeling, regulation-facing analysis, and decision support "
            "for value chain finance and credit risk workflows."
        ),
        "tags": ["Credit risk", "Value chain finance", "Monitoring", "Model governance"],
    },
    {
        "title": "Learning and Optimization",
        "summary": (
            "Reinforcement learning, evolution strategies, graph-based modeling, and optimization methods "
            "for environments with delayed feedback and hard constraints."
        ),
        "tags": ["Reinforcement learning", "Evolution strategies", "GNNs", "Stochastic optimization"],
    },
    {
        "title": "Operational Systems",
        "summary": (
            "Production planning, inventory control, scheduling, and data-driven performance analysis for "
            "industrial systems that have to work outside the lab."
        ),
        "tags": ["Manufacturing", "Scheduling", "Inventory", "Operations research"],
    },
]


SELECTED_WORK = [
    {
        "title": "Value Chain Finance Analytics",
        "summary": (
            "Leading analytics initiatives at Rabobank with a focus on portfolio performance, risk, "
            "and decision quality."
        ),
        "meta": "Current work",
        "href": "/cv",
    },
    {
        "title": "Evolution Strategies for Inventory Control",
        "summary": (
            "Using gradient-free optimization and compact neural policies to solve lost-sales inventory problems."
        ),
        "meta": "Research thread",
        "href": "/posts/lost_sales_rl",
    },
    {
        "title": "Production Systems from Event Data",
        "summary": (
            "Doctoral work on modeling production systems from observed inter-event times and using those models "
            "for better control."
        ),
        "meta": "PhD work",
        "href": "/posts/phd_thesis",
    },
]


CV_PROFILE = {
    "name": "Nima Manafzadeh Dizbin, PhD",
    "title": "Data Analytics Lead in Value Chain Finance Portfolio",
    "location": "Rabobank, Utrecht, Netherlands",
    "summary": (
        "Applied researcher and analytics leader working across finance, machine learning, and optimization. "
        "Experienced in translating technical models into decisions used by teams with operational and regulatory constraints."
    ),
}


CV_EXPERIENCE = [
    {
        "title": "Data Analytics Lead in Value Chain Finance Portfolio",
        "organization": "Rabobank",
        "location": "Utrecht, Netherlands",
        "date_range": "August 2024 - Present",
        "points": [
            "Leading data analytics initiatives for the Value Chain Finance portfolio.",
            "Developing advanced analytics solutions to improve portfolio performance and decision quality.",
            "Applying machine learning models to risk assessment and operational prioritization.",
            "Working across business and technical teams to move analytics into day-to-day use.",
        ],
    },
    {
        "title": "Data Scientist",
        "organization": "Rabobank",
        "location": "Utrecht, Netherlands",
        "date_range": "May 2022 - August 2024",
        "points": [
            "Worked on EU AI Act interpretation and model governance questions.",
            "Evaluated machine learning models for ECB-facing reporting processes.",
            "Addressed methodological questions in the rural portfolio using advanced statistical techniques.",
            "Adapted EBA guideline logic in Python to streamline regulatory work.",
            "Built monitoring frameworks for risk-adjusted performance measurement.",
            "Designed challenger models in support of IFRS9 processes.",
        ],
    },
    {
        "title": "Postdoctoral Research Associate",
        "organization": "Eindhoven University of Technology",
        "location": "Eindhoven, Netherlands",
        "date_range": "September 2020 - April 2022",
        "points": [
            "Researched evolutionary training of deep neural networks.",
            "Applied reinforcement learning and evolution strategies to inventory management problems.",
            "Supervised Bachelor's and Master's theses.",
            "Published research in optimization and machine learning for decision systems.",
        ],
    },
    {
        "title": "PhD Industrial Sabbatical",
        "organization": "Bosch Center for Artificial Intelligence",
        "location": "Stuttgart, Germany",
        "date_range": "February 2019 - January 2020",
        "points": [
            "Studied Q-learning and Monte Carlo Tree Search for scheduling problems.",
            "Developed the reinforced genetic algorithm for optimization problems.",
            "Analyzed semiconductor manufacturing network data at scale.",
            "Co-invented the Learn Offline, Search Online methodology.",
        ],
    },
]


CV_EDUCATION = [
    {
        "degree": "PhD in Business Administration",
        "institution": "Koc University",
        "location": "Istanbul, Turkey",
        "date_range": "September 2016 - September 2020",
        "details": [
            "Thesis: Modelling and Control of Production Systems Based on Observed Inter-event Times.",
            "PhD Academic Excellence Award recipient.",
            "Supervisor: Prof. Dr. Baris Tan.",
        ],
    },
    {
        "degree": "MSc in Operations and Information Systems",
        "institution": "Koc University",
        "location": "Istanbul, Turkey",
        "date_range": "September 2014 - September 2016",
        "details": [
            "Full scholarship recipient.",
            "Focused on operations research and data analytics.",
        ],
    },
    {
        "degree": "BSc in Industrial Engineering",
        "institution": "Sharif University of Technology",
        "location": "Tehran, Iran",
        "date_range": "September 2010 - September 2014",
        "details": [
            "Ranked among the top 0.1% of students in the national university entrance exam.",
        ],
    },
]


CV_SKILLS = [
    {
        "title": "Machine Learning and AI",
        "items": [
            "Reinforcement learning",
            "Deep learning and neural networks",
            "Graph neural networks",
            "Supervised and unsupervised learning",
            "Statistical learning",
            "Evolution strategies",
        ],
    },
    {
        "title": "Optimization Methods",
        "items": [
            "Combinatorial and stochastic optimization",
            "Genetic algorithms",
            "Monte Carlo Tree Search",
            "Markov decision processes",
            "Dynamic programming",
            "Mathematical programming",
        ],
    },
    {
        "title": "Programming and Tools",
        "items": [
            "Python, C++, SQL",
            "PyTorch, scikit-learn, XGBoost, LightGBM",
            "Pandas, NumPy, SciPy, cuDF",
            "Matplotlib, Seaborn, Plotly",
            "Gurobi, CPLEX, OR-Tools",
            "Ray, MPI, multiprocessing, Git",
        ],
    },
    {
        "title": "Application Domains",
        "items": [
            "Supply chain management",
            "Inventory control",
            "Production planning and scheduling",
            "Healthcare operations",
            "Semiconductor manufacturing",
            "Financial risk analytics",
        ],
    },
]


CV_HIGHLIGHTS = [
    {
        "title": "Awards and Recognition",
        "items": [
            "PhD Academic Excellence Award, Koc University (2020)",
            "Exceptional Talent Student distinction in the national exam (2010)",
            "Top 1% in the National Chemistry Olympiad, Iran (2009)",
        ],
    },
    {
        "title": "Patents and Grants",
        "items": [
            "Learn Offline, Search Online hybrid scheduling methodology.",
            "NWO 1,000,000 hours computing time grant.",
            "NWO 500,000 hours computing time grant.",
            "Productive 4.0 project scholarship (2017).",
        ],
    },
    {
        "title": "Languages and Interests",
        "items": [
            "English, Turkish, Persian, and basic Dutch.",
            "Optimization algorithms, financial analytics, teaching, and mentoring.",
        ],
    },
]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_date(value: Any, fallback_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value.strip())
    raise ValueError(f"Post {fallback_name} is missing a valid date")


def _split_front_matter(raw_text: str, filename: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_PATTERN.match(raw_text.lstrip("\ufeff"))
    if not match:
        raise ValueError(f"{filename} does not start with YAML front matter")
    metadata = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    return metadata, body


def _rewrite_asset_urls(markdown_text: str) -> str:
    replacements = {
        "](/gif/": "](/static/gif/",
        "](/images/": "](/static/images/",
        'src="/gif/': 'src="/static/gif/',
        'src="/images/': 'src="/static/images/',
        'href="/gif/': 'href="/static/gif/',
        'href="/images/': 'href="/static/images/',
    }
    updated = markdown_text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def _render_markdown(markdown_text: str) -> str:
    renderer = markdown.Markdown(
        extensions=["fenced_code", "tables", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    return renderer.convert(markdown_text)


def _make_excerpt(html_text: str, fallback: str = "", max_length: int = 220) -> str:
    stripped = re.sub(r"<[^>]+>", " ", html_text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    source = fallback.strip() or stripped
    if len(source) <= max_length:
        return source
    return source[: max_length - 1].rsplit(" ", 1)[0] + "..."


def _load_post(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw_text, path.name)
    body = _rewrite_asset_urls(body)
    html = _render_markdown(body)
    post_date = _coerce_date(metadata.get("date"), path.name)
    description = (metadata.get("description") or "").strip()
    return {
        "slug": path.stem,
        "title": metadata.get("title") or path.stem.replace("_", " ").title(),
        "author": metadata.get("author", HOME_PAGE["name"]),
        "date": post_date,
        "date_display": post_date.strftime("%b %d, %Y"),
        "description": description,
        "excerpt": _make_excerpt(html, fallback=description),
        "html": html,
        "math": _coerce_bool(metadata.get("math", False)),
        "draft": _coerce_bool(metadata.get("draft", False)),
    }


@lru_cache(maxsize=1)
def get_posts() -> list[dict[str, Any]]:
    posts = []
    for path in POSTS_DIR.glob("*.md"):
        post = _load_post(path)
        if not post["draft"]:
            posts.append(post)
    posts.sort(key=lambda item: item["date"], reverse=True)
    return posts


def get_post_by_slug(slug: str) -> dict[str, Any] | None:
    for post in get_posts():
        if post["slug"] == slug:
            return post
    return None
