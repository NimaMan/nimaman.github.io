(() => {
  const journey = document.querySelector("[data-story-journey]");
  if (!journey) return;

  const chapters = Array.from(journey.querySelectorAll("[data-story-chapter]"));
  const indicators = Array.from(journey.querySelectorAll("[data-story-indicator]"));
  const progress = journey.querySelector("[data-story-progress]");
  const consoleMeter = journey.querySelector("[data-story-console-meter]");
  const consoleCount = journey.querySelector("[data-story-console-count]");
  const consolePeriod = journey.querySelector("[data-story-console-period]");
  const consolePlace = journey.querySelector("[data-story-console-place]");
  const consoleCountry = journey.querySelector("[data-story-console-country]");
  const consoleContext = journey.querySelector("[data-story-console-context]");
  const consoleBody = journey.querySelector("[data-story-console-body]");

  if (!chapters.length) return;

  const setText = (element, value) => {
    if (element) element.textContent = value || "";
  };

  const setActive = (index) => {
    const safeIndex = Math.max(0, Math.min(index, chapters.length - 1));
    const ratio = chapters.length === 1 ? 1 : safeIndex / (chapters.length - 1);
    const activeChapter = chapters[safeIndex];

    chapters.forEach((chapter, chapterIndex) => {
      chapter.classList.toggle("is-active", chapterIndex === safeIndex);
    });

    indicators.forEach((indicator, indicatorIndex) => {
      const isActive = indicatorIndex === safeIndex;
      indicator.classList.toggle("is-active", isActive);
      indicator.classList.toggle("is-passed", indicatorIndex < safeIndex);
      if (isActive) {
        indicator.setAttribute("aria-current", "step");
      } else {
        indicator.removeAttribute("aria-current");
      }
    });

    const percent = `${ratio * 100}%`;
    journey.style.setProperty("--story-route-progress", percent);
    journey.dataset.activeScene = activeChapter.dataset.storyScene || "";

    if (progress) progress.style.height = percent;
    if (consoleMeter) consoleMeter.style.width = percent;

    setText(consoleCount, activeChapter.dataset.storyCount);
    setText(consolePeriod, activeChapter.dataset.storyPeriod);
    setText(consolePlace, activeChapter.dataset.storyPlace);
    setText(consoleCountry, activeChapter.dataset.storyCountry);
    setText(consoleContext, activeChapter.dataset.storyContext);
    setText(consoleBody, activeChapter.dataset.storyBody);
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);

      if (!visible.length) return;

      const index = chapters.indexOf(visible[0].target);
      if (index >= 0) setActive(index);
    },
    {
      rootMargin: "-35% 0px -35% 0px",
      threshold: [0.2, 0.4, 0.6],
    }
  );

  chapters.forEach((chapter) => observer.observe(chapter));

  indicators.forEach((indicator) => {
    indicator.addEventListener("click", (event) => {
      event.preventDefault();
      const targetIndex = Number(indicator.dataset.storyTarget || "0");
      const chapter = chapters[targetIndex];
      if (!chapter) return;
      chapter.scrollIntoView({ behavior: "smooth", block: "start" });
      setActive(targetIndex);
    });
  });

  setActive(0);
})();
