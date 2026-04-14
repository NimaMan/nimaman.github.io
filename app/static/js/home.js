(() => {
  const journey = document.querySelector("[data-story-journey]");
  if (!journey) return;

  const chapters = Array.from(journey.querySelectorAll("[data-story-chapter]"));
  const indicators = Array.from(journey.querySelectorAll("[data-story-indicator]"));
  const progress = journey.querySelector("[data-story-progress]");

  if (!chapters.length || !progress) return;

  const setActive = (index) => {
    const safeIndex = Math.max(0, Math.min(index, chapters.length - 1));
    const ratio = chapters.length === 1 ? 1 : safeIndex / (chapters.length - 1);

    chapters.forEach((chapter, chapterIndex) => {
      chapter.classList.toggle("is-active", chapterIndex === safeIndex);
    });

    indicators.forEach((indicator, indicatorIndex) => {
      indicator.classList.toggle("is-active", indicatorIndex <= safeIndex);
    });

    progress.style.height = `${ratio * 100}%`;
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
