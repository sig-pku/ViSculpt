export function initializeUiPreferences() {
  const root = document.documentElement;
  let storedTheme: string | null = null;
  let storedSize: string | null = null;

  try {
    storedTheme = window.localStorage.getItem("geometry-agent.theme");
    storedSize = window.localStorage.getItem("geometry-agent.size");
  } catch {
    // Fall back to the system theme when browser storage is unavailable.
  }

  const themeMode =
    storedTheme === "light" ||
    storedTheme === "dark" ||
    storedTheme === "system"
      ? storedTheme
      : "system";
  const systemTheme =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  const size =
    storedSize === "small" ||
    storedSize === "medium" ||
    storedSize === "large"
      ? storedSize
      : "medium";

  root.dataset.theme = themeMode === "system" ? systemTheme : themeMode;
  root.dataset.themeMode = themeMode;
  root.dataset.uiSize = size;
  root.dataset.keyboardModifier = /Macintosh|Mac OS X|iPhone|iPad|iPod/i.test(
    window.navigator.userAgent,
  )
    ? "command"
    : "control";
}

export const UI_PREFERENCE_BOOTSTRAP_SCRIPT =
  `(${initializeUiPreferences.toString()})();`;
