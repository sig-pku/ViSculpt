export function readBrowserSetting(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

export function writeBrowserSetting(key: string, value: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage?.setItem(key, value);
  } catch {
    // Restricted WebViews keep preferences only for the current React session.
  }
}

export function removeBrowserSetting(key: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage?.removeItem(key);
  } catch {
    // Restricted WebViews require no additional cleanup.
  }
}
