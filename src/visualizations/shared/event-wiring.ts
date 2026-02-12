// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

type EventEntry = [EventTarget, string, EventListener];

export function wireEvents(
  entries: EventEntry[]
): () => void {
  for (const [el, event, handler] of entries) {
    el.addEventListener(event, handler);
  }
  return () => {
    for (const [el, event, handler] of entries) {
      el.removeEventListener(event, handler);
    }
  };
}
