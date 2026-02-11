// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// --- Playback controller ---

interface PlaybackCallbacks {
  getSnapshotCount: () => number;
  getCurrentIndex: () => number;
  setCurrentIndex: (i: number) => void;
  onUpdate: () => void;
}

const TOTAL_PLAYBACK_DURATION_MS = 3000;

export interface PlaybackController {
  isPlaying(): boolean;
  start(): void;
  stop(): void;
  toggle(): void;
  destroy(): void;
}

export function createPlaybackController(
  cb: PlaybackCallbacks
): PlaybackController {
  let playing = false;
  let timeoutId: number | null = null;

  function stepDelayMs(): number {
    return Math.max(
      16,
      TOTAL_PLAYBACK_DURATION_MS
        / Math.max(1, cb.getSnapshotCount() - 1)
    );
  }

  function stop(): void {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    playing = false;
  }

  function scheduleNext(): void {
    if (!playing) {
      return;
    }
    if (cb.getCurrentIndex() >= cb.getSnapshotCount() - 1) {
      stop();
      cb.onUpdate();
      return;
    }

    timeoutId = window.setTimeout(() => {
      if (!playing) {
        return;
      }
      if (cb.getCurrentIndex() < cb.getSnapshotCount() - 1) {
        cb.setCurrentIndex(cb.getCurrentIndex() + 1);
        cb.onUpdate();
        scheduleNext();
      } else {
        stop();
        cb.onUpdate();
      }
    }, stepDelayMs());
  }

  function start(): void {
    if (playing) {
      return;
    }
    if (cb.getCurrentIndex() >= cb.getSnapshotCount() - 1) {
      cb.setCurrentIndex(0);
      cb.onUpdate();
    }
    playing = true;
    cb.onUpdate();
    scheduleNext();
  }

  function toggle(): void {
    if (playing) {
      stop();
      cb.onUpdate();
    } else {
      start();
    }
  }

  return {
    isPlaying: () => playing,
    start,
    stop,
    toggle,
    destroy: stop
  };
}

// --- Hold-to-repeat button ---

const REPEAT_INITIAL_DELAY_MS = 300;
const REPEAT_INTERVAL_MS = 90;

export interface RepeatButtonHandlers {
  handlePointerDown: (e: PointerEvent) => void;
  handlePointerUp: (e: PointerEvent) => void;
  handleClick: () => void;
  destroy: () => void;
}

export function createRepeatButton(
  button: HTMLButtonElement,
  stepFn: () => void
): RepeatButtonHandlers {
  let delayTimerId: number | null = null;
  let repeatTimerId: number | null = null;
  let didRepeat = false;

  function stopRepeat(pointerId: number): void {
    if (delayTimerId !== null) {
      window.clearTimeout(delayTimerId);
      delayTimerId = null;
    }
    if (repeatTimerId !== null) {
      window.clearInterval(repeatTimerId);
      repeatTimerId = null;
    }
    if (button.hasPointerCapture(pointerId)) {
      button.releasePointerCapture(pointerId);
    }
  }

  function handlePointerDown(e: PointerEvent): void {
    if (e.button !== 0) {
      return;
    }
    e.preventDefault();
    didRepeat = false;
    button.setPointerCapture(e.pointerId);
    delayTimerId = window.setTimeout(() => {
      stepFn();
      didRepeat = true;
      repeatTimerId = window.setInterval(() => {
        stepFn();
      }, REPEAT_INTERVAL_MS);
    }, REPEAT_INITIAL_DELAY_MS);
  }

  function handlePointerUp(e: PointerEvent): void {
    stopRepeat(e.pointerId);
  }

  function handleClick(): void {
    if (didRepeat) {
      didRepeat = false;
      return;
    }
    stepFn();
  }

  function destroy(): void {
    if (delayTimerId !== null) {
      window.clearTimeout(delayTimerId);
    }
    if (repeatTimerId !== null) {
      window.clearInterval(repeatTimerId);
    }
  }

  return {
    handlePointerDown,
    handlePointerUp,
    handleClick,
    destroy
  };
}
