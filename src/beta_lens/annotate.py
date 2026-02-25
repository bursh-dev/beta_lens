"""
Hold annotation tool for bouldering routes.

Opens an anchor image and lets you click to mark holds.
Saves hold positions to a route JSON file.

Controls:
  Left click     — place a new hold at cursor
  Right click    — delete nearest hold (within 50px)
  Scroll up/down — increase/decrease radius of nearest hold
  U              — undo last action
  S              — save route JSON
  Q / ESC        — quit
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Defaults
DEFAULT_RADIUS = 40
MIN_RADIUS = 10
MAX_RADIUS = 200
RADIUS_STEP = 5
DELETE_THRESHOLD = 50  # pixels


class AnnotationState:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            print(f"Error: cannot load image {image_path}")
            sys.exit(1)
        self.holds: list[dict] = []
        self.history: list[list[dict]] = []  # for undo
        self.dirty = False

    def _snapshot(self):
        """Save current state for undo."""
        self.history.append([h.copy() for h in self.holds])

    def add_hold(self, x: int, y: int):
        self._snapshot()
        hold_id = f"h{len(self.holds) + 1}"
        self.holds.append({"id": hold_id, "x": x, "y": y, "r": DEFAULT_RADIUS, "kind": "hand"})
        self.dirty = True

    def delete_nearest(self, x: int, y: int):
        if not self.holds:
            return
        dists = [((h["x"] - x) ** 2 + (h["y"] - y) ** 2) ** 0.5 for h in self.holds]
        idx = int(np.argmin(dists))
        if dists[idx] < DELETE_THRESHOLD:
            self._snapshot()
            self.holds.pop(idx)
            self._renumber()
            self.dirty = True

    def resize_nearest(self, x: int, y: int, delta: int):
        if not self.holds:
            return
        dists = [((h["x"] - x) ** 2 + (h["y"] - y) ** 2) ** 0.5 for h in self.holds]
        idx = int(np.argmin(dists))
        self._snapshot()
        new_r = self.holds[idx]["r"] + delta * RADIUS_STEP
        self.holds[idx]["r"] = max(MIN_RADIUS, min(MAX_RADIUS, new_r))
        self.dirty = True

    def undo(self):
        if self.history:
            self.holds = self.history.pop()
            self.dirty = True

    def _renumber(self):
        for i, h in enumerate(self.holds):
            h["id"] = f"h{i + 1}"

    def render(self) -> np.ndarray:
        frame = self.original.copy()
        for h in self.holds:
            center = (h["x"], h["y"])
            r = h["r"]
            # Semi-transparent circle
            overlay = frame.copy()
            cv2.circle(overlay, center, r, (0, 0, 255), 2)
            cv2.circle(overlay, center, r, (0, 0, 255, 80), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            # Outline and label
            cv2.circle(frame, center, r, (0, 0, 255), 2)
            cv2.circle(frame, center, 4, (0, 255, 255), -1)
            cv2.putText(frame, h["id"], (h["x"] - 10, h["y"] - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Status bar
        status = f"Holds: {len(self.holds)} | LClick=add  RClick=del  Scroll=resize  U=undo  S=save  Q=quit"
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def save(self, output_path: Path):
        h, w = self.original.shape[:2]
        route = {
            "routeId": f"r_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "name": "",
            "anchor": {
                "imageFile": self.image_path.name,
                "widthPx": w,
                "heightPx": h,
            },
            "holds": self.holds,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }
        output_path.write_text(json.dumps(route, indent=2))
        self.dirty = False
        print(f"Saved {len(self.holds)} holds to {output_path}")


# Mouse position tracking for scroll events
_mouse_x, _mouse_y = 0, 0


def _mouse_callback(event, x, y, flags, state: AnnotationState):
    global _mouse_x, _mouse_y
    _mouse_x, _mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        state.add_hold(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.delete_nearest(x, y)
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = 1 if flags > 0 else -1
        state.resize_nearest(x, y, delta)


def main():
    parser = argparse.ArgumentParser(description="Annotate bouldering holds on a wall photo")
    parser.add_argument("image", nargs="?", default="simulation_data/for_annotation_20260222_204347.jpg",
                        help="Path to anchor image")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: simulation_data/route.json)")
    parser.add_argument("-l", "--load", default=None,
                        help="Load existing route JSON to edit")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    output_path = Path(args.output) if args.output else image_path.parent / "route.json"

    state = AnnotationState(image_path)

    # Load existing annotations if provided
    if args.load:
        load_path = Path(args.load)
        if load_path.exists():
            data = json.loads(load_path.read_text())
            state.holds = data.get("holds", [])
            print(f"Loaded {len(state.holds)} holds from {load_path}")

    window = "BetaLens Annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 960)
    cv2.setMouseCallback(window, _mouse_callback, state)

    while True:
        frame = state.render()
        cv2.imshow(window, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            if state.dirty:
                print("Unsaved changes! Press Q again to quit or S to save.")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord("s"):
                    state.save(output_path)
                elif key2 != ord("q") and key2 != 27:
                    continue
            break
        elif key == ord("s"):
            state.save(output_path)
        elif key == ord("u"):
            state.undo()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
