"""
Tracking pipeline for bouldering route overlay.

Loads an anchor image + route JSON, processes video files,
and outputs videos with hold circles overlaid using homography tracking.

Usage:
  track <video>              — process one video
  track --all <directory>    — process all .mp4 files in directory
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class AnchorTracker:
    """Feature-based anchor tracking using ORB + homography."""

    def __init__(self, anchor_image: np.ndarray, min_matches: int = 15):
        self.anchor = anchor_image
        self.anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
        self.min_matches = min_matches

        # ORB detector — increase features for better matching
        self.detector = cv2.ORB_create(nfeatures=2000)

        # Compute anchor features once
        self.anchor_kp, self.anchor_desc = self.detector.detectAndCompute(self.anchor_gray, None)
        if self.anchor_desc is None:
            print("Error: no features found in anchor image")
            sys.exit(1)
        print(f"Anchor: {len(self.anchor_kp)} features detected")

        # Brute-force matcher with Hamming distance (for ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_homography(self, frame: np.ndarray) -> tuple[np.ndarray | None, int]:
        """
        Find homography from anchor image to frame.
        Returns (homography_matrix, match_count) or (None, match_count) if tracking lost.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = self.detector.detectAndCompute(frame_gray, None)

        if frame_desc is None or len(frame_kp) < self.min_matches:
            return None, 0

        # KNN match + Lowe's ratio test
        matches = self.matcher.knnMatch(self.anchor_desc, frame_desc, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            return None, len(good)

        # Extract matched point coordinates
        src_pts = np.float32([self.anchor_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None, len(good)

        inliers = int(mask.sum()) if mask is not None else 0
        return H, inliers


def transform_holds(holds: list[dict], H: np.ndarray) -> list[dict]:
    """Transform hold positions from anchor coords to frame coords using homography."""
    transformed = []
    for hold in holds:
        # Transform center point
        pt = np.float32([[hold["x"], hold["y"]]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pt, H)
        cx, cy = dst[0][0]

        # Transform a point at radius distance to estimate scaled radius
        pt_r = np.float32([[hold["x"] + hold["r"], hold["y"]]]).reshape(-1, 1, 2)
        dst_r = cv2.perspectiveTransform(pt_r, H)
        rx, ry = dst_r[0][0]
        r = ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5

        transformed.append({"x": int(cx), "y": int(cy), "r": int(r),
                            "id": hold["id"], "kind": hold.get("kind", "hand")})
    return transformed


TEST_LABELS = {
    "left": "LEFT: center > left > center",
    "left_right": "LEFT-RIGHT: center > right > fast left > center (full loss)",
    "up_down": "UP-DOWN: center > up > down > center (partial loss)",
    "zoom_in": "ZOOM-IN: left angle + phone zoom + movement",
    "all_around": "ALL-AROUND: movement + step in/out (route always visible)",
}


def _get_test_label(video_name: str) -> str:
    """Derive test type label from video filename."""
    name = video_name.lower()
    # Match longest prefix first to avoid "left" matching "left_right"
    for key in sorted(TEST_LABELS, key=len, reverse=True):
        if name.startswith(key):
            return TEST_LABELS[key]
    return video_name


RED = (0, 0, 255)
RED_DIM = (0, 0, 180)
RED_DARK = (0, 0, 100)


def _draw_scan_lines(img: np.ndarray, y_start: int, y_end: int, spacing: int = 3):
    """Draw horizontal scan lines for Terminator HUD effect."""
    overlay = img.copy()
    for y in range(y_start, y_end, spacing):
        cv2.line(overlay, (0, y), (img.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)


def _draw_bracket(img: np.ndarray, x: int, y: int, size: int, thickness: int = 2):
    """Draw angular targeting bracket."""
    s = size
    # Top-left corner
    cv2.line(img, (x, y), (x + s, y), RED, thickness)
    cv2.line(img, (x, y), (x, y + s), RED, thickness)
    # Top-right corner
    cv2.line(img, (x + size * 3, y), (x + size * 2, y), RED, thickness)
    cv2.line(img, (x + size * 3, y), (x + size * 3, y + s), RED, thickness)
    # Bottom-left corner
    cv2.line(img, (x, y + size * 3), (x + s, y + size * 3), RED, thickness)
    cv2.line(img, (x, y + size * 3), (x, y + size * 2), RED, thickness)
    # Bottom-right corner
    cv2.line(img, (x + size * 3, y + size * 3), (x + size * 2, y + size * 3), RED, thickness)
    cv2.line(img, (x + size * 3, y + size * 3), (x + size * 3, y + size * 2), RED, thickness)


def draw_overlay(frame: np.ndarray, holds: list[dict], tracking: bool,
                 match_count: int, frame_num: int, total_frames: int,
                 fps: float, test_label: str, tracked_frames: int) -> np.ndarray:
    """Draw hold circles and Terminator-style telemetry HUD on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Scale factor based on frame height (reference: 1080p)
    sf = h / 1080.0
    hud_h = int(160 * sf)  # HUD bar height

    if tracking:
        # Draw hold targeting brackets + circles
        overlay = out.copy()
        for hold in holds:
            center = (hold["x"], hold["y"])
            r = max(hold["r"], 5)

            # Red filled circle at low opacity
            cv2.circle(overlay, center, r, RED, -1)
            # Red outline
            cv2.circle(out, center, r, RED, 2)
            # Targeting crosshair
            cx, cy = hold["x"], hold["y"]
            cross = int(r * 0.4)
            cv2.line(out, (cx - cross, cy), (cx + cross, cy), RED_DIM, 1)
            cv2.line(out, (cx, cy - cross), (cx, cy + cross), RED_DIM, 1)
            # Hold label
            cv2.putText(out, hold["id"].upper(), (cx + r + 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sf, RED, max(1, int(sf)))

        cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)

    # --- TERMINATOR HUD (bottom) ---
    hud_top = h - hud_h

    # Dark translucent background
    hud_bg = out.copy()
    cv2.rectangle(hud_bg, (0, hud_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(hud_bg, 0.7, out, 0.3, 0, out)

    # Scan lines across HUD
    _draw_scan_lines(out, hud_top, h)

    # Top border line
    cv2.line(out, (0, hud_top), (w, hud_top), RED_DARK, 2)

    # Font sizes scaled to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_large = 0.8 * sf
    f_med = 0.6 * sf
    f_small = 0.5 * sf
    t_large = max(2, int(2 * sf))
    t_med = max(1, int(1.5 * sf))
    t_small = max(1, int(sf))

    margin = int(15 * sf)
    line1_y = hud_top + int(35 * sf)
    line2_y = hud_top + int(75 * sf)
    line3_y = hud_top + int(110 * sf)
    line4_y = hud_top + int(140 * sf)

    # Line 1: Test scenario label
    cv2.putText(out, f"// {test_label}", (margin, line1_y),
                font, f_med, RED_DIM, t_small)

    # Line 2: TRACKING / TARGET LOST — big and bold
    if tracking:
        cv2.putText(out, "TARGET LOCK", (margin, line2_y),
                    font, f_large, RED, t_large)
    else:
        # Flashing effect: alternate visibility based on frame number
        if frame_num % 10 < 7:
            cv2.putText(out, "TARGET LOST", (margin, line2_y),
                        font, f_large, (0, 0, 255), t_large)

    # Line 2 right side: match count with bar
    match_label = f"FEAT MATCH: {match_count:03d}"
    text_size = cv2.getTextSize(match_label, font, f_med, t_med)[0]
    cv2.putText(out, match_label, (w - text_size[0] - margin, line2_y),
                font, f_med, RED_DIM, t_med)

    # Match strength bar
    bar_x = w - int(280 * sf)
    bar_w = int(120 * sf)
    bar_h_px = int(14 * sf)
    bar_y = line2_y - bar_h_px + int(2 * sf)
    bar_fill = min(match_count / 100.0, 1.0)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), RED_DARK, 1)
    cv2.rectangle(out, (bar_x + 1, bar_y + 1),
                  (bar_x + 1 + int((bar_w - 2) * bar_fill), bar_y + bar_h_px - 1), RED, -1)

    # Line 3: Frame / Track % / FPS
    track_pct = (tracked_frames / frame_num * 100) if frame_num > 0 else 0
    cv2.putText(out, f"FRM {frame_num:05d}/{total_frames:05d}", (margin, line3_y),
                font, f_small, RED_DIM, t_small)
    cv2.putText(out, f"TRK {track_pct:5.1f}%", (int(300 * sf), line3_y),
                font, f_small, RED_DIM, t_small)
    cv2.putText(out, f"PROC {fps:5.1f} FPS", (int(520 * sf), line3_y),
                font, f_small, RED_DIM, t_small)

    # Line 4: Hold count
    cv2.putText(out, f"HOLDS: {len(holds):02d}", (margin, line4_y),
                font, f_small, RED_DARK, t_small)

    # Corner brackets (decorative)
    bsz = int(12 * sf)
    # Bottom-left
    cv2.line(out, (5, h - 5), (5 + bsz, h - 5), RED_DARK, 1)
    cv2.line(out, (5, h - 5), (5, h - 5 - bsz), RED_DARK, 1)
    # Bottom-right
    cv2.line(out, (w - 5, h - 5), (w - 5 - bsz, h - 5), RED_DARK, 1)
    cv2.line(out, (w - 5, h - 5), (w - 5, h - 5 - bsz), RED_DARK, 1)

    return out


def process_video(video_path: Path, tracker: AnchorTracker, holds: list[dict],
                  output_dir: Path) -> dict:
    """Process a single video and write output with overlay."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / f"tracked_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    test_label = _get_test_label(video_path.stem)
    print(f"\nProcessing: {video_path.name} ({w}x{h}, {fps:.0f}fps, {total_frames} frames)")
    print(f"  Test: {test_label}")

    frame_num = 0
    tracked_frames = 0
    total_matches = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Find homography
        H, match_count = tracker.find_homography(frame)
        total_matches += match_count

        if H is not None:
            transformed = transform_holds(holds, H)
            tracked_frames += 1
        else:
            transformed = []

        elapsed = time.time() - start_time
        proc_fps = frame_num / elapsed if elapsed > 0 else 0
        out = draw_overlay(frame, transformed, H is not None, match_count,
                           frame_num, total_frames, proc_fps,
                           test_label, tracked_frames)

        writer.write(out)

        if frame_num % 30 == 0:
            elapsed = time.time() - start_time
            speed = frame_num / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_num}/{total_frames} | "
                  f"Matches: {match_count} | "
                  f"Speed: {speed:.1f} fps")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    track_pct = (tracked_frames / frame_num * 100) if frame_num > 0 else 0
    avg_matches = total_matches / frame_num if frame_num > 0 else 0

    stats = {
        "video": video_path.name,
        "frames": frame_num,
        "tracked_frames": tracked_frames,
        "track_percentage": round(track_pct, 1),
        "avg_matches": round(avg_matches, 1),
        "processing_fps": round(frame_num / elapsed, 1) if elapsed > 0 else 0,
        "output": str(output_path),
    }

    print(f"  Done: {track_pct:.1f}% tracked ({tracked_frames}/{frame_num}), "
          f"avg matches: {avg_matches:.1f}, "
          f"speed: {stats['processing_fps']} fps")
    print(f"  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Track bouldering route overlay on video")
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video file (or use --all)")
    parser.add_argument("--all", metavar="DIR", default=None,
                        help="Process all .mp4 files in directory")
    parser.add_argument("--anchor", default="simulation_data/for_annotation_20260222_204347.jpg",
                        help="Path to anchor image")
    parser.add_argument("--route", default="simulation_data/route.json",
                        help="Path to route JSON")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for tracked videos")
    parser.add_argument("--min-matches", type=int, default=15,
                        help="Minimum feature matches to accept homography")
    args = parser.parse_args()

    # Resolve paths
    anchor_path = Path(args.anchor)
    route_path = Path(args.route)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load anchor image
    anchor = cv2.imread(str(anchor_path))
    if anchor is None:
        print(f"Error: cannot load anchor image {anchor_path}")
        sys.exit(1)
    print(f"Anchor image: {anchor_path.name} ({anchor.shape[1]}x{anchor.shape[0]})")

    # Load route JSON
    if not route_path.exists():
        print(f"Error: route file not found at {route_path}")
        print("Run the annotate tool first: uv run annotate")
        sys.exit(1)

    route = json.loads(route_path.read_text())
    holds = route["holds"]
    print(f"Route: {len(holds)} holds loaded")

    # Initialize tracker
    tracker = AnchorTracker(anchor, min_matches=args.min_matches)

    # Collect videos to process
    if args.all:
        video_dir = Path(args.all)
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {video_dir}")
            sys.exit(1)
    elif args.video:
        videos = [Path(args.video)]
    else:
        print("Provide a video path or use --all <directory>")
        sys.exit(1)

    # Process all videos
    all_stats = []
    for video_path in videos:
        stats = process_video(video_path, tracker, holds, output_dir)
        if stats:
            all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"  {s['video']:40s} | tracked: {s['track_percentage']:5.1f}% | avg matches: {s['avg_matches']:5.1f}")

    # Save stats
    stats_path = output_dir / "tracking_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
