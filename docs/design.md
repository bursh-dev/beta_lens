# BetaLens — Design Document

## Overview

AR-like overlay system for bouldering routes. A trainer marks holds on a wall photo; students see those holds highlighted in real-time through their phone camera — **no physical markers or stickers**.

## System Architecture

### Two-Part System

| Component | Platform | Purpose |
|-----------|----------|---------|
| **Trainer App** | Web (phone browser) | Freeze camera frame, zoom/pan, circle holds, save route |
| **Student App** | Android (initially) | Load route, recognize wall, overlay holds in real-time |

### Core Concept: Photo as Anchor (Markerless)

The wall photo itself serves as the tracking anchor. All annotations are stored in **anchor-image coordinates** — not screen or frame coordinates — making them device-independent and reusable.

---

## Route Data Model

Shared JSON format between trainer and student apps.

```json
{
  "routeId": "r_2026_02_21_001",
  "name": "Blue Wall Compression",
  "gymId": "gym_01",
  "wallId": "sector_b_left",
  "anchor": {
    "imageUrl": "anchors/r_2026_02_21_001_anchor.jpg",
    "widthPx": 1920,
    "heightPx": 1080,
    "featureType": "ORB",
    "physicalWidthMeters": 2.4
  },
  "holds": [
    { "id": "h1", "x": 1042, "y": 154, "r": 62, "kind": "start" },
    { "id": "h2", "x": 699, "y": 306, "r": 78, "kind": "hand" },
    { "id": "h3", "x": 846, "y": 569, "r": 95, "kind": "finish" }
  ],
  "createdBy": "trainer_01",
  "createdAt": "2026-02-21T12:00:00Z",
  "version": 1
}
```

### Hold Annotation (MVP)

- Circle per hold: center `(x, y)` in anchor image pixels, radius `r`
- Optional: `kind` (start, hand, foot, finish), sequence number

### Coordinate Systems

1. **Screen/UI coordinates** — touch positions on zoomed/panned view
2. **Video frame coordinates** — pixel positions in current camera frame
3. **Anchor image coordinates** (canonical) — where all annotations are stored

Conversion chain (trainer): screen → invert zoom/pan → frame coords → inverse homography → anchor coords

---

## Tracking Approach (MVP)

Feature-based homography estimation using OpenCV:

1. Detect features in anchor photo (ORB / SIFT)
2. Detect features in live/video frame
3. Match features (BFMatcher / FLANN)
4. Estimate homography (RANSAC)
5. Transform hold coordinates: anchor → current frame
6. Draw overlay

This gives AR-like results without AR SDKs.

---

## Simulation Data

Test dataset for validating the tracking prototype before any phone work.

### Anchor Image
- `simulation_data/annotated_20260222_204347.jpg` — wall photo with red circle marks around route holds (~8-9 holds)
- Wall features: white wall, colorful holds, black diamond panel, green triangle volume — good texture variety for feature matching

### Test Videos

All filmed by the same phone, roughly same distance, same wall section.

| Video | Movement | Challenge Being Tested |
|-------|----------|----------------------|
| `left_*.mp4` | center → left → center | Partial route visibility, lateral shift |
| `left_right_*.mp4` | center → right → fast left → center | Complete loss of route area, fast motion |
| `up_down_*.mp4` | center → up → down → center | Vertical shift, partial loss |
| `zoom_in_*.mp4` | slightly left angle, phone zoom in, movements while zoomed | Scale change via digital zoom, angle offset |
| `all_around_*.mp4` | general movement, step in/out (physical zoom) | Natural movement, physical scale changes, route always visible |

### What Simulation Validates
- Anchor matching quality
- Tracking robustness (motion, angle, zoom, partial occlusion)
- Overlay stability
- Failure modes and recovery

### What Simulation Does NOT Validate
- Real-time phone performance
- ARCore behavior
- Battery/heat
- Mobile UX

---

## Tooling & Dependencies

- **Package manager:** `uv` (fast Python package manager)
- **Project config:** `pyproject.toml` only — no `requirements.txt` or `setup.py`

---

## Build Phases

### Phase 1 — Laptop Tracking Prototype
- Python + OpenCV
- Load anchor image + test videos
- Extract hold positions from annotated image (or manual JSON)
- Feature matching + homography per frame
- Draw overlay circles on video
- Evaluate stability across all 5 test scenarios
- **Output:** annotated video files + tracking quality metrics

### Phase 2 — Trainer Annotation UI (Web)
- Web app (phone browser)
- Camera preview → freeze frame
- Pinch zoom / pan
- Tap to add hold circles, resize, move, delete
- Undo/redo
- Save route JSON + anchor image

### Phase 3 — Phone Web Viewer (Real-Environment Test)
- Self-contained web app served over local network or hosted
- Phone opens URL in Chrome, runs everything on-device
- Opens rear camera via `getUserMedia`
- Loads anchor image + route JSON from server
- Runs OpenCV.js feature matching + homography on-device
- Draws overlay on `<canvas>` over camera feed
- Telemetry HUD (fps, match count, tracking status)
- Expected: 5-15 fps on mid-range phones — enough to validate tracking accuracy in real conditions
- No laptop needed at the gym

### Phase 4 — Native Android Student Viewer
- Kotlin Android app
- Camera overlay + CV tracking
- Possibly ARCore-assisted for smoother tracking

### Phase 5 — Backend (Optional)
- Route upload/download, user roles, route library
- Firebase / Supabase / PocketBase / custom API

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Wall appearance changes (chalk, lighting, hold swaps) | Multiple anchor images, distinctive anchor patches, re-anchor workflow |
| Marking precision on phone | Freeze frame, pinch zoom, magnifier lens |
| Occlusion (people in front) | Periodic re-detection, preserve last good transform, reacquire |
| Repetitive wall textures | Choose anchor regions with distinctive features |
| Real-time performance on phone | Start with laptop validation, optimize before mobile |

---

## Open Questions

1. Mark on frozen frame only, or also live-mark mode in MVP?
2. Circles only, or also polygons in MVP?
3. Web-based student MVP first, or straight to Android native?
4. File export/import first, or lightweight backend early?
5. How to organize gymId, wallId, sector naming?
6. Physical scale needed in MVP, or only 2D overlay alignment?
7. How to extract hold positions from the annotated image — manual JSON, or detect the red circles programmatically?
