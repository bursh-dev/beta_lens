# BetaLens 🧗

**See the beta. Climb the route.**

AR-like overlay system for bouldering routes. Mark holds on a wall photo, then see them highlighted in real-time through your phone camera — no physical markers, stickers, or apps to install. Just a browser.

> **Live demo:** [bursh-dev.github.io/beta_lens](https://bursh-dev.github.io/beta_lens/) · Current version: **v49**

---

## What It Does

| Annotate | Track Live |
|----------|-----------|
| Snap a photo of the wall, tap to place holds | Point your camera at the wall — holds appear in real-time |
| Color-coded by type: 🟢 start · 🔵 hand · 🟡 foot · 🔴 finish | ORB feature matching + homography overlay at ~30 FPS |
| Pinch to zoom, drag to pan, long-press to edit | HUD shows FPS, match count, tracking status |
| Grade routes V0–V16, name them, organize by session | Works on any phone with a browser and camera |

## Quick Start

**On your phone** — open the [live demo](https://bursh-dev.github.io/beta_lens/), tap **Annotate Route**, take a photo, and start placing holds. That's it.

**For live tracking** — annotate a route first, then switch to **Live Tracking** and point your camera at the same wall. OpenCV.js handles the rest.

## Features

- **Hold annotation** — tap to place, long-press to edit type/size/delete, pinch-zoom for precision
- **4 hold types** — start (green), hand (blue), foot (yellow), finish (red) with color-coded glow markers
- **Route grading** — V0 through V16, tap to cycle
- **Route library** — save multiple routes, reorder, rename, preview thumbnails
- **Session management** — "End Session" archives all active routes with auto-backup JSON download
- **Archive browser** — browse past sessions, restore individual routes
- **IndexedDB storage** — anchor images stored in IndexedDB (hundreds of MB), no more 5 MB localStorage limit
- **Export / Import** — self-contained JSON files with embedded images for sharing and backup
- **Live AR overlay** — real-time hold projection using ORB feature detection + homography
- **Zero install** — single HTML file, runs entirely in the browser, no server needed for annotation

## Running Locally (for camera access)

Camera requires HTTPS. Serve from your laptop on the same WiFi:

```bash
# Generate self-signed cert (one time)
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

# Start HTTPS server
python serve.py          # serves on https://0.0.0.0:9443

# On your phone, open:
# https://<laptop-ip>:9443/betalens.html
```

> Tap through Chrome's "Not private" warning (Advanced → Proceed). Camera access requires `https://` — `file://` won't work on Android.

## Tech Stack

| Layer | Tech |
|-------|------|
| **App** | Single static HTML — vanilla JS, zero dependencies, no build step |
| **Vision** | OpenCV.js — ORB feature detection, BFMatcher, `findHomography` |
| **Storage** | IndexedDB (images) + localStorage (metadata) |
| **Hosting** | GitHub Pages (`docs/index.html`) |
| **Dev server** | `serve.py` — Python HTTPS server with self-signed cert |

## Project Structure

```
src/web/betalens.html     ← the entire app (annotation + live tracking)
docs/index.html           ← GitHub Pages copy (synced from src/web)
docs/game/                ← Guess the Grade — bonus mini-game
serve.py                  ← local HTTPS dev server
src/beta_lens/track.py    ← Python tracking pipeline (desktop)
src/beta_lens/annotate.py ← Python annotation tool (desktop)
```

## Bonus: Guess the Grade

A fun [mini-game](https://bursh-dev.github.io/beta_lens/game/) — look at a bouldering route photo, guess the V-grade, see how close you get. Timed rounds with zoom/pan support.

---

*Built for climbers who want to share beta without chalk marks.*
