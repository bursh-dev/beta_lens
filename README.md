# BetaLens

AR-like overlay system for bouldering routes. Trainers mark holds on a wall photo, students see them highlighted in real-time through their phone camera — no physical markers or stickers needed.

## How It Works

### 1. Annotate (Phone or Laptop Browser)
- Open `betalens.html` → **Annotate Route**
- Load a photo of the climbing wall (camera or file)
- **Tap** to place holds, **pinch** to zoom, **drag** to pan
- **Long-press** a hold to resize or delete it
- **SAVE** stores the route (anchor image + hold positions) in localStorage

### 2. Live Tracking (Phone Browser)
- From the menu → **Live Tracking**
- Point your phone camera at the same wall
- OpenCV.js detects ORB features, computes homography, and overlays hold markers in real-time
- HUD shows FPS, match count, and tracking status

### Tech Stack
- **Frontend**: Single static HTML file (`src/web/betalens.html`) — vanilla JS, no build step
- **Computer Vision**: OpenCV.js (ORB feature detection + homography)
- **Storage**: localStorage (route JSON with base64 anchor image)
- **Python tools**: Desktop annotation (`annotate.py`) and tracking pipeline (`track.py`)

## Running on Your Phone

Camera access requires HTTPS. Serve from your laptop:

```bash
# 1. Generate self-signed cert (one time)
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

# 2. Start HTTPS server
python serve.py
# Serves on https://0.0.0.0:9443

# 3. Find your laptop's IP
ipconfig  # Windows — look for Wi-Fi adapter IPv4
```

On your phone (same WiFi), open in Chrome:
```
https://<laptop-ip>:9443/betalens.html
```
Tap through the "Not private" warning (Advanced → Proceed). Chrome will then allow camera access.

> **Why not just open the HTML file directly?**
> `getUserMedia` (camera API) requires a secure context: `https://` or `localhost`.
> Opening via `file://` is blocked on Android Chrome, and `content://` URLs (from Telegram/WhatsApp) don't qualify.

## Project Structure

```
src/
  web/
    betalens.html    # Main app — annotation + live tracking
  beta_lens/
    track.py         # Python tracking pipeline (desktop)
    annotate.py      # Python annotation tool (desktop)
docs/
  design.md          # Architecture and design document
serve.py             # HTTPS dev server for phone testing
simulation_data/     # Test images/video (gitignored)
```

## UI Design

Terminator HUD aesthetic — red on black, monospace font, scan lines. Left side panel (52px) for controls, optimized for mobile. Cyan hold markers with glow effect.

## Future Plans / Ideas

### Short Term
- [ ] Test tracking at the gym — tune ORB parameters (feature count, match ratio, min matches) for real lighting conditions
- [ ] Multiple routes — save/load different routes on the same wall (route list UI)
- [ ] Route naming and metadata (grade, setter, date)
- [ ] Share routes between devices (QR code with route data, or URL-based sharing)

### Medium Term
- [ ] Hold types — differentiate start, hand, foot, finish holds with distinct colors/shapes
- [ ] Sequence mode — number holds to show climbing order, animate sequence
- [ ] Tracking stability — temporal smoothing, kalman filter for jitter reduction
- [ ] Offline support — service worker for PWA, cache OpenCV.js
- [ ] Better anchor image matching — SIFT (more robust than ORB) if performance allows

### Long Term
- [ ] Backend service (Firebase/Supabase) for route storage and sharing
- [ ] Gym integration — route database per gym, QR codes on walls linking to routes
- [ ] Native Android app for better camera performance and background tracking
- [ ] Multi-wall support — detect which wall you're looking at automatically
- [ ] Social features — share beta, comment on routes, video recording with overlay
- [ ] AI hold detection — auto-detect holds from wall photo using computer vision/ML

### Technical Debt
- [ ] Move from localStorage to IndexedDB for larger route storage
- [ ] Add error recovery for tracking loss (re-detect anchor)
- [ ] Responsive layout testing on various phone sizes
- [ ] Unit tests for coordinate transforms and feature matching
