Regenerate the Guess the Grade game and push it live.

## Data source

All route metadata lives in `docs/game/routes.json`. Images live in `docs/game/images/`.

The JSON is an array of objects. Current fields:
- `img` (string, required) — filename in `images/` folder
- `grade` (string, required) — V-grade like "V4", "V6"
- `sit` (boolean, required) — true if sit start

New fields may be added over time — pass them through to the ROUTES array as-is.

## Steps

1. **Read** `docs/game/routes.json`. Validate every entry has `img` and `grade`. Check each `img` file exists in `docs/game/images/`.

2. **Build ROUTES array**: Convert the JSON into a JS array literal for embedding in HTML. Format each entry on its own line with 2-space indent, preserving all fields.

3. **Build GRADES array**: Collect unique grades, sort by numeric value (V4 < V5 < V6).

4. **Generate game HTML**: Read `docs/game/template.html` and replace:
   - `%%ROUTES_JSON%%` with the ROUTES array
   - `%%GRADES_JSON%%` with the GRADES array
   - `%%ROUTE_COUNT%%` with the number of routes
   - `%%GAME_VER%%` — read current version from `docs/game/index.html` (look for `v{number}` in the `game-ver` div), increment by 1. If not found, start at 1.

5. **Write** the result to `docs/game/index.html`.

6. **Commit and push**:
   - Stage `docs/game/` (index.html, routes.json, images/)
   - Commit with message: `Game: update Guess the Grade routes`
   - Push to remote

7. Report: route count, grades, sit start count, live URL `https://bursh-dev.github.io/beta_lens/game/`
