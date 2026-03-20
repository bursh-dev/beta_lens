Regenerate the Guess the Grade game and push it live.

## Image naming convention

- `V{grade}_{name}.jpg` — standing start (e.g. `V4_crimps.jpg`)
- `V{grade}_{name}_sit.jpg` — sit start (e.g. `V5_arete_sit.jpg`)

The grade is everything before the first `_`. Sit start is detected by `_sit` before the file extension.

## Steps

1. **Scan images**: List all `.jpg` and `.png` files in `docs/game/images/`. Extract grade and sit-start from filename.

2. **Validate**: If no images found or any filename doesn't match the `V{number}_*` pattern, stop and tell the user.

3. **Build ROUTES array**: Create a JSON array of `{ img: 'images/filename.jpg', grade: 'V5', sit: true }` for each image, sorted by filename. `sit` is `true` if filename contains `_sit` before the extension, `false` otherwise.

4. **Build GRADES array**: Collect the unique grades from all images, sort them by numeric value (V4 < V5 < V6), output as a JSON array of strings.

5. **Generate game HTML**: Read `docs/game/template.html` and replace:
   - `%%ROUTES_JSON%%` with the ROUTES array (formatted on multiple lines, 2-space indent)
   - `%%GRADES_JSON%%` with the GRADES array
   - `%%ROUTE_COUNT%%` with the number of routes
   - `%%GAME_VER%%` with current date as version (YYYY-MM-DD)

6. **Write** the result to `docs/game/index.html`.

7. **Commit and push**:
   - Stage `docs/game/index.html`, `docs/game/template.html`, and `docs/game/images/`
   - Commit with message: `Game: update Guess the Grade routes`
   - Push to remote

8. Report what was generated: how many routes, which grades, how many sit starts, and the live URL `https://bursh-dev.github.io/beta_lens/game/`
