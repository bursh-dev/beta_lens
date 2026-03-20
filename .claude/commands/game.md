Regenerate the Guess the Grade game and push it live.

## Steps

1. **Scan images**: List all `.jpg` and `.png` files in `docs/game/images/`. Each file must be named with a grade prefix like `V4_something.jpg`. Extract the grade (everything before the first `_`).

2. **Validate**: If no images found or any filename doesn't match the `V{number}_*` pattern, stop and tell the user.

3. **Build ROUTES array**: Create a JSON array of `{ img: 'images/filename.jpg', grade: 'V5' }` for each image, sorted by filename.

4. **Build GRADES array**: Collect the unique grades from all images, sort them by numeric value (V4 < V5 < V6), output as a JSON array of strings.

5. **Generate game HTML**: Read `docs/game/template.html` and replace:
   - `%%ROUTES_JSON%%` with the ROUTES array (formatted on multiple lines, 2-space indent)
   - `%%GRADES_JSON%%` with the GRADES array
   - `%%ROUTE_COUNT%%` with the number of routes

6. **Write** the result to `docs/game/index.html`.

7. **Commit and push**:
   - Stage `docs/game/index.html` and `docs/game/images/`
   - Commit with message: `Game: update Guess the Grade routes`
   - Push to remote

8. Report what was generated: how many routes, which grades, and the live URL `https://bursh-dev.github.io/beta_lens/game/`
