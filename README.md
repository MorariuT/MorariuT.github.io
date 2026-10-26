# Tudor Morariu's personal site

A static portfolio with an i3wm-inspired layout: numbered workspaces, square tiled panels, page outlines, a live status bar, and the original charcoal and green palette.

## Preview locally

```sh
python3 -m http.server 8000
```

Open http://localhost:8000. No packages or build step are required.

## Structure

- `style.css` defines the shared responsive layout and colors for every page.
- `js/site.js` handles workspace shortcuts (Alt+1–4), the page outline, the clock, and the homepage GitHub feed. The feed has an offline fallback.
- `js/game.js` contains the Galton board simulation, including restart handling and pointer coordinates for a scaled canvas.
- `Projects/` holds project explanations, code, media, the AI notebook catalog, and the game.
- `Misc/awards.html` holds the awards timeline.
- `index-new.html` and `Projects/AI_Work-new.html` retain their existing URLs and use the same design and full content as their main counterparts.
- `data/projects.js` and `js/render-projects.js` remain available for experiments with data-driven project rendering. Published pages contain their project content directly in HTML.

All page content and navigation work without JavaScript; the live repository feed, clock, keyboard shortcuts, and game require JavaScript. Project syntax highlighting optionally loads Prism from jsDelivr; code remains readable if it is unavailable.
