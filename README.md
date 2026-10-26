# Tudor Morariu's personal site

A plain HTML and CSS portfolio with an i3wm-inspired layout: numbered navigation, tiled panels, page outlines, and the original charcoal and green palette. No JavaScript, package installation, server-side application, or build step is needed to view the published pages.

## Preview locally

Open `index.html` in a browser, or serve this directory with:

```sh
python3 -m http.server 8000
```

## Pages

- `style.css` defines the shared responsive layout.
- `Projects/AI_Work.html` lists the full AI catalog, with explanations for all 18 main entries and four gradient-descent subprojects.
- `Projects/ai/` contains 22 static explanations with workflow descriptions, illustrative code, implementation notes, source references, and previous/next navigation.
- Project explanations distinguish inspected notebook details from conceptual examples. Competition entries explain collections without inventing task-specific architectures or scores. The financial-inclusion entry records the unavailable notebook and dataset link.
- `Projects/proiect.html` presents an accessible, static Galton board illustration. The original simulation source remains downloadable, but is not executed by the site.
- `Misc/awards.html` contains the awards timeline.
- `index-new.html` and `Projects/AI_Work-new.html` preserve alternate URLs and match the main versions.

Native browser video controls work without site scripts. The homepage repository shelf is curated HTML, rather than a live API feed. Navigation and page outlines are regular links; the footer has no live clock or keyboard-shortcut claims.

Legacy JavaScript files in `js/` and `data/` remain as source artifacts. No HTML page loads them. All Python code displayed in the explanations is escaped text for reading; snippets do not execute in the browser.
