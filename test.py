#!/usr/bin/env python3
import os
import argparse
import fnmatch
import json
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import sys
import html

GITHUB_API = "https://api.github.com"

@dataclass
class Node:
    name: str
    is_dir: bool
    children: Dict[str, "Node"] = field(default_factory=dict)

    def add_path(self, parts: List[str]):
        if not parts:
            return
        head, *tail = parts
        if head not in self.children:
            self.children[head] = Node(head, is_dir=bool(tail))
        # If later a directory gets added where a file existed, promote it
        if tail:
            self.children[head].is_dir = True
            self.children[head].add_path(tail)

def build_tree_from_local(root_path: str, ignore: List[str]) -> Node:
    root = Node(os.path.basename(os.path.abspath(root_path)) or root_path, True)

    def should_ignore(path_rel: str) -> bool:
        return any(fnmatch.fnmatch(path_rel, pattern) for pattern in ignore)

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        rel_dir = os.path.relpath(dirpath, root_path)
        # Normalize '.' to ''
        rel_dir = "" if rel_dir == "." else rel_dir

        # Filter ignored directories in-place (prevents walking them)
        dirnames[:] = [d for d in dirnames if not should_ignore(os.path.join(rel_dir, d))]

        # Add directories
        for d in dirnames:
            path_parts = list(filter(None, os.path.join(rel_dir, d).split(os.sep)))
            root.add_path(path_parts)

        # Add files
        for f in filenames:
            rel_file = os.path.join(rel_dir, f)
            if should_ignore(rel_file):
                continue
            path_parts = list(filter(None, rel_file.split(os.sep)))
            root.add_path(path_parts)
    return root

def build_tree_from_github(repo: str, branch: str, ignore: List[str], token) -> Node:
    """
    Uses the git/trees API with recursive=1 to pull the whole repo tree.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Hit git/trees endpoint. Passing branch name usually resolves to that commit.
    url = f"{GITHUB_API}/repos/{repo}/git/trees/{branch}?recursive=1"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        sys.stderr.write(f"GitHub API error {r.status_code}: {r.text}\n")
        sys.exit(1)

    data = r.json()
    if "tree" not in data:
        sys.stderr.write("Unexpected GitHub API response; missing 'tree'\n")
        sys.exit(1)

    root_name = repo.split("/")[-1]
    root = Node(root_name, True)

    def should_ignore(path_rel: str) -> bool:
        return any(fnmatch.fnmatch(path_rel, pattern) for pattern in ignore)

    for entry in data["tree"]:
        path = entry.get("path", "")
        if not path or should_ignore(path):
            continue
        parts = path.split("/")
        # Only mark dirs as dirs
        is_dir = entry.get("type") == "tree"
        # We add the path regardless; the add_path logic will mark dirs appropriately
        root.add_path(parts)

    return root

def render_ul(node: Node, indent: int = 0) -> str:
    """
    Render the children of `node` as a UL. The top-level caller should call this on the root node
    and optionally wrap it.
    """
    # Sort: dirs first, then files, then alphabetically
    def sort_key(n: Node):
        return (0 if n.is_dir else 1, n.name.lower())

    items = []
    for child in sorted(node.children.values(), key=sort_key):
        cls = "" if child.is_dir else ' class="file"'
        safe_name = html.escape(child.name)
        if child.is_dir and child.children:
            items.append(" " * indent + f"<li{cls}>{safe_name}\n" +
                         " " * indent + "  <ul>\n" +
                         render_ul(child, indent + 4) +
                         " " * indent + "  </ul>\n" +
                         " " * indent + "</li>\n")
        else:
            items.append(" " * indent + f"<li{cls}>{safe_name}</li>\n")
    return "".join(items)

def wrap_full_html(title: str, ul_content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; line-height: 1.4; }}
    ul {{ list-style-type: none; padding-left: 20px; margin: 0; }}
    li::before {{ content: "📁 "; }}
    li.file::before {{ content: "📄 "; }}
    h2 {{ margin-bottom: 0.5rem; }}
    .wrapper {{ max-width: 800px; margin: 2rem auto; }}
    code {{ background: #f5f5f5; padding: 0 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <h2>📦 {html.escape(title)}</h2>
    <ul>
{ul_content.rstrip()}
    </ul>
  </div>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Generate an HTML UL/LI file tree from a local or GitHub repo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local", type=str, help="Path to local repo")
    group.add_argument("--github", type=str, help="GitHub repo in form owner/name")
    parser.add_argument("--branch", type=str, default="main", help="Branch or commit-ish (GitHub mode)")
    parser.add_argument("--ignore", nargs="*", default=[], help="Glob patterns to ignore (e.g. .git node_modules/*)")
    parser.add_argument("--full-page", action="store_true", help="Wrap output in full HTML page")
    args = parser.parse_args()

    token = "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"

    if args.local:
        root = build_tree_from_local(args.local, args.ignore)
        title = os.path.basename(os.path.abspath(args.local)) or args.local
    else:
        root = build_tree_from_github(args.github, args.branch, args.ignore, token)
        title = args.github

    ul = render_ul(root, indent=4)
    if args.full_page:
        html_out = wrap_full_html(title, ul)
    else:
        html_out = f"<h2>📦 {html.escape(title)}</h2>\n<ul>\n{ul}</ul>"

    sys.stdout.write(html_out)

if __name__ == "__main__":
    main()
