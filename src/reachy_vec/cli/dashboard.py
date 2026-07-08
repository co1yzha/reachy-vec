"""Local read-only web dashboard: browse everything in the LanceDB store.

Stdlib http.server only - no web framework. The page fetches /api/tables,
which re-reads the database on every request, so a browser refresh always
shows the current state (memories saved mid-conversation, banked voices...).
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import typer

from reachy_vec.config import settings

TABLE_META = {
    "docs": ("knowledge base for RAG answers", "384-dim BGE"),
    "people": ("face embeddings, one row per enrolled frame", "512-dim insightface"),
    "voices": ("speaker embeddings, enrolled + passively banked", "192-dim ECAPA"),
    "memories": ("per-person notes (save_note + distillation)", "384-dim BGE"),
    "greetings": ("per-person greeting cooldown", "no vector"),
    "messages": ("queued relays, spoken on next sighting", "no vector"),
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Reachy LanceDB dashboard</title>
<style>
  :root {
    --paper: #f6f7f6; --card: #ffffff; --ink: #20262b; --muted: #5c6670;
    --line: #dde2e0; --accent: #0f7a6c; --accent-soft: #e3f0ed;
    --warn: #b0680f; --warn-soft: #f6ebdb; --code-bg: #eef1ef;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #161b1e; --card: #1e2529; --ink: #e4e9e7; --muted: #93a09b;
      --line: #313a3f; --accent: #4cc2ae; --accent-soft: #12332e;
      --warn: #dfa155; --warn-soft: #3a2c17; --code-bg: #262e33;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--paper); color: var(--ink);
    font-family: "Avenir Next", "Seravek", system-ui, sans-serif;
    line-height: 1.5; margin: 0; padding: 2rem 1.25rem 4rem;
  }
  .wrap { max-width: 60rem; margin: 0 auto; }
  h1 { font-size: 1.45rem; font-weight: 600; margin: 0 0 0.2rem; }
  .sub { color: var(--muted); font-size: 0.9rem; margin: 0 0 1.6rem; }
  .sub button {
    font: inherit; font-size: 0.8rem; color: var(--accent);
    background: var(--accent-soft); border: none; border-radius: 5px;
    padding: 0.15rem 0.7rem; cursor: pointer; margin-left: 0.5rem;
  }
  code {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.82em;
    background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 4px;
  }
  nav.summary {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
    gap: 0.6rem; margin-bottom: 1.2rem;
  }
  nav.summary a {
    background: var(--card); border: 1px solid var(--line); border-radius: 6px;
    padding: 0.6rem 0.75rem; text-decoration: none; color: inherit; display: block;
  }
  nav.summary a:hover { border-color: var(--accent); }
  .t-name {
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); display: block;
  }
  .t-count {
    font-size: 1.5rem; font-weight: 600; line-height: 1.2;
    font-variant-numeric: tabular-nums;
  }
  .t-dim { font-size: 0.75rem; color: var(--muted); }
  .filterbar { margin: 0 0 1.8rem; }
  .filterbar input {
    width: 100%; padding: 0.55rem 0.8rem; font: inherit; font-size: 0.9rem;
    color: inherit; background: var(--card); border: 1px solid var(--line);
    border-radius: 6px;
  }
  section { margin-bottom: 2.2rem; }
  section h2 {
    font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent); border-bottom: 1px solid var(--line);
    padding-bottom: 0.35rem; margin: 0 0 0.6rem;
    display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
    cursor: pointer; user-select: none;
  }
  section h2:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  section h2 .caret {
    display: inline-block; margin-right: 0.4rem;
    transition: transform 0.15s; transform: rotate(90deg);
  }
  section.collapsed h2 .caret { transform: rotate(0deg); }
  section.collapsed .content { display: none; }
  @media (prefers-reduced-motion: reduce) { section h2 .caret { transition: none; } }
  section h2 .meta {
    color: var(--muted); text-transform: none; letter-spacing: 0; font-weight: 400;
  }
  .beyond-limit { display: none; }
  section.show-all .beyond-limit, .filtering .beyond-limit { display: revert; }
  button.show-more {
    font: inherit; font-size: 0.78rem; color: var(--accent);
    background: none; border: 1px dashed var(--line); border-radius: 6px;
    width: 100%; padding: 0.4rem; cursor: pointer; margin-top: 0.1rem;
  }
  button.show-more:hover { border-color: var(--accent); }
  .filtering button.show-more { display: none; }
  .row {
    background: var(--card); border: 1px solid var(--line); border-radius: 6px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.55rem;
  }
  .row .head {
    display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem;
    align-items: baseline; font-size: 0.8rem;
  }
  .row .head .who { font-weight: 600; }
  .row .head .id {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.75rem;
    color: var(--muted); overflow-wrap: anywhere;
  }
  .row .head .when {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem;
    color: var(--muted); margin-left: auto; white-space: nowrap;
  }
  .row .body {
    margin-top: 0.45rem; font-size: 0.88rem; white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .vec {
    margin-top: 0.4rem; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.7rem; color: var(--muted);
  }
  .chip {
    font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
    border-radius: 999px; padding: 0.1em 0.6em;
    background: var(--accent-soft); color: var(--accent);
  }
  .chip.passive { background: var(--warn-soft); color: var(--warn); }
  .chip.pending { background: var(--warn-soft); color: var(--warn); }
  .empty { color: var(--muted); font-size: 0.85rem; font-style: italic; }
  .hidden { display: none !important; } /* filter mismatch beats show-all/limit rules */
  .scroll { overflow-x: auto; }
  table {
    border-collapse: collapse; width: 100%; font-size: 0.82rem;
    background: var(--card); border: 1px solid var(--line);
  }
  th, td {
    text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid var(--line);
  }
  tr:last-child td { border-bottom: none; }
  th {
    font-size: 0.68rem; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--muted); font-weight: 600;
  }
  td { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.75rem; }
  td.name { font-family: inherit; font-size: 0.82rem; font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Reachy&rsquo;s memory</h1>
  <p class="sub">
    Live view of <code>data/lancedb</code> &middot; loaded <span id="stamp">&hellip;</span>
    <button id="reload" type="button">Reload</button>
  </p>
  <nav class="summary" id="summary" aria-label="Tables"></nav>
  <div class="filterbar">
    <input id="filter" type="search" aria-label="Filter rows"
           placeholder="Filter rows by any text&hellip;" />
  </div>
  <div id="sections"></div>
</div>
<script>
  const META = __META__;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
  const when = (iso) => iso ? iso.replace("T", " ").replace(/\\.\\d+\\+00:00$/, " UTC") : "";
  const vec = (v) => v ? `${v.dim}-dim [${v.head.join(", ")}, \\u2026]` : "";

  const card = (title, id, ts, body, vector, chips = "") => `
    <div class="row">
      <div class="head">
        <span class="who">${esc(title)}</span>${chips}
        <span class="id">${esc(id)}</span>
        <span class="when">${esc(when(ts))}</span>
      </div>
      <div class="body">${esc(body)}</div>
      ${vector ? `<div class="vec">${esc(vec(vector))}</div>` : ""}
    </div>`;

  const table = (headers, rows) => `
    <div class="scroll"><table>
      <thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((cells) => `<tr>${cells.join("")}</tr>`).join("")}</tbody>
    </table></div>`;
  const td = (v, cls = "") => `<td${cls ? ` class="${cls}"` : ""}>${v}</td>`;

  const RENDER = {
    docs: (rows) => rows.map((r) => card(
      r.source.replace(/^demo: /, ""), r.chunk_id, r.ingested_at, r.text, r.vector)).join(""),
    memories: (rows) => rows.map((r) => card(
      r.person_id, r.memory_id, r.created_at, r.text, r.vector)).join(""),
    people: (rows) => table(
      ["embedding_id", "name", "person_id", "vector", "created"],
      rows.map((r) => [
        td(esc(r.embedding_id)), td(esc(r.name), "name"), td(esc(r.person_id)),
        td(esc(vec(r.vector))), td(esc(when(r.created_at)))])),
    voices: (rows) => table(
      ["voice_id", "name", "source", "vector", "created"],
      rows.map((r) => [
        td(esc(r.voice_id)), td(esc(r.name), "name"),
        td(`<span class="chip ${esc(r.source)}">${esc(r.source)}</span>`),
        td(esc(vec(r.vector))), td(esc(when(r.created_at)))])),
    greetings: (rows) => table(
      ["person_id", "last_greeted"],
      rows.map((r) => [td(esc(r.person_id)), td(esc(when(r.last_greeted)))])),
    messages: (rows) => table(
      ["from", "to", "message", "created", "status"],
      rows.map((r) => [
        td(esc(r.from_name), "name"), td(esc(r.to_name), "name"),
        td(esc(r.text), "name"), td(esc(when(r.created_at))),
        td(r.delivered_at
          ? `<span class="chip">delivered</span>`
          : `<span class="chip pending">pending</span>`)])),
  };

  async function load() {
    const data = await (await fetch("/api/tables")).json();
    document.getElementById("stamp").textContent = new Date().toLocaleString();
    document.getElementById("summary").innerHTML = Object.keys(META).map((name) => `
      <a href="#${name}">
        <span class="t-name">${name}</span>
        <span class="t-count">${(data[name] || []).length}</span>
        <div class="t-dim">${esc(META[name][1])}</div>
      </a>`).join("");
    document.getElementById("sections").innerHTML = Object.keys(META).map((name) => {
      const rows = data[name] || [];
      const body = rows.length
        ? RENDER[name](rows)
        : `<p class="empty">No rows.</p>`;
      return `<section id="${name}">
        <h2 tabindex="0" role="button" aria-expanded="true">
          <span><span class="caret">\\u25B8</span>${name}</span>
          <span class="meta">${rows.length} row${rows.length === 1 ? "" : "s"}
          &middot; ${esc(META[name][0])}</span></h2>
        <div class="content">${body}</div></section>`;
    }).join("");
    setUpSections();
    applyFilter();
  }

  const ROW_LIMIT = 5;

  function setUpSections() {
    document.querySelectorAll("#sections section").forEach((section) => {
      const heading = section.querySelector("h2");
      const toggle = () => {
        const collapsed = section.classList.toggle("collapsed");
        heading.setAttribute("aria-expanded", String(!collapsed));
      };
      heading.addEventListener("click", toggle);
      heading.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });

      const rows = section.querySelectorAll(".row, tbody tr");
      if (rows.length <= ROW_LIMIT) return;
      rows.forEach((row, i) => { if (i >= ROW_LIMIT) row.classList.add("beyond-limit"); });
      const more = document.createElement("button");
      more.className = "show-more";
      more.type = "button";
      const label = () => section.classList.contains("show-all")
        ? `Show first ${ROW_LIMIT} only`
        : `Show all ${rows.length} rows`;
      more.textContent = label();
      more.addEventListener("click", () => {
        section.classList.toggle("show-all");
        more.textContent = label();
      });
      section.querySelector(".content").appendChild(more);
    });
  }

  function applyFilter() {
    const q = document.getElementById("filter").value.trim().toLowerCase();
    // while filtering, the row cap is lifted so every match is visible
    document.getElementById("sections").classList.toggle("filtering", q !== "");
    document.querySelectorAll(".row, tbody tr").forEach((el) => {
      el.classList.toggle("hidden", q !== "" && !el.textContent.toLowerCase().includes(q));
    });
  }

  document.getElementById("filter").addEventListener("input", applyFilter);
  document.getElementById("reload").addEventListener("click", load);
  load();
</script>
</body>
</html>
"""


def render_page() -> str:
    return PAGE.replace("__META__", json.dumps(TABLE_META))


def make_server(store, host: str, port: int) -> ThreadingHTTPServer:
    """HTTP server serving the dashboard page and a live JSON dump of `store`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                body = render_page().encode()
                content_type = "text/html; charset=utf-8"
            elif self.path == "/api/tables":
                body = json.dumps(store.dump_tables()).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 (http.server API)
            pass  # keep the terminal quiet

    return ThreadingHTTPServer((host, port), Handler)


def dashboard(
    port: int = typer.Option(8400, help="Port to serve on."),
    host: str = typer.Option("127.0.0.1", help="Bind address (localhost only by default)."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the dashboard in the browser."
    ),
) -> None:
    """Browse the LanceDB store (docs, people, voices, memories...) in the browser."""
    from reachy_vec.store.db import Store

    store = Store(settings.lancedb_dir)
    server = make_server(store, host, port)
    url = f"http://{host}:{port}"
    typer.echo(f"Dashboard at {url} - Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
    finally:
        server.server_close()
