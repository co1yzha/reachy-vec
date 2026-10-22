"""Local read-only web dashboard: browse everything in the LanceDB store,
and watch a running `reachy-vec run` live.

Stdlib http.server only - no web framework. The page fetches /api/tables,
which re-reads the database on every request, so a browser refresh always
shows the current state (memories saved mid-conversation, banked voices...).
/live tails data/reachy.log through /api/log (polled), so it never touches
the running Oracle. The log holds everything people say near the robot -
keep the dashboard on localhost.
"""

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import typer

from reachy_vec.config import settings

TABLE_META = {
    "docs": ("knowledge base for RAG answers", "384-dim BGE"),
    "voices": ("speaker embeddings, enrolled + passively banked", "192-dim ECAPA"),
    "memories": ("per-person notes (save_note + distillation)", "384-dim BGE"),
    "greetings": ("per-person greeting cooldown", "no vector"),
    "messages": ("queued relays, spoken when the recipient next talks to Reachy", "no vector"),
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
  .navlink { color: var(--accent); margin-left: 0.8rem; font-size: 0.85rem; }
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
    <a class="navlink" href="/live">Live conversation &rarr;</a>
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


LIVE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Reachy live</title>
<style>
  :root {
    --paper: #f6f7f6; --card: #ffffff; --ink: #20262b; --muted: #5c6670;
    --line: #dde2e0; --accent: #0f7a6c; --accent-soft: #e3f0ed;
    --you: #2a5d9f; --you-soft: #e2ebf7; --warn: #b0680f; --warn-soft: #f6ebdb;
    --bad: #b23a32; --bad-soft: #f7e1df;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #161b1e; --card: #1e2529; --ink: #e4e9e7; --muted: #93a09b;
      --line: #313a3f; --accent: #4cc2ae; --accent-soft: #12332e;
      --you: #84aee6; --you-soft: #1a2a40; --warn: #dfa155; --warn-soft: #3a2c17;
      --bad: #ef8a80; --bad-soft: #3d1f1c;
    }
  }
  * { box-sizing: border-box; }
  body {
    background: var(--paper); color: var(--ink); margin: 0;
    font-family: "Avenir Next", "Seravek", system-ui, sans-serif; line-height: 1.45;
  }
  header {
    position: sticky; top: 0; z-index: 1; background: var(--paper);
    border-bottom: 1px solid var(--line); padding: 0.9rem 1rem 0.7rem;
  }
  .wrap { max-width: 52rem; margin: 0 auto; }
  .top { display: flex; align-items: baseline; gap: 0.8rem; flex-wrap: wrap; }
  h1 { font-size: 1.2rem; font-weight: 600; margin: 0; }
  a { color: var(--accent); font-size: 0.85rem; }
  .status {
    display: flex; align-items: center; gap: 0.6rem; margin-top: 0.55rem;
    font-size: 0.9rem; flex-wrap: wrap;
  }
  .dot { width: 0.7rem; height: 0.7rem; border-radius: 50%; background: var(--muted); }
  .status.listening .dot { background: var(--accent); }
  .status.thinking .dot { background: var(--warn); animation: pulse 1s infinite; }
  .status.stuck .dot { background: var(--bad); }
  .status.stuck .state { color: var(--bad); font-weight: 600; }
  @keyframes pulse { 50% { opacity: 0.3; } }
  @media (prefers-reduced-motion: reduce) { .status.thinking .dot { animation: none; } }
  .state { font-weight: 600; }
  .mic {
    font-size: 0.8rem; border: 1px solid var(--line); border-radius: 999px;
    padding: 0.05em 0.7em; background: var(--card);
  }
  .since { color: var(--muted); font-variant-numeric: tabular-nums; }
  .controls { margin-left: auto; display: flex; gap: 0.4rem; }
  .controls button {
    font: inherit; font-size: 0.78rem; color: var(--accent); background: var(--accent-soft);
    border: 1px solid transparent; border-radius: 5px; padding: 0.15rem 0.65rem; cursor: pointer;
  }
  .controls button[aria-pressed="true"] { border-color: var(--accent); }
  main { padding: 0.8rem 1rem 3rem; }
  .ev {
    display: grid; grid-template-columns: 4.6rem 1fr; gap: 0.6rem;
    padding: 0.35rem 0; border-bottom: 1px solid var(--line); font-size: 0.88rem;
  }
  .ev .t { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem;
           color: var(--muted); padding-top: 0.15rem; }
  .ev .b { white-space: pre-wrap; overflow-wrap: anywhere; }
  .ev .k {
    font-size: 0.66rem; letter-spacing: 0.06em; text-transform: uppercase;
    border-radius: 999px; padding: 0.05em 0.55em; margin-right: 0.45rem;
    background: var(--accent-soft); color: var(--accent);
  }
  .ev .lat { color: var(--muted); font-size: 0.75rem; margin-left: 0.4rem; }
  .ev.heard .b { color: var(--you); font-weight: 600; }
  .ev.heard .k { background: var(--you-soft); color: var(--you); }
  .ev.reply .b { font-weight: 600; }
  .ev.warning .k, .ev.barge .k { background: var(--warn-soft); color: var(--warn); }
  .ev.error .k { background: var(--bad-soft); color: var(--bad); }
  .ev.error .b { color: var(--bad); font-family: ui-monospace, Menlo, monospace;
                 font-size: 0.76rem; }
  .ev.info .b, .ev.sleep .b { color: var(--muted); }
  .quiet .ev.info, .quiet .ev.sleep { display: none; }
  .empty { color: var(--muted); font-style: italic; font-size: 0.9rem; }
  @media (max-width: 30rem) { .ev { grid-template-columns: 1fr; gap: 0.1rem; } }
</style>
</head>
<body>
<header><div class="wrap">
  <div class="top">
    <h1>Reachy live</h1>
    <a href="/">&larr; Memory</a>
  </div>
  <div class="status" id="status" role="status" aria-live="polite">
    <span class="dot"></span><span class="state" id="state">connecting&hellip;</span>
    <span class="since" id="since"></span>
    <span class="mic" id="mic" title="microphone in use">mic: unknown</span>
    <span class="controls">
      <button id="quiet" type="button" aria-pressed="true">Conversation only</button>
      <button id="follow" type="button" aria-pressed="true">Follow</button>
    </span>
  </div>
</div></header>
<main><div class="wrap quiet" id="events">
  <p class="empty" id="empty">Waiting for <code>data/reachy.log</code> &mdash;
    start <code>reachy-vec run</code>.</p>
</div></main>
<script>
  const STUCK_S = 30;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
  const LABEL = { heard: "heard", route: "route", reply: "reply", action: "action",
    hermes: "hermes", wake: "wake", sleep: "sleep", barge: "barge-in",
    warning: "warn", error: "error", info: "log", mic: "mic" };
  const toDate = (t) => new Date(t.replace(" ", "T"));
  let offset = -1, lastHeard = null, lastEvent = null, state = "idle";
  const box = document.getElementById("events");

  function render(e) {
    let lat = "";
    if (lastHeard && ["route", "reply", "action", "hermes"].includes(e.kind)) {
      lat = `<span class="lat">+${((toDate(e.time) - lastHeard) / 1000).toFixed(1)}s</span>`;
    }
    const div = document.createElement("div");
    div.className = `ev ${e.kind}`;
    div.innerHTML = `<span class="t">${esc(e.time.slice(11))}</span>
      <span class="b"><span class="k">${LABEL[e.kind] || e.kind}</span>${esc(e.text)}${lat}</span>`;
    return div;
  }

  function track(e) {
    lastEvent = toDate(e.time);
    if (e.kind === "heard") { lastHeard = lastEvent; state = "thinking"; }
    else if (e.kind === "reply" || e.kind === "error") state = "listening";
    else if (e.kind === "wake") state = "listening";
    else if (e.kind === "sleep") state = "asleep";
    else if (e.kind === "mic") document.getElementById("mic").textContent = e.text;
  }

  function showStatus() {
    const el = document.getElementById("status");
    const since = lastEvent ? (Date.now() - lastEvent) / 1000 : 0;
    const stuck = state === "thinking" && since > STUCK_S;
    el.className = `status ${stuck ? "stuck" : state}`;
    document.getElementById("state").textContent = stuck
      ? `no reply for ${Math.round(since)} s - stuck?` : state;
    document.getElementById("since").textContent = lastEvent
      ? `last activity ${Math.round(since)} s ago` : "";
  }

  async function poll() {
    try {
      const data = await (await fetch(`/api/log?offset=${offset}`)).json();
      if (data.reset) { box.querySelectorAll(".ev").forEach((n) => n.remove()); }
      offset = data.offset;
      if (data.events.length) document.getElementById("empty")?.remove();
      for (const e of data.events) { box.appendChild(render(e)); track(e); }
      while (box.children.length > 600) box.firstElementChild.remove();
      if (data.events.length && follow.getAttribute("aria-pressed") === "true") {
        window.scrollTo(0, document.body.scrollHeight);
      }
    } catch (err) {
      document.getElementById("state").textContent = "dashboard disconnected";
    }
    showStatus();
  }

  const toggle = (btn, fn) => btn.addEventListener("click", () => {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", String(on)); fn(on);
  });
  const follow = document.getElementById("follow");
  toggle(follow, () => {});
  toggle(document.getElementById("quiet"), (on) => box.classList.toggle("quiet", on));
  poll();
  setInterval(poll, 1000);
</script>
</body>
</html>
"""

LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (reachy_vec\.\S+|\S+) "
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL) (.*)$"
)
# first match wins; anything else is "info", or its level for warnings/errors
KINDS = [
    ("heard", re.compile(r"^heard\b")),
    ("route", re.compile(r"^route \w+")),
    ("reply", re.compile(r"^reply to ")),
    ("action", re.compile(r"^(open_url|save_note|send_message|delivered message)\b")),
    ("hermes", re.compile(r"^hermes (task|result)")),
    ("wake", re.compile(r"^wake word")),
    ("sleep", re.compile(r"going to sleep|^interaction ended")),
    ("barge", re.compile(r"^barge-in")),
    ("mic", re.compile(r"^mic: ")),
]
INITIAL_TAIL_BYTES = 64_000  # first load shows roughly the last few conversations


def parse_log(text: str) -> list[dict]:
    """reachy.log lines -> events; traceback lines fold into the event above."""
    events: list[dict] = []
    for line in text.splitlines():
        match = LOG_LINE.match(line)
        if match is None:
            if events and line.strip():
                events[-1]["text"] += "\n" + line
            continue
        time, source, level, message = match.groups()
        kind = next((k for k, pattern in KINDS if pattern.search(message)), None)
        if level in ("ERROR", "CRITICAL"):
            kind = "error"
        elif kind is None:
            kind = "warning" if level == "WARNING" else "info"
        events.append({"time": time, "level": level, "kind": kind,
                       "source": source.removeprefix("reachy_vec."), "text": message})
    return events


def read_log(path: Path, offset: int) -> dict:
    """Events written since byte `offset` (-1 = start with the recent tail).
    Stops at the last complete line; restarts from 0 if the log shrank."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"offset": 0, "events": [], "reset": True}
    reset = offset < 0 or offset > size
    if offset < 0:
        offset = max(0, size - INITIAL_TAIL_BYTES)
    elif offset > size:
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        chunk = f.read(size - offset)
    end = chunk.rfind(b"\n") + 1   # hold back a half-written last line
    if reset and offset > 0:        # mid-file start: drop the partial first line
        first = chunk.find(b"\n") + 1
        chunk, end, offset = chunk[first:], end - first, offset + first
    events = parse_log(chunk[:end].decode("utf-8", errors="replace"))
    return {"offset": offset + end, "events": events, "reset": reset}


def render_page() -> str:
    return PAGE.replace("__META__", json.dumps(TABLE_META))


def make_server(store, host: str, port: int, log_path: Path | None = None
                ) -> ThreadingHTTPServer:
    """HTTP server serving the dashboard page, a live JSON dump of `store`,
    and the /live view of `log_path` (default data/reachy.log)."""
    log_path = log_path or settings.data_dir / "reachy.log"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                body = render_page().encode()
                content_type = "text/html; charset=utf-8"
            elif url.path == "/api/tables":
                body = json.dumps(store.dump_tables()).encode()
                content_type = "application/json"
            elif url.path == "/live":
                body = LIVE_PAGE.encode()
                content_type = "text/html; charset=utf-8"
            elif url.path == "/api/log":
                try:
                    offset = int(parse_qs(url.query).get("offset", ["-1"])[0])
                except ValueError:
                    offset = -1
                body = json.dumps(read_log(log_path, offset)).encode()
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
    """Browse the LanceDB store (docs, voices, memories...) and watch a running
    `reachy-vec run` live (/live) in the browser."""
    from reachy_vec.store.db import Store

    store = Store(settings.lancedb_dir)
    server = make_server(store, host, port)
    url = f"http://{host}:{port}"
    typer.echo(f"Dashboard at {url} (live conversation: {url}/live) - Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nBye.")
    finally:
        server.server_close()
