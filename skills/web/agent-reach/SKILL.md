---
name: agent-reach
description: "Give Hermes read-only access to 10+ web platforms via CLI."
version: 0.1.0
author: Reachy (reachy), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Web, Scraping, Search, YouTube, GitHub, Reddit, Read-Only]
    related_skills: [youtube-content, grounded-citations, web-search]
---

# Agent Reach Skill

Agent Reach is a CLI tool installed in the reachy project's `.venv` that gives
Hermes read-only access to platforms it cannot normally reach: YouTube subtitles,
GitHub repos/issues, Reddit, V2EX, Bilibili, and web search (Exa/Jina).

**Read-only only.** No posting, commenting, liking, or messaging.

Full channel/action matrix: `references/channel-reference.md`.

## When to use

Use when the user asks to:
- Extract YouTube subtitles or video metadata
- Search or read a GitHub repository, issue, or PR
- Search the web via Exa semantic search (as a complement to `web_search`)
- Read any public web page as clean markdown (Jina Reader fallback)
- Browse V2EX topics, Bilibili videos, or trending content

**Do not use for:** writing/posting to any platform, authenticated/private
content, or platforms that require login (Reddit, LinkedIn, Xiaohongshu, Boss
Zhipin, Twitter, Facebook, Instagram) — these are installed but skipped until
explicitly requested with login configured.

## Prerequisites

Agent Reach is installed in the reachy project's `.venv`. The CLI entry point
is `.venv/bin/agent-reach` (relative to the project root). No API keys needed
for default channels. YouTube subtitles work without keys; audio transcription
(Groq/OpenAI Whisper) requires a key configured via `agent-reach configure
groq-key`.

On this machine the project root is `/Users/vec-datascience/Documents/git-projects/reachy-vec` — use the full path when invoking from outside the project directory.

## How to Run

All output is JSON to stdout. Use the `terminal` tool. Resolve the project root
first (see Prerequisites), then invoke `.venv/bin/agent-reach`.

### List available channels
```bash
.venv/bin/agent-reach channels
```

### YouTube — video info, subtitles, search
```bash
# Video metadata (title, channel, date, description)
.venv/bin/agent-reach youtube read "<url_or_id>"

# Full subtitle text (manual or auto captions)
.venv/bin/agent-reach youtube subtitle "<url_or_id>"

# Search YouTube videos
.venv/bin/agent-reach youtube search "<query>" [--limit N]
```

### GitHub — repos, issues, PRs
```bash
# Search repositories
.venv/bin/agent-reach github search "<query>" [--limit N]

# Read a repository (with README) or an issue/PR with comments
.venv/bin/agent-reach github read "<owner/repo or URL>"
```

### Web — search and page reading
```bash
# Semantic web search (Exa)
.venv/bin/agent-reach web search "<query>" [--limit N]

# Read any web page as clean markdown (Jina Reader, falls back to Exa)
.venv/bin/agent-reach web read "<url>"
```

### V2EX — Chinese developer community
```bash
# Today's hot topics
.venv/bin/agent-reach v2ex hot [--limit N]

# Topics in a specific node (e.g. python, jobs)
.venv/bin/agent-reach v2ex node "<node_name>" [--limit N]

# Read a topic with replies
.venv/bin/agent-reach v2ex read "<url_or_id>"
```

### Bilibili — Chinese video platform
```bash
# Search videos
.venv/bin/agent-reach bilibili search "<query>" [--limit N]

# Video info (title, uploader, stats)
.venv/bin/agent-reach bilibili read "<BV_id_or_url>"

# Trending videos
.venv/bin/agent-reach bilibili hot [--limit N]
```

### Read any URL (auto-selects the right channel)
```bash
.venv/bin/agent-reach read "<url>"
```

## Quick Reference

| Channel | Action | Use case |
|---------|--------|----------|
| `youtube` | `read` | Video metadata |
| `youtube` | `subtitle` | Subtitle text |
| `youtube` | `search` | Search videos |
| `github` | `search` | Search repos |
| `github` | `read` | Repo/README or issue/PR |
| `web` | `search` | Semantic web search (Exa) |
| `web` | `read` | Page as markdown (Jina) |
| `v2ex` | `hot` | Today's hot topics |
| `v2ex` | `node` | Topics in a node |
| `v2ex` | `read` | Topic with replies |
| `bilibili` | `search` | Search videos |
| `bilibili` | `read` | Video info |
| `bilibili` | `hot` | Trending videos |

## Procedure

1. **Invoke** the CLI via `terminal` with the full path to `.venv/bin/agent-reach`.
2. **Parse** the JSON output from stdout. Exit code 0 = success, 1 = error.
3. **Handle errors**: the JSON contains `error.code` and `error.message`. Common codes:
   - `BAD_INPUT` — malformed URL or missing required argument
   - `UPSTREAM_BROKEN` — upstream tool unavailable or returned no content
   - `RATE_LIMITED` — API rate limit hit (GitHub without token: 60/hour)
   - `NEED_SETUP` — required tool not installed
4. **Present** the result to the user in readable form.

## Pitfalls

- **YouTube JS runtime**: yt-dlp needs Node.js or deno for modern YouTube videos.
  If subtitles fail, check that Node.js is installed (`which node`).
- **GitHub rate limits**: 60 requests/hour without a token. Set `GH_TOKEN` env var
  for 500/hour. Configure via `agent-reach configure github-token`.
- **Exa free tier**: the hosted MCP has rate limits. If Exa fails, fall back to
  `web_search` or `web_extract` tools.
- **Jina anti-bot**: some sites return Cloudflare challenge pages via Jina. The
  tool detects this and falls back to Exa automatically.
- **All output is JSON** — never parse prose output. The CLI prints one JSON
  object per invocation to stdout; logs go to stderr.
- **Optional/login channels** (Reddit, LinkedIn, Twitter, Facebook, Instagram,
  Xiaohongshu, Boss Zhipin, Xueqiu, Xiaoyuzhou) are installed but require login
  or API keys. They will return `NEED_LOGIN` errors until configured.

## Verification

Run `agent-reach doctor` to check all channels:
```bash
.venv/bin/agent-reach doctor
```

Or live-test specific channels:
```bash
.venv/bin/agent-reach check
```
