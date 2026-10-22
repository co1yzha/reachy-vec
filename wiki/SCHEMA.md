# Wiki Schema

## Domain

**reachy-vec / Team Familiar** — the embodied team assistant on a Reachy Mini
robot. This wiki compiles knowledge about:

- the system architecture (Mac brain, thin robot body, LanceDB vector store)
- the models running at every pipeline step (face ID, speaker ID, STT, RAG, TTS)
- the run loop, per-turn identity, and the memory / messenger features
- configuration conventions, hardware bring-up (Phase 4), and the roadmap

It **complements** the repo's `docs/` (operational docs) and
`docs/superpowers/` (design specs + phase plans) — it does not replace them.
The repo files are the canonical, current source of truth; the wiki adds
synthesized, cross-referenced entity/concept pages and a query archive. When
they disagree, trust the repo docs and flag the page as stale.

## Conventions
- File names: lowercase, hyphens, no spaces (e.g. `oracle-loop.md`)
- Every wiki page starts with YAML frontmatter (see below)
- Use `[[wikilinks]]` to link between pages (minimum 2 outbound links per page)
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md` under the correct section
- Every action must be appended to `log.md`
- **Provenance markers:** on pages that synthesize 3+ sources, append
  `^[raw/docs/source-file.md]` at the end of paragraphs whose claims come from
  a specific source. Optional on single-source pages where `sources:` frontmatter
  is enough.
- **Repo-doc sources:** the immutable snapshots in `raw/docs/` are copies of
  repo files. To re-sync one, copy the current repo file over the snapshot and
  recompute its `sha256:` (drift = the repo doc changed since ingest).

## Frontmatter
  ```yaml
  ---
  title: Page Title
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  type: entity | concept | comparison | query | summary
  tags: [from taxonomy below]
  sources: [raw/docs/pipelines.md, ...]
  # Optional quality signals:
  confidence: high | medium | low        # how well-supported the claims are
  contested: true                        # set when the page has unresolved contradictions
  contradictions: [other-page-slug]      # pages this one conflicts with
  ---
  ```

### raw/ Frontmatter
Raw sources get a small frontmatter block so re-ingests can detect drift:
```yaml
---
source_path: docs/architecture.md   # repo path for local sources (source_url for web)
source_kind: repo-doc               # repo-doc | article | paper | transcript
ingested: YYYY-MM-DD
sha256: <hex digest of the body below the frontmatter>
---
```
`sha256:` is computed over the body only (everything after the closing `---`).

## Tag Taxonomy
Every tag on a page must appear here. Add a new tag here BEFORE using it.

- Hardware: `robot`, `sim`, `media-source`
- Perception/audio: `face-id`, `speaker-id`, `stt`, `tts`
- Brain: `llm`, `rag`, `lancedb`, `state-machine`, `memory`, `messaging`, `barge-in`, `tools`
- Architecture: `protocol`, `config`, `testing`
- Meta: `roadmap`, `gotcha`

## Page Thresholds
- **Create a page** when an entity/concept appears in 2+ sources OR is central to one source
- **Add to existing page** when a source mentions something already covered
- **DON'T create a page** for passing mentions or things outside the domain
- **Split a page** when it exceeds ~200 lines — break into sub-topics with cross-links
- **Archive a page** when fully superseded — move to `_archive/`, remove from index

## Entity Pages
One page per notable entity (hardware, models, subsystems). Include:
overview, key facts (dims, thresholds, where it runs, config knobs),
relationships via `[[wikilinks]]`, source references.

## Concept Pages
One page per concept (pipelines, the run loop, identity, features). Include:
definition, how it works (steps), current state, open questions, related links.

## Comparison Pages
Side-by-side analyses (e.g. STT backends, TTS backends). Include:
what is compared and why, a comparison table, verdict, sources.

## Update Policy
When new information conflicts with existing content:
1. Check the dates — newer sources generally supersede older ones
2. If genuinely contradictory, note both positions with dates and sources
3. Mark the contradiction in frontmatter: `contradictions: [page-name]`
4. Flag for review in the lint report
