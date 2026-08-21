# Agent Note: Adapt DSH memory principles to a scientific repository

Status: implemented

## Problem

The v1 harness improved discoverability but repeated routing rules, made every session read current experiment material, kept one-time bootstrap machinery, and let a green structural verifier sound like scientific closure. The Codex repository-development method used to build DSH offers useful persistent-memory mechanisms, but DSH runtime design, multi-author PR workflow, bilingual site and product tooling would add noise here.

## Decision

Adopt four portable rules: standing instructions, on-demand Skills, durable Agent Notes and scientific evidence have separate owners; each fact has one owner; changes covered by the canonical `non-trivial` rule add or update a Note; active memory is maintainable while archived implemented Notes can be SHA-256 sealed.

The canonical trigger and its research exceptions live only in `.agents/notes/README.md`; standing entry points and Skills link there instead of restating the classification. Scientific method, Gate, stop, claim or authorization semantics still require a protocol revision in addition to a Note, while an implementation repair that restores an existing protocol does not rewrite that protocol.

Use task-conditioned navigation, advisory Unicode-character ceilings for accretion-prone standing docs, three routine repo Skills plus one archive-only Agent Notes Skill, a static experiment catalog, pointer-only current-state, and separate governance/scientific/persistence result axes. Historical catalog bindings are checked in their own scientific source commits rather than kept byte-identical in the current checkout. Release mapping records fixed-point publication status without repeating experiment membership. Ordinary Note authoring is owned by standing instructions and `.agents/notes/README.md`; semantic deduplication, supersession and frozen archive mechanics remain in the archive Skill, while scoped simplification has a separate evidence-gathering workflow. Agent Note status is stored in a DSH-style header, the filename records the first-proposed date, and Git records later history. Skill UI metadata is optional; when present it is validated without forcing every Skill to carry product metadata.

Scoped simplification starts from a bounded flow and traces real callers, scientific consumers and call sites before proposing net deletion. It permits reasonable, explainable light behavior differences covered by owning tests; duplicate representations are inspected only after drift, manual synchronization, missing readers or another concrete deletion signal appears. Broad searches use independent read-only subagents directly, with the main agent consolidating evidence. Strong candidates become proposed Notes, small candidates become tagged `FIXME`/`TODO`/`XXX` markers, and a pass does not implement its own functional proposals. Simplification is user- or task-triggered rather than an automatic milestone, pre-push or per-change gate.

An implemented Note states current shipped reality in present tense and follows only factual changes that it already states or needs to locate its decision. The verifier rejects proposal-era headings outside fenced examples but does not guess natural-language tense. Archived bodies remain frozen and are not retroactively checked against later active-body templates. Active implemented Notes have no minimum count; each remains active only while its rationale has future decision value.

Validation follows credible impact: owning tests and identifiable downstream tests are the norm; full pytest is reserved for changes whose effects cannot be narrowed, span multiple experiment families, or are explicitly requested.

Keep scientific stage order, Gate flow, source binding and provenance explicit where auditability requires them.

## Alternatives considered

- Copy DSH's full Note taxonomy, translations, site sync and PR workflow: rejected as product-development overhead.
- Create one Note per change: rejected because it becomes a changelog; existing owner updates satisfy the rule.
- Keep a generic Agent Note authoring Skill: rejected because it repeats the root trigger and Note format owner without adding a distinct mechanism; the simplification Skill instead owns a distinct caller/consumer audit.
- Use Git history alone after a Note is archived: rejected because an explicit offline seal makes frozen Note drift machine-checkable; the manifest is created lazily with the first real archive.
- Require exact behavioral equivalence or a mandatory duplicate-representation taxonomy: rejected because both add review work without proving scientific value; explicit governing-contract changes already have their own review path.
- Run simplification automatically at milestones, pre-push or for every change: rejected because scoped passes should support experiment progress, not become a universal gate.

## Consequences

Standing context stays small, while specialized workflows load only when needed. Durable behavior and process changes remain explainable without creating one Note per edit. Existing owners absorb only factual movement relevant to their decision, and repositories with no durable implemented decision need no placeholder Note. Scoped simplification can record high-value deletion work without forcing a candidate quota or changing scientific behavior during the audit. Archive sealing adds a small deterministic manifest only when an archive first exists; old bodies are frozen rather than reformatted. Scientific readiness remains outside governance verification.

Reconsider this split if task-conditioned routing consistently misses required evidence, Note updates become noisy, archive sealing obstructs legitimate recovery, or a real new workflow requires an additional Skill. Budget ceilings are advisory signals; malformed policy still fails, while an overage cannot block governance or experiment progress.

## Verification

The repo-local Skills are checked with `quick_validate.py`; the simplification workflow is also forward-tested from a fresh read-only context against a real registered flow. The verifier's default governance mode, advisory budget report, Python compilation, `tests/test_spec_system.py`, and `git diff --check` are the direct implementation checks. Tests cover source-commit catalog integrity, release mapping, lifecycle formats, zero active implemented Notes, active-only present-state structure, frozen archives, optional Skill metadata, links, budgets and Harness Git persistence behavior.

The Harness catalog binds the U3 pilot and retry8 execution definitions to their separate scientific fixed points without altering scientific code or artifacts. Git persistence is evaluated only by `--committed`; `--strict-history`, formal experiments and publication remain separate work.
