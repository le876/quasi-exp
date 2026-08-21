# Research Harness activation audit — 2026-08-20

## Fixed points and scope

- Harness target worktree: `/mnt/ML_projects/quasi_exp`
- Branch/HEAD at activation: `canonical-layer-field-u3@ecce9be2f9b157e2270ad3eb9e4d3d7721257e13`
- Active scientific binding: `canonical_layer_field_u3_pilot_v1`
- Scientific source fixed point: `e94ddb2e45116e4a3786b1383dc58dee94f9bbe9`
- Observation: `2026-08-20T20:31:53+08:00`

The worktree already contains user-maintained changes and untracked research material. Activation adds repository-memory, navigation, Skills and governance verification without changing scientific code, configs, tests, artifacts, `data/`, `runs/`, tmux or experiment processes.

## Binding evidence

The governing record, method-source document, robot config, direct runner and three registered tests all exist at `e94ddb2…` and are byte-identical in the current worktree. The pilot has no separate launcher; execution is through the Python runner. Its report root is `runs/diagnostics/canonical_layer_field_u3_pilot_v1`, while generated datasets live under the runner-declared data path.

The YAML robot config predates embedded `experiment_id`, `output_root` and Markdown source inventory fields. Registry therefore owns the external experiment binding; the verifier checks embedded identity fields when they exist and does not require governance-only edits to the scientific config.

## Evidence boundary

The current report and datasets are existing unsealed artifacts. They remain useful empirical evidence inside their recorded domain, but the Harness does not label them formal, deployment-ready or sealed. `docs/current-state.md` only points to the report and caches no result boolean.

`整体项目原文.md` is empty. It cannot support new source facts. The BACRA retry8 implementation remains in its separate worktree and is not silently registered or copied into this branch.

## Harness decisions

- Root instructions route tasks conditionally instead of loading all experiment material every session.
- Registry, current pointer, standing docs, Agent Notes and scientific evidence have separate owners.
- Direct runners and honestly labelled unsealed artifacts are representable; optional Skill UI metadata and frozen archived Note bodies do not become blocking bureaucracy.
- Tests are selected from credible impact; full pytest is reserved for genuinely repo-wide effects or an explicit request.

The Harness files are present in the working tree but are not committed by this activation task. Governance structure can be checked locally; Git persistence must remain `not_checked` until a later authorized commit.

## Activation checks

- Three repo-local Skills: `quick_validate.py` passed.
- Harness verifier tests: `49 passed`.
- Registered canonical-layer-field tests: `7 passed`.
- Default verifier: `governance_structure: pass`, `scientific_contract: not_evaluated`, `persistence: not_checked`.
- Budget listing, Python compilation and `git diff --check`: passed.

No full pytest, strict-history audit, formal run, longrun, commit, merge, push or publication was performed.
