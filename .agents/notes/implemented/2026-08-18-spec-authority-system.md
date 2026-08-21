# Agent Note: One authoritative owner for each specification fact

Status: implemented

## Problem

Scientific contracts already existed in configs, runners, tests, protocols, provenance and artifacts, but a new session lacked an explicit map from each fact to its authority. Versioned filenames and duplicated prose encouraged guessing the current experiment, conflating mutable status with evidence, and treating public release identity as scientific source identity.

## Decision

Keep one owner per fact class: a short session entry, stable Architecture and objective, a machine-readable static experiment catalog and release map, pointer-only current state, deterministic governance verification, repo-local workflows, and .agents/notes as the sole durable-rationale owner. Existing governing records and scientific protocols retain their provenance-bound paths and content.

Governance structure, scientific contract, and Git persistence are separate claims. Registry schema v3 stores definitions and upstream relationships, while current-state owns the current experiment, attempt and locators. Neither implies scientific pass or caches result conclusions. GPT-5 Pro question/answer pairs are advisory decision input; user-adopted scientific rules move into protocol before they govern execution.

## Alternatives considered

- Add another comprehensive experiment plan: rejected because it would duplicate executable contracts.
- Move and merge historical protocols immediately: rejected because provenance and source inventories bind their paths.
- Mirror the registry into a generated Markdown index: removed because it had no independent consumer and created synchronization work.
- Keep a parallel decision directory: rejected because two owners recreate ambiguity.

## Consequences

New sessions can load only the material relevant to their task. Repeated attempts reuse a definition without turning runtime history into catalog entries. Config or explicit runner parameters own executable values, artifacts own results, and consultation records preserve external interpretation without displacing either owner. A direct runner needs no artificial launcher. Structural checks stay lightweight, while runner semantics require separate scientific-contract review and focused tests.

Reconsider this owner map if the registry starts carrying current attempt or results, current-state starts carrying conclusions, consultation records become a second protocol, or a simpler map provides equivalent navigation and auditability.

## Verification

scripts/spec/verify_spec_system.py checks the authority structure and exposes separate scientific-contract and persistence axes. tests/test_spec_system.py covers registry, pointers, budgets, Skills, Agent Notes, archive sealing and Git persistence behavior. Historical source gaps remain visible through --strict-history.
