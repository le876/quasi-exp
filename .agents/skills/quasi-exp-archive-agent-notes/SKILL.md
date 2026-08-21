---
name: quasi-exp-archive-agent-notes
description: Audit, deduplicate, supersede, and archive quasi-exp Agent Notes. Use when a new Note may overlap an active decision, or when reviewing, pruning, restoring, or SHA-256 sealing the Agent Note archive.
---

# Archive Quasi-exp Agent Notes

Keep the active decision corpus useful without erasing evidence. Read the [Agent Note rules](../../notes/README.md) before classifying anything; code, protocol, config, tests and current authority determine whether a Note still owns a decision.

## Check supersession

Every new Note triggers a scoped search for active Notes covering the same decision, mechanism or rejected alternative.

- **No overlap:** keep the existing owner or create the new Note as appropriate.
- **Partial supersession:** retain both Notes and add relative Markdown links in both directions. State exactly which decision remains owned by each.
- **Full supersession:** first preserve every unique rationale, alternative, consequence and named coverage gap in the current owner. Then consolidate or archive the old implemented Note and repair active inbound links.

Do not infer supersession from age, filename similarity or shared vocabulary alone.

## Classify future value

- Keep an implemented Note active while its rationale, alternatives, ownership boundary, negative guarantee, scientific claim boundary or reintroduction condition can guide a future change.
- Archive an implemented Note when the decision is complete and current owners make its implementation facts obvious, while its historical rationale remains worth retaining.
- Never archive a proposed Note. Reject it when the proposal is deliberately declined.
- Keep a rejected Note only while it prevents a plausible repeated mistake; otherwise delete it after repairing inbound links.

Do not archive toward a quota or use length as the decision rule.

## Archive one implemented Note

1. Confirm the Note is implemented and no longer active authority.
2. Move it to `archived/` without changing its body. Insert `Archived: YYYY-MM-DD` immediately after `Status: implemented`.
3. Repair active inbound links. Do not validate or rewrite outbound links or reformat the frozen body against newer active-Note rules.
4. Run the verifier with `--seal-archive`. The first real archive creates the manifest; do not create an empty manifest before then, and never hand-edit an existing path or hash.
5. Run the default verifier and `git diff --check`.

After sealing, do not edit, move or delete the archived file. Report Notes kept, consolidated, archived, rejected or deleted, plus any borderline case and the evidence used.

This Skill does not decide scientific truth. A method, Gate, authorization or claim change still requires a protocol revision or repair addendum; archiving a Note cannot change that contract.
