# Agent Note: Integrate scientific source history and publish a review branch

Status: implemented

## Problem

The local Harness main and the active scientific lineage diverged after a common historical base. GitHub `main` separately represented a path-filtered runnable-code publication whose rewritten history intentionally omitted research documentation. Treating any of these refs as interchangeable would either hide the active implementation, rewrite a published snapshot without an explicit decision, or misidentify a merge commit as the source fixed point of an ongoing experiment.

## Decision

Local `main` is the integrated project source line. It contains the repository Harness and the scientific history through retry8 fixed point `73b28039ca0013693c0cc321a4ec006e63fefc26`; the merge does not change that fixed point or the attempt already running from its clean worktree. Registry definitions continue to name their original scientific source commits instead of treating the integrated main HEAD as experiment evidence.

The public `source-main` branch mirrors this integrated source line for repository and GPT-5 Pro review. Public GitHub `main` remains the filtered scientific publication at `76cd445e279e4d57f36759055915d135b8d31b2f` until a separate decision changes the default publication branch. `source-main` is not a scientific release, does not update the release map, and does not imply that any running attempt completed or passed a Gate.

Historical experiment branches remain available as fixed-point and provenance refs. A branch or worktree is removed only after checking its live-task ownership, unique commits and remaining evidentiary value; consolidation is not permission for bulk deletion.

## Alternatives considered

- Force-rewrite public `main` immediately: rejected because it would combine public-document exposure with a destructive default-branch transition before the full source mirror is independently checked.
- Squash retry8 into one current-state commit: rejected because it would erase the scientific lineage and make fixed-point provenance harder to audit.
- Publish only the Harness files: rejected because review of the Harness needs the actual scientific bindings and consumers it is meant to govern.
- Leave Harness and scientific development on permanently separate main lines: rejected because ordinary development would continue to require manual reconstruction of the current project.

## Consequences

The repository has one integrated local development line while retaining exact experiment commits as evidence identities. GitHub can expose the complete tracked source and documentation without moving the established filtered release. The public branch includes tracked research records, historical GPT-5 Pro questions and tracked robot inputs; ignored runtime outputs and untracked data remain outside Git.

Future default-branch replacement, branch deletion, release publication and scientific claim decisions remain separate operations. The integration merge and repository verifier can establish source and governance persistence; they cannot establish operational completion, scientific Gate passage, downstream authorization or formal claims.

## Verification

The integration is checked with the Harness verifier, its committed mode, retry8/V14.3 owning tests, `git diff --check`, Git object integrity, history-wide path and secret scans, and remote SHA closure. The active retry8 worktree, status pointer and process boundary are rechecked read-only after publication.
