# BACRA Workspace Inverse

This context describes how empirical workspace evidence becomes a branch-aware inverse representation without confusing finite search results with mathematical uniqueness or reachability proofs.

## Language

**Reach evidence**:
Evidence that a task-space cell contains, may contain, or is certified not to contain a reachable point. It does not assert that the whole cell is reachable.
_Avoid_: Reachability label, workspace truth

**Candidate family**:
A repeatable family of inverse solutions at one physical task probe, separated from other families in beta space and supported by neighboring continuation.
_Avoid_: Candidate cluster, unique solution

**Inverse section**:
A single-valued selection that assigns exactly one inverse candidate to every task probe in its domain.
_Avoid_: Product component, connected candidates

**Chart**:
An inverse section whose task domain is connected and whose selected edges, cycles, and multi-path endpoints satisfy the registered consistency policy.
_Avoid_: Product-graph component

**Primary section**:
The inverse section chosen to define one canonical label for each physical point used by a static inverse.
_Avoid_: Best branch, FK-selected branch

**Stitchable overlap**:
An overlap where two charts assign sufficiently close beta labels at the same physical probes to permit a continuous merge or switch.
_Avoid_: Shared bounding box

**Non-stitchable overlap**:
An overlap containing repeatable, simultaneously valid inverse families whose beta labels are too far apart for a stateless continuous switch.
_Avoid_: Router error, unsafe overlap

**Resolved single under budget**:
A task probe where the registered finite search found one stable candidate family. It is not a proof that no other inverse family exists.
_Avoid_: Safe single, unique inverse

**Supervision record**:
A unique labeled input-output record used by Student training, model selection, validation, or sealed evaluation.
_Avoid_: Candidate attempt, copied row

**Empirical workspace proxy**:
A finite cell representation derived from forward samples and active inverse probes. It is not an analytic description of the continuous reachable set.
_Avoid_: Reach set, exact workspace
