# BACRA Workspace Inverse

This context describes how empirical workspace evidence becomes a branch-aware inverse representation without confusing finite search results with mathematical uniqueness or reachability proofs.

## Language

**Canonical layer field**:
A deterministic low-dimensional map from a configuration coordinate such as `u=(a,b,eta)` to one beta/theta family before forward kinematics and data selection. It is an empirical construction policy, not proof of a unique physical inverse.
_Avoid_: Complete inverse, unique branch

**Layer path**:
The registered coefficient functions that distribute one configuration coordinate across robot sections, such as `s1(eta)` and `s2(eta)`. Different paths are different candidate constructions, not interchangeable samples from one proven surface.
_Avoid_: Ground-truth deformation mode

**Geometric Gate**:
A finite filter on generated candidates, for example workspace, radial or Jacobian criteria. Passing it means only that the registered sample satisfies those checks.
_Avoid_: Reachability proof, formal authorization

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
