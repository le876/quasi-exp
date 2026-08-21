# Scientific contract lifecycle

Scientific contract 规定一次实验允许如何执行：问题、方法、输入、Gate/stop 语义、产物和 claim boundary。可执行数值可以由 YAML config 或明确的 runner arguments/defaults 拥有；导航文档不复制它们。

## Lifecycle

- **proposed**：尚未成为可执行依据。
- **implemented**：对应实现与 binding 已存在；不等于 scientific pass 或 formal-ready。
- **rejected**：路线被明确拒绝，且理由仍能阻止现实的重复错误。
- **superseded**：历史内容保持原文和 provenance，并由后续 revision 取代。

新 formal 实验应在执行前建立独立 protocol 或明确的 repair addendum。已经执行过的 contract 不原地改变 method、threshold、confirmation set、stage order、stop/authorization semantics 或 claim boundary；新 evidence 使用新的 source fixed point 与 output root。
