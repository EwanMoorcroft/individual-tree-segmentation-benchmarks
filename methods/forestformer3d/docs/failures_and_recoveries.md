# ForestFormer3D failures and recoveries

## Recorded observations

- A foreground `srun` GPU probe was cancelled before allocation. It read no
  data and produced no runtime evidence. A detached replacement probe completed.
- The official inference entrypoint appears to use `torch.load` without an
  explicit `torch` import.
- Whole-plot inference is selected using the substring `test` in the input
  path. Development staging must not conceal this behaviour.
- Official inference reads reference arrays for output bookkeeping. Static
  inspection is insufficient to prove prediction independence.
- The authors' faster inference repository changes iteration count,
  configuration and post-processing and is not the unchanged default route.

## Recovery rules

Failed image builds retain their run root and logs. A retry uses a new run ID
and image path; existing evidence is not overwritten or deleted by a reusable
script. Dependency failure prevents validation submission from running.

Model-source defects are not repaired silently. If official interfaces cannot
produce row-aligned predictions without changing model logic, work stops with
the exact upstream block.
