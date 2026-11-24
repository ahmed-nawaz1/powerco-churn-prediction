# Historical material - do not use as the active entry point

`main.py` is the student's unchanged original 600+ line script. Its SHA256 is
fe7978ac3a73bb82c4f496910f19fe566205b89ea6141a8c88a4cdc3a3c5a0fb.
It runs at import time and expects data in the shell's current directory.
The original first chart is retained next to it.

`previous_pipeline/` preserves the first AI-assisted workflow, tools, tests and
README before restructuring. Historical outputs and source snapshots are under
`artifacts/runs/validated_20261117`; earlier ZIP/PDF iterations and visual QA
are under `artifacts/archive/`. Do not run these historical copies for new work.

The active root `main.py` launches `powerco.cli`, which orchestrates the current
modular package. Historical run manifests were not rewritten. Their validators
explicitly map relocated inputs and verify archived executed source when current
source differs. This distinction preserves history without pretending old results
were produced by the new source.
