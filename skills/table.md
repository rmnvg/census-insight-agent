---
name: table
description: Plan a tabular artifact from citation-safe Census evidence.
task_types: [artifact_table]
---

Preserve row/column meaning, units, and a citation for every factual value. Artifact execution is
performed only after the evidence-derived dataset passes lineage validation. Use descriptive
columns, put units in headings or metadata, format comparable numbers consistently, retain explicit
missing values without imputation, preserve approved rows exactly in CSV, and produce a concise
readable Markdown representation. Copy the supplied source manifest without modification.

## Known-good program

This program passes the executor's code policy and the backend's lineage validation for every
table dataset. Start from it and change only presentation (title wording, Markdown columns). Keep
the CSV line exactly as written: it must preserve every approved row and column unchanged. Do not
call `.rename()`, build paths with `/`, or add a `__main__` guard.

```python
import json
from pathlib import Path

import pandas as pd

payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
dataset = payload["dataset"]
frame = pd.DataFrame(dataset["rows"])
frame.to_csv("output/table.csv", index=False)

unit = dataset["units"].get("value", "")
lines = [f"# {dataset['title']}", "", f"| Label | Series | Value ({unit}) |", "|---|---|---:|"]
for row in dataset["rows"]:
    lines.append(f"| {row['label']} | {row.get('series') or ''} | {row['value']:,} |")
Path("output/table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"], indent=2), encoding="utf-8"
)
```
