---
name: chart
description: Plan a chart from citation-safe Census evidence.
task_types: [artifact_chart]
---

Identify the required measures, units, categories, and source citations before creating a chart.
Build the smallest chart that answers the request. Use a bar chart for unordered categories, a
grouped or stacked bar chart only when multiple comparable series are present, and a line chart
only for a genuine ordered trend. Preserve the approved input rows exactly in the companion CSV.
Label axes and units, include a legend for multiple series, use a deterministic style and
`tight_layout`, and keep the numeric axis at zero unless a non-zero baseline is explicitly and
truthfully justified. Copy the supplied source manifest without modification.

## Known-good program

This program passes the executor's code policy and the backend's lineage validation for single-
and multi-series bar charts. Start from it and change only presentation (figure size, title,
axis wording). Keep the CSV line exactly as written. Do not call `.rename()`,
`matplotlib.use()`, build paths with `/`, or add a `__main__` guard.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
dataset = payload["dataset"]
frame = pd.DataFrame(dataset["rows"])
frame.to_csv("output/plotted-data.csv", index=False)

unit = dataset["units"].get("value", "")
labels = list(dict.fromkeys(row["label"] for row in dataset["rows"]))
series = list(dict.fromkeys(row.get("series") or "Value" for row in dataset["rows"]))
width = 0.8 / len(series)
fig, ax = plt.subplots(figsize=(8, 4.5))
for index, name in enumerate(series):
    values = [
        next(
            (
                row["value"]
                for row in dataset["rows"]
                if row["label"] == label and (row.get("series") or "Value") == name
            ),
            0,
        )
        for label in labels
    ]
    positions = [position + index * width for position in range(len(labels))]
    ax.bar(positions, values, width=width, label=name)
ax.set_xticks([position + width * (len(series) - 1) / 2 for position in range(len(labels))])
ax.set_xticklabels(labels)
ax.set_ylabel(unit)
ax.set_title(dataset["title"])
ax.set_ylim(bottom=0)
if len(series) > 1:
    ax.legend()
fig.tight_layout()
fig.savefig("output/chart.png", dpi=150)
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"], indent=2), encoding="utf-8"
)
```
