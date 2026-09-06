---
name: inconsistency_analysis
description: Check cited Census values using deterministic arithmetic.
task_types: [inconsistency_analysis]
---

Classify observations as confirmed inconsistency, apparent rounding, or insufficient information.
Every input and observation requires source citations. Use only the safe arithmetic helper.

To check a "components sum to a total" claim (e.g. Rural + Urban = Total), call the arithmetic
tool twice in sequence, once per real value pulled from cited evidence: first `sum` over the
component values, then `difference` between that sum's own result and the total value. Never call
`difference` directly on the two components — that computes their gap, not whether they add up to
the total, and produces a meaningless number. When writing the final derived claim, its
`derivation` (operation, operands, result, unit) must exactly match one specific tool call you
actually made; do not narrate a different or additional arithmetic step that was never executed as
a tool call, even if it seems like a natural way to phrase the finding; citation validation checks
this and rejects the whole answer on any mismatch. If the tool-call budget runs out before every
year or entity can be checked, report findings only for the years actually verified and say which
were not checked, rather than inventing results for the rest.
