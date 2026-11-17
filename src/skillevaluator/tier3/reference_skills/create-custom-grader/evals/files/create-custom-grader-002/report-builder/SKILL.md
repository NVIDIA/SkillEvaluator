---
name: report-builder
description: Use when turning a CSV of weekly sales into a Markdown report with totals per region.
---

# Report Builder

Turn a weekly sales CSV into `report.md`.

## Instructions

1. Read the CSV columns `region`, `product` and `units`.
2. Sum `units` per region and overall.
3. Write `report.md` with a `# Weekly sales` heading, a Markdown table of
   regions and their totals sorted by total (largest first), and a final
   `Total units: N` line.
